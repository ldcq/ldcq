from argparse import ArgumentParser
import os

import numpy as np
import torch
import random
import gym
import d4rl
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel

import multiprocessing as mp


def q_policy(diffusion_model,
        skill_model,
        state_0,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        predict_noise,
        append_goals,
        dqn_agent,
    ):

    state_dim = state_0.shape[1]
    state = state_0.repeat_interleave(num_diffusion_samples, 0)
    latent,q_vals = dqn_agent.get_max_skills(state, is_eval=True)
    if args.state_decoder_type == 'autoregressive':
        state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), None, latent.unsqueeze(1), evaluation=True)
        state = state_pred.squeeze(1)
    else:
        state, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim], latent)

    best_state = torch.zeros((num_parallel_envs, state_dim)).to(args.device)
    best_latent = torch.zeros((num_parallel_envs, latent.shape[1])).to(args.device)

    for env_idx in range(num_parallel_envs):
        start_idx = env_idx * num_diffusion_samples
        end_idx = start_idx + num_diffusion_samples

        #print('q val', torch.max(q_vals[start_idx:end_idx]), torch.min(q_vals[start_idx:end_idx]))
        max_idx = torch.argmax(q_vals[start_idx:end_idx])

        best_state[env_idx] = state[start_idx + max_idx, :state_dim]
        best_latent[env_idx] = latent[start_idx + max_idx]

    return best_latent

def diffusion_prior_policy(diffusion_model,
        skill_model,
        state_0,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        predict_noise,
        append_goals,
        dqn_agent,
    ):

    state_dim = state_0.shape[1]

    latent = diffusion_model.sample_extra((state_0 - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean
    return latent


def prior_policy(diffusion_model,
        skill_model,
        state_0,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        predict_noise,
        append_goals,
        dqn_agent,
    ):

    state_dim = state_0.shape[1]

    latent, latent_prior_std = skill_model.prior(state_0)
    return latent


def eval_func(diffusion_model,
              skill_model,
              policy,
              envs,
              state_dim,
              state_mean,
              state_std,
              latent_mean,
              latent_std,
              num_evals,
              num_parallel_envs,
              num_diffusion_samples,
              extra_steps,
              exec_horizon,
              predict_noise,
              render,
              append_goals,
              dqn_agent=None,
              env_name=None):

    with torch.no_grad():
        assert num_evals % num_parallel_envs == 0
        num_evals = num_evals // num_parallel_envs

        scores = 0

        for eval_step in range(num_evals):
            state_0 = torch.zeros((num_parallel_envs, state_dim)).to(args.device)

            done = [False] * num_parallel_envs

            for env_idx in range(len(envs)):
                state_0[env_idx] = torch.from_numpy(envs[env_idx].reset())

            env_step = 0
            if 'kitchen' in env_name:
              total_steps = 280
            else:
              total_steps = 1000

            while env_step < total_steps:

                best_latent = policy(
                                diffusion_model,
                                skill_model,
                                state_0,
                                state_mean,
                                state_std,
                                latent_mean,
                                latent_std,
                                num_parallel_envs,
                                num_diffusion_samples,
                                extra_steps,
                                predict_noise,
                                append_goals,
                                dqn_agent,
                            )

                for _ in range(exec_horizon):
                    for env_idx in range(len(envs)):
                        if not done[env_idx]:
                            action = skill_model.decoder.ll_policy.numpy_policy(state_0[env_idx], best_latent[env_idx])
                            new_state, reward, done[env_idx], _ = envs[env_idx].step(action)
                            scores += reward

                            state_0[env_idx] = torch.from_numpy(new_state)

                            if render and env_idx == 0:
                                envs[env_idx].render()

                    env_step += 1
                    #print(env_step, scores)
                    if env_step > total_steps:
                        break

                if sum(done) == num_parallel_envs:
                    break

            total_runs = (eval_step + 1) * num_parallel_envs
            if 'kitchen' in env_name:
              print(f'Total score: {scores / 4.0} out of {total_runs} i.e. {scores / total_runs * 25}%')
            else:
              print(f'Total score: {scores} out of {total_runs}; Normalized Score: {envs[0].get_normalized_score(scores/total_runs)*100}')


def evaluate(args):
    env = gym.make(args.env)
    dataset = env.get_dataset()
    state_dim = dataset['observations'].shape[1]
    a_dim = dataset['actions'].shape[1]

    skill_model = None
    skill_model = SkillModel(state_dim,
                             a_dim,
                             args.z_dim,
                             args.h_dim,
                             args.horizon,
                             a_dist=args.a_dist,
                             beta=args.beta,
                             fixed_sig=None,
                             encoder_type=args.encoder_type,
                             state_decoder_type=args.state_decoder_type,
                             policy_decoder_type=args.policy_decoder_type,
                             per_element_sigma=args.per_element_sigma,
                             conditional_prior=args.conditional_prior,
                             ).to(args.device)

    skill_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename))['model_state_dict'])
    skill_model.eval()

    diffusion_model = None
    if not args.policy == 'prior':
        if args.append_goals:
          diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_gc_best.pt')).to(args.device)
        else:
          diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)

        diffusion_model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=state_dim + args.append_goals*2,
            y_dim=args.z_dim,
            drop_prob=None,
            guide_w=args.cfg_weight,
        )
        diffusion_model.eval()

    envs = [gym.make(args.env) for _ in range(args.num_parallel_envs)]

    if not args.append_goals:
      #state_all = np.load(os.path.join(args.dataset_dir, args.skill_model_filename[:-4] + "_states.npy"), allow_pickle=True)
      state_mean = 0#torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
      state_std = 1#torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

      #latent_all = np.load(os.path.join(args.dataset_dir, args.skill_model_filename[:-4] + "_latents.npy"), allow_pickle=True)
      latent_mean = 0#torch.from_numpy(latent_all.mean(axis=0)).to(args.device).float()
      latent_std = 1#torch.from_numpy(latent_all.std(axis=0)).to(args.device).float()
    else:
      #state_all = np.load(os.path.join(args.dataset_dir, args.skill_model_filename[:-4] + "_goals_states.npy"), allow_pickle=True)
      state_mean = 0#torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
      state_std = 1#torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

      #latent_all = np.load(os.path.join(args.dataset_dir, args.skill_model_filename[:-4] + "_goals_latents.npy"), allow_pickle=True)
      latent_mean = 0#torch.from_numpy(latent_all.mean(axis=0)).to(args.device).float()
      latent_std = 1#torch.from_numpy(latent_all.std(axis=0)).to(args.device).float()

    dqn_agent = None
    if args.policy == 'prior':
        policy_fn = prior_policy
    elif args.policy == 'diffusion_prior':
        policy_fn = diffusion_prior_policy
    elif args.policy == 'q':
      dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt')).to(args.device)
      dqn_agent.diffusion_prior = diffusion_model
      dqn_agent.extra_steps = args.extra_steps
      dqn_agent.target_net_0 = dqn_agent.q_net_0
      dqn_agent.target_net_1 = dqn_agent.q_net_1
      dqn_agent.eval()
      dqn_agent.num_prior_samples = args.num_diffusion_samples
      policy_fn = q_policy
    else:
        raise NotImplementedError

    eval_func(diffusion_model,
              skill_model,
              policy_fn,
              envs,
              state_dim,
              state_mean,
              state_std,
              latent_mean,
              latent_std,
              args.num_evals,
              args.num_parallel_envs,
              args.num_diffusion_samples,
              args.extra_steps,
              args.exec_horizon,
              args.predict_noise,
              args.render,
              args.append_goals,
              dqn_agent,
              args.env
              )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='kitchen-complete-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--num_parallel_envs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--q_checkpoint_dir', type=str, default='q_checkpoints')
    parser.add_argument('--q_checkpoint_steps', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--policy', type=str, default='q') #greedy/exhaustive/q
    parser.add_argument('--num_diffusion_samples', type=int, default=50)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--extra_steps', type=int, default=4)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--exec_horizon', type=int, default=30)

    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=30)

    parser.add_argument('--render', type=int, default=1)

    args = parser.parse_args()

    evaluate(args)
