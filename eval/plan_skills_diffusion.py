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

ANTMAZE = plt.imread('img/maze-large.png')

def visualize_states(state_0, states, best_state):
    plt.imshow(ANTMAZE, extent=[-6, 42, -6, 30])
    plt.scatter(state_0[:, 0], state_0[:, 1], color='red')
    plt.scatter(states[:, 0], states[:, 1], color='yellow')
    plt.scatter(best_state[:, 0], best_state[:, 1], color='green')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def q_policy(diffusion_model,
        skill_model,
        state_0,
        goal_state,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        visualize,
        append_goals,
        dqn_agent,
    ):

    if append_goals:
      state_0 = torch.cat([state_0, goal_state],dim=1)

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

        max_idx = torch.argmax(q_vals[start_idx:end_idx])

        best_state[env_idx] = state[start_idx + max_idx, :state_dim]
        best_latent[env_idx] = latent[start_idx + max_idx]

    if visualize:
        p = mp.Process(target=visualize_states, args=(state_0.cpu().numpy(), state.cpu().numpy(), best_state.cpu().numpy()))
        p.start()
        p.join()

    return best_latent


def diffusion_prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        goal_state,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        visualize,
        append_goals,
        dqn_agent,
    ):

    if append_goals:
      state_0 = torch.cat([state_0, goal_state], dim=1)

    latent = diffusion_model.sample_extra((state_0 - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean
    return latent


def prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        goal_state,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        visualize,
        append_goals,
        dqn_agent,
    ):

    if append_goals:
      state_0 = torch.cat([state_0, goal_state], dim=1)

    latent, latent_prior_std = skill_model.prior(state_0)
    return latent


def greedy_policy(
        diffusion_model,
        skill_model,
        state_0,
        goal_state,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        visualize,
        append_goals,
        dqn_agent,
    ):
    
    state_dim = state_0.shape[1]
    if append_goals:
      state_0 = torch.cat([state_0,goal_state],dim=1)
    state = state_0.repeat_interleave(num_diffusion_samples, 0)

    latent_0 = diffusion_model.sample_extra((state - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean

    if args.state_decoder_type == 'autoregressive':
        state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), None, latent_0.unsqueeze(1), evaluation=True)
        state[:,:state_dim] = state_pred.squeeze(1)
    else:
        state[:,:state_dim], _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim], latent_0)

    for depth in range(1, planning_depth):
        latent = diffusion_model.sample_extra((state - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean

        if args.state_decoder_type == 'autoregressive':
            state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), None, latent.unsqueeze(1), evaluation=True)
            state[:,:state_dim] = state_pred.squeeze(1)
        else:
            state[:,:state_dim], _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim], latent)

    best_state = torch.zeros((num_parallel_envs, state_dim)).to(args.device)
    best_latent = torch.zeros((num_parallel_envs, latent_0.shape[1])).to(args.device)

    for env_idx in range(num_parallel_envs):
        start_idx = env_idx * num_diffusion_samples
        end_idx = start_idx + num_diffusion_samples

        cost = torch.linalg.norm(state[start_idx : end_idx][:, :2] - goal_state[env_idx], axis=1)

        min_idx = torch.argmin(cost)

        best_state[env_idx] = state[start_idx + min_idx, :state_dim]
        best_latent[env_idx] = latent_0[start_idx + min_idx]

    if visualize:
        p = mp.Process(target=visualize_states, args=(state_0.cpu().numpy(), state.cpu().numpy(), best_state.cpu().numpy()))
        p.start()
        p.join()

    return best_latent


def exhaustive_policy(
        diffusion_model,
        skill_model,
        state_0,
        goal_state,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        visualize,
        append_goals,
        dqn_agent,
    ):

    state_dim = state_0.shape[1]
    if append_goals:
      state_0 = torch.cat([state_0,goal_state],dim=1)
    state = state_0.repeat_interleave(num_diffusion_samples, 0)

    latent_0 = diffusion_model.sample_extra((state - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean

    if args.state_decoder_type == 'autoregressive':
        state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), None, latent_0.unsqueeze(1), evaluation=True)
        state[:,:state_dim] = state_pred.squeeze(1)
    else:
        state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), latent_0.unsqueeze(1))
        state[:,:state_dim] = state_pred.squeeze(1)

    for depth in range(1, planning_depth):
        state = state.repeat_interleave(num_diffusion_samples, 0)
        latent = diffusion_model.sample_extra((state - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps) * latent_std + latent_mean
        if args.state_decoder_type == 'autoregressive':
            state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), None, latent.unsqueeze(1), evaluation=True)
            state[:,:state_dim] = state_pred.squeeze(1)
        else:
            state_pred, _ = skill_model.decoder.abstract_dynamics(state[:,:state_dim].unsqueeze(1), latent.unsqueeze(1))
            state[:,:state_dim] = state_pred.squeeze(1)

    best_state = torch.zeros((num_parallel_envs, state_dim)).to(args.device)
    best_latent = torch.zeros((num_parallel_envs, latent_0.shape[1])).to(args.device)

    for env_idx in range(num_parallel_envs):
        start_idx = env_idx * (num_diffusion_samples ** planning_depth)
        end_idx = start_idx + (num_diffusion_samples ** planning_depth)

        cost = torch.linalg.norm(state[start_idx : end_idx][:, :2] - goal_state[env_idx], axis=1)

        min_idx = torch.argmin(cost)

        best_state[env_idx] = state[(env_idx * num_diffusion_samples ** planning_depth) + min_idx, :state_dim]
        best_latent[env_idx] = latent_0[(env_idx * num_diffusion_samples ** planning_depth) + min_idx // (num_diffusion_samples ** (planning_depth - 1))]

    if visualize:
        p = mp.Process(target=visualize_states, args=(state_0.cpu().numpy(), state.cpu().numpy(), best_state.cpu().numpy()))
        p.start()
        p.join()

    return best_latent


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
              planning_depth,
              exec_horizon,
              predict_noise,
              visualize,
              render,
              append_goals,
              dqn_agent=None,
              ):

    nearby_goals = 0

    with torch.no_grad():
        assert num_evals % num_parallel_envs == 0
        num_evals = num_evals // num_parallel_envs

        success_evals = 0

        for eval_step in range(num_evals):
            state_0 = torch.zeros((num_parallel_envs, state_dim)).to(args.device)
            goal_state = torch.zeros((num_parallel_envs, 2)).to(args.device)
            done = [False] * num_parallel_envs

            for env_idx in range(len(envs)):
                state_0[env_idx] = torch.from_numpy(envs[env_idx].reset())
                goal_state[env_idx][0] = envs[env_idx].target_goal[0]
                goal_state[env_idx][1] = envs[env_idx].target_goal[1]

            env_step = 0
            nearby_goal = 0

            while env_step < 1000:

                best_latent = policy(
                                diffusion_model,
                                skill_model,
                                state_0,
                                goal_state,
                                state_mean,
                                state_std,
                                latent_mean,
                                latent_std,
                                num_parallel_envs,
                                num_diffusion_samples,
                                extra_steps,
                                planning_depth,
                                predict_noise,
                                visualize,
                                append_goals,
                                dqn_agent,
                            )

                for _ in range(exec_horizon):
                    for env_idx in range(len(envs)):
                        if not done[env_idx]:
                            action = skill_model.decoder.ll_policy.numpy_policy(torch.cat([state_0[env_idx], goal_state[env_idx]]) if append_goals else state_0[env_idx], best_latent[env_idx])
                            new_state, reward, done[env_idx], _ = envs[env_idx].step(action)
                            success_evals += reward
                            state_0[env_idx] = torch.from_numpy(new_state)

                            if np.linalg.norm(state_0[0][:2].cpu().numpy() - np.array([envs[env_idx].target_goal[0], envs[env_idx].target_goal[1]])) < 5:
                                nearby_goal = 1

                            if render and env_idx == 0:
                                envs[env_idx].render()

                    env_step += 1
                    if env_step > 1000:
                        break

                if sum(done) == num_parallel_envs:
                    break

            nearby_goals += nearby_goal

            total_runs = (eval_step + 1) * num_parallel_envs
            print(f'Total successful evaluations: {success_evals} out of {total_runs} i.e. {success_evals / total_runs * 100}% and nearby goals {nearby_goals}')


def evaluate(args):
    state_dim = 29
    a_dim = 8

    skill_model = SkillModel(state_dim + args.append_goals * 2,
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

    diffusion_model = None
    dqn_agent = None

    if args.policy == 'greedy' or args.policy == 'exhaustive' or args.policy == 'q' or args.policy == 'diffusion_prior':
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

    if args.policy == 'prior':
        policy_fn = prior_policy
    elif args.policy == 'diffusion_prior':
        policy_fn = diffusion_prior_policy
    elif args.policy == 'greedy':
        policy_fn = greedy_policy
    elif args.policy == 'exhaustive':
        policy_fn = exhaustive_policy
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
              args.planning_depth,
              args.exec_horizon,
              args.predict_noise,
              args.visualize,
              args.render,
              args.append_goals,
              dqn_agent,
              )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--num_parallel_envs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--q_checkpoint_dir', type=str, default='q_checkpoints')
    parser.add_argument('--q_checkpoint_steps', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--policy', type=str, default='greedy') #greedy/exhaustive/q
    parser.add_argument('--num_diffusion_samples', type=int, default=50)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--planning_depth', type=int, default=3)
    parser.add_argument('--extra_steps', type=int, default=5)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--exec_horizon', type=int, default=10)

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
    parser.add_argument('--visualize', type=int, default=0)

    args = parser.parse_args()

    evaluate(args)
