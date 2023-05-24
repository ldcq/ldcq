from argparse import ArgumentParser
import os

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.skill_model import SkillModel
from utils.utils import get_dataset

def collect_data(args):
    if 'antmaze' in args.env:
        state_dim = 29 + args.append_goals * 2
        a_dim = 8
    elif 'kitchen' in args.env:
        state_dim = 60
        a_dim = 9
    elif 'maze' in args.env:
        state_dim = 4 + args.append_goals * 2
        a_dim = 2
    else:
        raise NotImplementedError

    skill_model_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)

    checkpoint = torch.load(skill_model_path)

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
    skill_model.load_state_dict(checkpoint['model_state_dict'])
    skill_model.eval()

    dataset = get_dataset(args.env, args.horizon, args.stride, 0.0, args.append_goals)

    obs_chunks_train = dataset['observations_train']
    action_chunks_train = dataset['actions_train']

    inputs_train = torch.cat([obs_chunks_train, action_chunks_train], dim=-1)

    train_loader = DataLoader(
        inputs_train,
        batch_size=args.batch_size,
        num_workers=0)

    states_gt = np.zeros((inputs_train.shape[0], state_dim))
    latent_gt = np.zeros((inputs_train.shape[0], args.z_dim))
    if args.save_z_dist:
        latent_std_gt = np.zeros((inputs_train.shape[0], args.z_dim))
    sT_gt = np.zeros((inputs_train.shape[0], state_dim))

    for batch_id, data in enumerate(train_loader):
        data = data.to(args.device)
        states = data[:, :, :skill_model.state_dim]
        actions = data[:, :, skill_model.state_dim:]

        start_idx = batch_id * args.batch_size
        end_idx = start_idx + args.batch_size
        states_gt[start_idx : end_idx] = data[:, 0, :skill_model.state_dim].cpu().numpy()
        sT_gt[start_idx: end_idx] = states[:, -1, :skill_model.state_dim].cpu().numpy()
        output, output_std = skill_model.encoder(states, actions)
        latent_gt[start_idx : end_idx] = output.detach().cpu().numpy().squeeze(1)
        if args.save_z_dist:
            latent_std_gt[start_idx : end_idx] = output_std.detach().cpu().numpy().squeeze(1)

    np.save('data/' + args.skill_model_filename[:-4] + '_states.npy', states_gt)
    np.save('data/' + args.skill_model_filename[:-4] + '_latents.npy', latent_gt)
    np.save('data/' + args.skill_model_filename[:-4] + '_sT.npy', sT_gt)
    if args.save_z_dist:
        np.save('data/' + args.skill_model_filename[:-4] + '_latents_std.npy', latent_std_gt)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--append_goals', type=int, default=0)
    parser.add_argument('--save_z_dist', type=int, default=0)

    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)

    args = parser.parse_args()

    collect_data(args)
