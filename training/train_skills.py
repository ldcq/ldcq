from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
import h5py
from utils.utils import get_dataset
import os
import pickle
import argparse

def train(model, optimizer, train_state_decoder):
	losses = []
	
	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]
		actions = data[:,:,model.state_dim:]
		if train_state_decoder:
			loss_tot, s_T_loss, a_loss, kl_loss, diffusion_loss = model.get_losses(states, actions, train_state_decoder)
		else:
			loss_tot, a_loss, kl_loss, diffusion_loss = model.get_losses(states, actions, train_state_decoder)
		model.zero_grad()
		loss_tot.backward()
		optimizer.step()
		# log losses
		losses.append(loss_tot.item())
		
	return np.mean(losses)

def test(model, test_state_decoder):
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []
	diffusion_losses = []

	with torch.no_grad():
		for batch_id, data in enumerate(test_loader):
			data = data.cuda()
			states = data[:,:,:model.state_dim]
			actions = data[:,:,model.state_dim:]
			if test_state_decoder:
				loss_tot, s_T_loss, a_loss, kl_loss, diffusion_loss  = model.get_losses(states, actions, test_state_decoder)
				s_T_losses.append(s_T_loss.item())
			else:
				loss_tot, a_loss, kl_loss, diffusion_loss  = model.get_losses(states, actions, test_state_decoder)
			# log losses
			losses.append(loss_tot.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())
			diffusion_losses.append(diffusion_loss.item() if train_diffusion_prior else diffusion_loss)

	if train_diffusion_prior:
		return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(diffusion_losses)
	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), None

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='antmaze-large-diverse-v2')
parser.add_argument('--beta', type=float, default=0.05)
parser.add_argument('--conditional_prior', type=int, default=1)
parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
parser.add_argument('--state_decoder_type', type=str, default='mlp')
parser.add_argument('--a_dist', type=str, default='normal')
parser.add_argument('--horizon', type=int, default=30)
parser.add_argument('--separate_test_trajectories', type=int, default=0)
parser.add_argument('--test_split', type=float, default=0.2)
parser.add_argument('--get_rewards', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50000)
parser.add_argument('--start_training_state_decoder_after', type=int, default=0)
parser.add_argument('--normalize_latent', type=int, default=0)
parser.add_argument('--append_goals', type=int, default=0)
args = parser.parse_args()

batch_size = 128

h_dim = 256
z_dim = args.z_dim
lr = args.lr#5e-5
wd = 0.0
H = args.horizon
stride = 1
n_epochs = args.num_epochs
test_split = args.test_split
a_dist = args.a_dist#'normal' # 'tanh_normal' or 'normal'
encoder_type = 'gru' # 'transformer' #'state_sequence'
state_decoder_type = args.state_decoder_type
policy_decoder_type = args.policy_decoder_type
load_from_checkpoint = False
per_element_sigma = True
start_training_state_decoder_after = args.start_training_state_decoder_after
train_diffusion_prior = False

beta = args.beta # 1.0 # 0.1, 0.01, 0.001
conditional_prior = args.conditional_prior # True

env_name = args.env_name

dataset_file = 'data/'+env_name+'.pkl'
with open(dataset_file, "rb") as f:
	dataset = pickle.load(f)

checkpoint_dir = 'checkpoints/'
states = dataset['observations'] #[:10000]
#next_states = dataset['next_observations']
actions = dataset['actions'] #[:10000]

N = states.shape[0]

state_dim = states.shape[1] + args.append_goals * 2
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

dataset = get_dataset(env_name, H, stride, test_split, get_rewards=args.get_rewards, separate_test_trajectories=args.separate_test_trajectories, append_goals=args.append_goals)

obs_chunks_train = dataset['observations_train']
action_chunks_train = dataset['actions_train']
if test_split>0.0:
	obs_chunks_test = dataset['observations_test']
	action_chunks_test = dataset['actions_test']

filename = 'skill_model_'+env_name+'_encoderType('+encoder_type+')_state_dec_'+str(state_decoder_type)+'_policy_dec_'+str(policy_decoder_type)+'_H_'+str(H)+'_b_'+str(beta)+'_conditionalp_'+str(conditional_prior)+'_zdim_'+str(z_dim)+'_adist_'+a_dist+'_testSplit_'+str(test_split)+'_separatetest_'+str(args.separate_test_trajectories)+'_getrewards_'+str(args.get_rewards)+'_appendgoals_'+str(args.append_goals)

experiment = Experiment(api_key = '', project_name = '')
#experiment.add_tag('noisy2')

model = SkillModel(state_dim,a_dim,z_dim,h_dim,horizon=H,a_dist=a_dist,beta=beta,fixed_sig=None,encoder_type=encoder_type,state_decoder_type=state_decoder_type,policy_decoder_type=policy_decoder_type,per_element_sigma=per_element_sigma, conditional_prior=conditional_prior, train_diffusion_prior=train_diffusion_prior, normalize_latent=args.normalize_latent).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

if load_from_checkpoint:
	PATH = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	E_optimizer.load_state_dict(checkpoint['E_optimizer_state_dict'])
	M_optimizer.load_state_dict(checkpoint['M_optimizer_state_dict'])

experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'z_dim':z_dim,
							'H':H,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'beta':beta,
							'env_name':env_name,
							'a_dist':a_dist,
							'filename':filename,
							'encoder_type':encoder_type,
							'state_decoder_type':state_decoder_type,
							'policy_decoder_type':policy_decoder_type,
							'per_element_sigma':per_element_sigma,
       						'conditional_prior': conditional_prior,
       						'train_diffusion_prior': train_diffusion_prior,
       						'test_split': test_split,
                            'separate_test_trajectories': args.separate_test_trajectories,
                            'get_rewards': args.get_rewards,
                            'normalize_latent': args.normalize_latent,
                            'append_goals': args.append_goals})

inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
if test_split>0.0:
	inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0,
	shuffle=True)
if test_split>0.0:
	test_loader = DataLoader(
		inputs_test,
		batch_size=batch_size,
		num_workers=0)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
for i in range(n_epochs):
	if test_split>0.0:
		test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_diffusion_loss = test(model, test_state_decoder = i > start_training_state_decoder_after)
		
		print("--------TEST---------")
		
		print('test_loss: ', test_loss)
		print('test_s_T_loss: ', test_s_T_loss)
		print('test_a_loss: ', test_a_loss)
		print('test_kl_loss: ', test_kl_loss)
		if test_diffusion_loss is not None:
			print('test_diffusion_loss ', test_diffusion_loss)
		print(i)
		experiment.log_metric("test_loss", test_loss, step=i)
		experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
		experiment.log_metric("test_a_loss", test_a_loss, step=i)
		experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
		if test_diffusion_loss is not None:
			experiment.log_metric("test_diffusion_loss", test_diffusion_loss, step=i)
		
		if test_loss < min_test_loss:
			min_test_loss = test_loss	
			checkpoint_path = os.path.join(checkpoint_dir,filename+'_best.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
		if test_s_T_loss < min_test_s_T_loss:
			min_test_s_T_loss = test_s_T_loss

			checkpoint_path = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
		if test_a_loss < min_test_a_loss:
			min_test_a_loss = test_a_loss

			checkpoint_path = os.path.join(checkpoint_dir,filename+'_best_a.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	loss = train(model, optimizer, train_state_decoder = i > start_training_state_decoder_after)
	
	print("--------TRAIN---------")
	
	print('Loss: ', loss)
	print(i)
	experiment.log_metric("Train loss", loss, step=i)

	if i % 50 == 0:
		checkpoint_path = os.path.join(checkpoint_dir,filename+'_'+str(i)+'_'+'.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
