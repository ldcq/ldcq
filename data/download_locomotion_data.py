import gym
import numpy as np

import collections
import pickle

import d4rl
import pickle

datasets = []
envs = ['halfcheetah','walker2d','hopper']

for env_name in envs:
	for dataset_type in ['medium','medium-replay','medium-expert','expert']:
		name = f'{env_name}-{dataset_type}-v2'
		env = gym.make(name)
		dataset = env.get_dataset()
		with open(name+".pkl","wb") as f:
			pickle.dump(dataset,f)