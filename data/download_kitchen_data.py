import gym
import numpy as np

import collections
import pickle

import d4rl
import pickle

datasets = []
env_name = 'kitchen'

for dataset_type in ['partial', 'mixed', 'complete']:
	name = f'{env_name}-{dataset_type}-v0'
	env = gym.make(name)
	dataset = env.get_dataset()
	with open(name+".pkl","wb") as f:
		pickle.dump(dataset,f)