import gym
import pickle
import numpy as np
from gym import spaces
from collections import defaultdict

class RandomWalk(gym.Env):
    def __init__(self, flip_prob = 0.0): # flip_prob adds stochasitcity to dynamics
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.max_episode_length = 200
        self.flip_prob = flip_prob
    
    def reset(self):
        self.position = 0
        self.steps = 0
        return np.array([self.position])
    
    def step(self, action):
        step_size = np.clip(action[0], -1.0, 1.0)  # clip the action to [-1, 1]
        self.position += np.random.choice([-1, 1], p=[self.flip_prob, 1 - self.flip_prob]) * step_size
        self.position = np.clip(self.position, -10, 10)  # clip the position to [-10, 10]
        
        reward = 1.0 * (self.position == -10 or self.position == 10) - 1.0 * (self.position > -10 and self.position < 10)
        self.steps += 1
        done = (self.steps == self.max_episode_length) or (self.position == -10) or (self.position == 10)

        return np.array([self.position]), reward, done, {}
    
class BehaviorPolicy():
    def __init__(self,):
        pass
    def __call__(self,):
        return np.random.choice([-1, 1], p=[0.5, 0.5]) * np.random.uniform(low=0.8, high=1.0, size=(1,))

if __name__ == "__main__":
    env = RandomWalk()
    policy = BehaviorPolicy()
    dataset = defaultdict(list)
    dataset['observations'] = []
    dataset['actions'] = []
    dataset['rewards'] = []
    dataset['terminals'] = []
    num_episodes = 100
    for _ in range(num_episodes):
        done = False
        state = env.reset()
        while not done:
            action = policy()
            next_state, reward, done, info = env.step(action)
            dataset['observations'].append(state)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(done)        
            state = next_state
    dataset['observations'] = np.expand_dims(np.array(dataset['observations']),1)
    dataset['actions'] = np.array(dataset['actions'])
    dataset['terminals'] = np.array(dataset['terminals'])
    dataset['rewards'] = np.array(dataset['rewards'])
    print(np.sum(dataset['rewards']), np.where(dataset['rewards']==1))
    with open('data/random_walk.pkl', 'wb') as handle:
        pickle.dump(dataset, handle)