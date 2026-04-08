import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill


class ObsNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, 
            env, 
            o_noise,
            seed=None
        ):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Box), "ObsNoiseWrapper only supports Box observation spaces"

        self.o_noise = o_noise
        self.obs_min, self.obs_max = env.observation_space.low, env.observation_space.high
        self.np_random = np.random.default_rng(seed)
    
    def observation(self, observation):
        observation += self.o_noise * self.np_random.normal(size=observation.shape)
        observation = np.clip(observation, self.obs_min, self.obs_max)
        return observation