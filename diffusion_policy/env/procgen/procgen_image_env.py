import gym
from gym import spaces
import numpy as np
from procgen import ProcgenGym3Env
from gym3 import ToGymEnv

class ProcgenImageEnv(gym.Env):
    """
    A Gym wrapper that uses ToGymEnv for standard API compliance.
    """
    def __init__(self, env_name, start_level=0, **kwargs):
        self.env_name = env_name
        self.kwargs = kwargs
        self.current_level = start_level
        
        self._gym_env = None
        self._create_env()

        self.n_actions = self._gym_env.action_space.n
        self.action_space = spaces.Box(
            low=np.float32(0),
            high=np.float32(self.n_actions - 1),
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32
            )
        })

    def _create_env(self):
        """Destroys and recreates the C++ binary locked to the current level."""
        raw_env = ProcgenGym3Env(
            num=1,
            env_name=self.env_name,
            start_level=self.current_level,
            num_levels=1,
            **self.kwargs
        )
        self._gym_env = ToGymEnv(raw_env)

    def seed(self, seed=None):
        """Intercepts the pipeline's seed call to force a C++ rebuild."""
        if seed is not None and seed != self.current_level:
            self.current_level = seed
            self._create_env()

        return [seed]

    def _format_obs(self, obs_dict):
        """Converts (H, W, C) uint8 -> (C, H, W) float32."""
        rgb = obs_dict["rgb"].astype(np.float32) / 255.0

        return {'image': np.transpose(rgb, (2, 0, 1))}

    def reset(self, **kwargs):
        obs = self._gym_env.reset(**kwargs)

        return self._format_obs(obs)

    def step(self, action):
        discrete_action = int(np.clip(np.round(action[0]), 0, self.n_actions - 1))
        obs, rew, done, info = self._gym_env.step(discrete_action)

        return self._format_obs(obs), rew, done, info

    def _format_obs(self, obs_dict):
        """Converts (H, W, C) uint8 -> (C, H, W) float32."""
        self.render_cache = obs_dict["rgb"]
        rgb = obs_dict["rgb"].astype(np.float32) / 255.0

        return {'image': np.transpose(rgb, (2, 0, 1))}

    def reset(self, **kwargs):
        obs = self._gym_env.reset(**kwargs)

        return self._format_obs(obs)

    def step(self, action):
        discrete_action = int(np.clip(np.round(action[0]), 0, self.n_actions - 1))
        obs, rew, done, info = self._gym_env.step(discrete_action)

        return self._format_obs(obs), rew, done, info

    def render(self, mode='rgb_array'):
        """Returns the raw (H, W, C) uint8 image for the VideoRecordingWrapper."""
        assert mode == 'rgb_array', "Only rgb_array rendering is supported"
        if self.render_cache is None:
             obs, _, _ = self._gym_env.observe()
             self.render_cache = obs["rgb"]
             
        return self.render_cache