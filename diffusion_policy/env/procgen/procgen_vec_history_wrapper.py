import numpy as np
from procgen import ProcgenGym3Env
from gym import spaces


class ProcgenVectorizedHistoryWrapper:
    """
    Wraps procgen's native vectorized env with batched observation/action history.

    Replaces the AsyncVectorEnv + MultiStepHistoryWrapper stack by using
    procgen's internal C++ parallelism instead of Python subprocesses.

    Output format matches what the policy expects:
        obs:         {'image': (N, T, 3, 64, 64)} float32 [0, 1]
        past_action: (N, T, 1)                    float32

    gym3 timing: act() submits actions, then observe() returns (rew, obs, first).
    first[i] is True when env i is at the start of a new episode (auto-reset just
    happened). The reward and obs correspond to the transition *into* that first
    frame, so first[i]=True is the done signal for the previous episode.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        start_level: int = 0,
        num_levels: int = 0,
        history_len: int = 8,
        n_action_steps: int = 1,
        max_episode_steps: int = 1000,
        no_op: float = 0.0,
        **procgen_kwargs,
    ):
        self.num_envs = num_envs
        self.history_len = history_len
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.no_op = no_op

        self._env = ProcgenGym3Env(
            num=num_envs,
            env_name=env_name,
            start_level=start_level,
            num_levels=num_levels,
            **procgen_kwargs,
        )

        self.n_actions = self._env.ac_space.eltype.n

        base_rgb_shape = self._env.ob_space["rgb"].shape
        # gym3 ob_space shape is (H, W, C)
        h, w, c = base_rgb_shape[0], base_rgb_shape[1], base_rgb_shape[2]

        self.action_space = spaces.Box(
            low=np.float32(0),
            high=np.float32(self.n_actions - 1),
            shape=(n_action_steps, 1),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            'obs': spaces.Dict({
                'image': spaces.Box(
                    low=0.0, high=1.0,
                    shape=(history_len, c, h, w),
                    dtype=np.float32,
                )
            }),
            'past_action': spaces.Box(
                low=np.float32(0),
                high=np.float32(self.n_actions - 1),
                shape=(history_len, 1),
                dtype=np.float32,
            ),
        })

        self._obs_buf = np.zeros((num_envs, history_len, c, h, w), dtype=np.float32)
        self._act_buf = np.full((num_envs, history_len, 1), no_op, dtype=np.float32)
        self._ep_steps = np.zeros(num_envs, dtype=np.int32)
        self._ep_rewards = np.zeros(num_envs, dtype=np.float32)
        self._completed_ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_chw(self, obs_dict) -> np.ndarray:
        """(N, H, W, C) uint8 -> (N, C, H, W) float32 [0, 1]"""
        rgb = obs_dict['rgb'].astype(np.float32) / 255.0
        return np.transpose(rgb, (0, 3, 1, 2))

    def _push_obs(self, new_obs: np.ndarray):
        self._obs_buf[:, :-1] = self._obs_buf[:, 1:]
        self._obs_buf[:, -1] = new_obs

    def _push_act(self, new_act: np.ndarray):
        self._act_buf[:, :-1] = self._act_buf[:, 1:]
        self._act_buf[:, -1] = new_act

    def _reset_history_for(self, env_mask: np.ndarray, current_obs: np.ndarray):
        """Flood-fill history for done envs using vectorized indexing."""
        if not env_mask.any():
            return
        # current_obs[env_mask]: (K, C, H, W) -> (K, T, C, H, W)
        self._obs_buf[env_mask] = (
            current_obs[env_mask][:, np.newaxis].repeat(self.history_len, axis=1)
        )
        self._act_buf[env_mask] = self.no_op

    def _record_episode_rewards(self, done: np.ndarray):
        """Store accumulated reward for each done env, then reset its counter."""
        for i in np.where(done)[0]:
            self._completed_ep_rewards[i].append(float(self._ep_rewards[i]))
        self._ep_rewards[done] = 0.0

    def _build_history(self) -> dict:
        return {
            'obs': {'image': self._obs_buf},
            'past_action': self._act_buf,
        }

    def _to_discrete(self, actions: np.ndarray) -> np.ndarray:
        """(N, 1) float32 -> (N,) int32 clamped to valid action range."""
        return np.clip(np.round(actions[:, 0]), 0, self.n_actions - 1).astype(np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Read the initial observation from the freshly created env (gym3 envs
        start ready without an explicit reset call) and flood-fill history.
        """
        _, obs_dict, _ = self._env.observe()
        first_obs = self._to_chw(obs_dict)

        self._obs_buf[:] = first_obs[:, np.newaxis].repeat(self.history_len, axis=1)
        self._act_buf[:] = self.no_op
        self._ep_steps[:] = 0
        self._ep_rewards[:] = 0.0
        self._completed_ep_rewards = [[] for _ in range(self.num_envs)]

        return self._build_history()

    def step(self, actions: np.ndarray):
        """
        Execute n_action_steps sub-steps and aggregate rewards/dones.

        Args:
            actions: (N, n_action_steps, 1) float32

        Returns:
            history:  dict — obs and past_action buffers
            rewards:  (N,) float32 — max reward over sub-steps
            dones:    (N,) bool    — any done over sub-steps
            infos:    list[dict]   — one per env (stub; gym3 has no per-env info)
        """
        assert actions.shape == (self.num_envs, self.n_action_steps, 1), \
            f"Expected ({self.num_envs}, {self.n_action_steps}, 1), got {actions.shape}"

        agg_rew = np.zeros(self.num_envs, dtype=np.float32)
        agg_done = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        for t in range(self.n_action_steps):
            act_t = actions[:, t, :]
            discrete = self._to_discrete(act_t)

            self._env.act(discrete)
            rew, obs_dict, first = self._env.observe()

            new_obs = self._to_chw(obs_dict)
            self._ep_steps += 1
            self._ep_rewards += rew

            truncated = self._ep_steps >= self.max_episode_steps
            # first[i]=True: env i just auto-reset after a natural episode end
            done = np.logical_or(truncated, first)

            self._push_obs(new_obs)
            self._push_act(act_t)

            agg_rew = np.maximum(agg_rew, rew)
            agg_done = np.logical_or(agg_done, done)

            if done.any():
                self._record_episode_rewards(done)
                self._reset_history_for(done, new_obs)
                self._ep_steps[done] = 0

            if agg_done.all():
                break

        return self._build_history(), agg_rew, agg_done, infos

    def get_episode_rewards(self) -> list[list[float]]:
        """Return per-env list of cumulative episode rewards collected so far."""
        return [list(r) for r in self._completed_ep_rewards]

    def episodes_completed(self) -> np.ndarray:
        """Boolean mask: True for envs that have completed at least one episode."""
        return np.array([len(r) > 0 for r in self._completed_ep_rewards])

    def close(self):
        self._env.close()