import numpy as np
import torch
import collections
import tqdm

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.procgen.procgen_vec_history_wrapper import ProcgenVectorizedHistoryWrapper


class ProcgenImageRunnerVec(BaseImageRunner):
    """
    Evaluation runner using procgen's native C++ vectorization.

    Creates three independent envs (train / val / test) and steps them in
    parallel within a single loop, replacing the subprocess-based
    AsyncVectorEnv + chunk-iteration approach of the original runner.

    Level ranges follow "The Generalization Gap in Offline RL" (arXiv 2312.05742):
        train : start_level=0,   num_levels in [1, 200]
        val   : start_level=200, num_levels in [1,  50]
        test  : start_level=250, num_levels in [1, 9750]  (upper bound ~10_000)

    With native vectorization all n_envs workers sample uniformly from the
    configured level range, so per-env index does not map to a fixed seed.
    Reward logging uses env index as identifier rather than the procgen seed.
    """

    def __init__(
        self,
        output_dir,
        env_name: str = 'coinrun',
        n_envs: int = 50,
        max_steps: int = 1000,
        history_len: int = 8,
        n_action_steps: int = 1,
        tqdm_interval_sec: float = 5.0,
        n_train_levels: int = 200,
        n_val_levels: int = 50,
        n_test_levels: int = 100,
    ):
        super().__init__(output_dir)

        assert 1 <= n_train_levels <= 200, "n_train_levels must be in [1, 200]"
        assert 1 <= n_val_levels <= 50,    "n_val_levels must be in [1, 50]"
        assert 1 <= n_test_levels <= 9750, "n_test_levels must be in [1, 9750]"

        self.env_name = env_name
        self.n_envs = n_envs
        self.history_len = history_len
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.split_configs = {
            'train': {'start_level': 0,   'num_levels': n_train_levels},
            'val':   {'start_level': 200, 'num_levels': n_val_levels},
            'test':  {'start_level': 250, 'num_levels': n_test_levels},
        }

    def _make_env(self, start_level: int, num_levels: int) -> ProcgenVectorizedHistoryWrapper:
        return ProcgenVectorizedHistoryWrapper(
            env_name=self.env_name,
            num_envs=self.n_envs,
            start_level=start_level,
            num_levels=num_levels,
            history_len=self.history_len,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
        )

    def run(self, policy: BaseImagePolicy) -> dict:
        device = policy.device

        envs = {
            split: self._make_env(**config)
            for split, config in self.split_configs.items()
        }

        histories = {split: env.reset() for split, env in envs.items()}
        policy.reset()

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc="Eval InpaintingProcgenImageRunnerVec",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        while not all(env.episodes_completed().all() for env in envs.values()):
            for split, env in envs.items():
                if env.episodes_completed().all():
                    continue

                obs_dict = {'obs': histories[split]['obs']}
                obs_dict = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.inference_mode():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                histories[split], _, _, _ = env.step(action)

            pbar.update(self.n_action_steps)

        pbar.close()

        for split, env in envs.items():
            env.close()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for split, env in envs.items():
            prefix = f"{split}/"
            ep_rewards = env.get_episode_rewards()

            for i, rewards in enumerate(ep_rewards):
                max_reward = float(np.max(rewards)) if rewards else 0.0
                max_rewards[prefix].append(max_reward)
                log_data[prefix + f'sim_max_reward_env{i}'] = max_reward

            log_data[prefix + 'mean_score'] = float(np.mean(max_rewards[prefix]))

        return log_data
