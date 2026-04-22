import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

from diffusion_policy.env.procgen.procgen_image_env import ProcgenImageEnv

class ProcgenImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            env_name='coinrun',
            n_envs=50,           
            n_train_vis=3,
            n_val_vis=3,
            n_test_vis=6,
            max_steps=1000,
            n_obs_steps=8,
            n_action_steps=1,
            fps=15,
            crf=22,
            tqdm_interval_sec=5.0,
            n_train_levels=200,
            n_val_levels=50,
            n_test_levels=100,
        ):
        super().__init__(output_dir)

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    ProcgenImageEnv(
                        env_name=env_name,
                        start_level=0 
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs

        # Evaluation splits are set as in "The Generalization Gap in Offline Reinforcement Learning": https://arxiv.org/abs/2312.05742
        assert n_train_levels < 200, "Train levels have to be less than 200"
        assert n_val_levels < 200, "Train levels have to be less than 50"
        
        self.eval_splits = {
            "train": {"start_level": 0,   "num_levels": n_train_levels, "num_vis": n_train_vis},
            "val":   {"start_level": 200, "num_levels": n_val_levels,  "num_vis": n_val_vis},
            "test":  {"start_level": 250, "num_levels": n_test_levels, "num_vis": n_test_vis}
        }

        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        for split_name, config in self.eval_splits.items():
            start_level = config["start_level"]
            num_levels = config["num_levels"]
            num_vis = config["num_vis"]
            
            for i in range(num_levels):
                seed = start_level + i
                enable_render = i < num_vis

                def init_fn(env, seed=seed, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)

                    assert isinstance(env, MultiStepWrapper)
                    env.seed(seed)
                
                env_seeds.append(seed)
                env_prefixs.append(f'{split_name}/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval ProcgenImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            
            while not done:
                # create obs dict
                np_obs_dict = {
                    'obs': obs
                }

                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            
        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
