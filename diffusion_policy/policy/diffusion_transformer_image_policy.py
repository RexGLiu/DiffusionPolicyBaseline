from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerWithEncoderForDiffusion


class DiffusionTransformerImagePolicy(BaseImagePolicy):
    def __init__(self,
            model: TransformerWithEncoderForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon,
            action_dim,
            n_action_steps,
            n_obs_steps,
            n_latency_steps=0,
            num_inference_steps=None,
            obs_as_cond=True,
            pred_action_steps_only=False,
            **kwargs):
        super().__init__()

        self.model = model
        self.noise_scheduler = noise_scheduler

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.n_latency_steps = n_latency_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. predict model output
            model_output = model(trajectory, t, cond)

            # 2. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        return trajectory


    def predict_action(self, obs_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet

        nobs = self.normalizer['image'].normalize(obs_dict['obs']['image'])
        B, _, C, H, W = nobs.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        assert self.obs_as_cond, "obs_as_cond == False not implemented"
        cond = nobs[:,:To]
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)

       # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch: dict) -> torch.Tensor:
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer['image'].normalize(batch['obs']['image'])         # (B, T, C, H, W)
        naction = self.normalizer['action'].normalize(batch['action'])           # (B, T, Da)

        # handle different ways of passing observation
        assert self.obs_as_cond, "obs_as_cond == False not implemented"
        cond = None
        trajectory = naction
        cond = nobs[:,:self.n_obs_steps,:]
        if self.pred_action_steps_only:
            To = self.n_obs_steps
            start = To - 1
            end = start + self.n_action_steps
            trajectory = naction[:,start:end]
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss
