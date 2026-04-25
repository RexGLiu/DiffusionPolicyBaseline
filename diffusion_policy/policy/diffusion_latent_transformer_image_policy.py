from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


class DiffusionLatentTransformerImagePolicy(BaseImagePolicy):
    """
    Diffusion transformer operating entirely in latent space.

    A frozen pre-trained encoder maps images to latents once; the diffusion
    transformer then jointly denoises (action, latent_obs) trajectories via
    inpainting. No decoder — obs loss is computed in latent space.
    """

    def __init__(self,
                 obs_encoder: nn.Module,
                 model: TransformerForDiffusion,
                 noise_scheduler: DDPMScheduler,
                 horizon: int,
                 action_dim: int,
                 embed_dim: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 n_latency_steps: int = 0,
                 num_inference_steps: int = None,
                 obs_as_cond=True,
                 pred_action_steps_only=False,
                 **kwargs):
        super().__init__()

        # Freeze encoder weights before registering as submodule
        for param in obs_encoder.parameters():
            param.requires_grad = False
        obs_encoder.eval()

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self._embed_dim = embed_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.n_latency_steps = n_latency_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def train(self, mode: bool = True):
        """Keep obs_encoder frozen regardless of training mode."""
        result = super().train(mode)
        self.obs_encoder.eval()
        return result

    # ========= helpers ============

    def _encode_obs(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, T, C, H, W)
        returns: (B, T, embed_dim) — no gradient flows through encoder
        """
        B, T, C, H, W = images.shape
        with torch.no_grad():
            latents = self.obs_encoder(images.reshape(B * T, C, H, W))
        return latents.reshape(B, T, self._embed_dim)

    # ========= inference ============

    def conditional_sample(self,
                           cond_data: torch.Tensor,
                           obs_cond: torch.Tensor,
                           generator=None,
                           **kwargs) -> torch.Tensor:
        """
        Joint denoising over action trajectory.
        cond_data: (B, T, action_dim)                     Partially denoised trajectory
        obs_cond: (B, T, embed_dim)                       Observation conditioning information
        returns:   (B, T, action_dim)
        """
        scheduler = self.noise_scheduler
        x = torch.randn(cond_data.shape, dtype=cond_data.dtype,
                        device=cond_data.device, generator=generator)
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            x_pred = self.model(x, t, obs_cond)
            x = scheduler.step(x_pred, t, x, generator=generator, **kwargs).prev_sample

        return x

    def predict_action(self, obs_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        assert self.obs_as_cond, "obs_as_cond == False not implemented"

        images = obs_dict['obs']['image']       # (B, Th, C, H, W)
        latent_obs_cond = self._encode_obs(images)  # (B, Th, embed_dim) -- note: encoder expects unnormalised pixel values in range [0,255]

        B = latent_obs_cond.shape[0]
        T = self.horizon
        Da = self.action_dim
        device, dtype = self.device, self.dtype

        shape = (B, self.n_action_steps, Da) if self.pred_action_steps_only else (B,T,Da)
        cond_data = torch.zeros(shape, device=device, dtype=dtype)

        x = self.conditional_sample(cond_data, obs_cond=latent_obs_cond, **self.kwargs)

        action_pred = self.normalizer['action'].unnormalize(x[:, :, :Da])

        # get action
        start = self.n_obs_steps - 1
        assert start > 0, "action start index must be at least 0"
        if self.pred_action_steps_only:
            action = action_pred
        else:
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        return {
            'action': action,
            'action_pred': action_pred[:, start:],
        }

    # ========= training ============

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self,
                      weight_decay: float,
                      learning_rate: float,
                      betas: Tuple[float, float]) -> torch.optim.Optimizer:
        # Only the transformer is optimized; obs_encoder is frozen
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas))

    def compute_loss(self, batch: dict) -> torch.Tensor:
        assert self.obs_as_cond, "obs_as_cond == False not implemented"

        images = batch['obs']['image'][:,:self.n_obs_steps,...]   # (B, To, C, H, W)
        actions = batch['action']        # (B, T, Da)

        naction = self.normalizer['action'].normalize(actions)
        latent_obs_cond = self._encode_obs(images)   # (B, To, embed_dim) -- note: encoder expects unnormalised pixel values in range [0,255]

        B, T, Da = naction.shape
        De = self._embed_dim

        x = naction
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            x = naction[:,start:end]

        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=x.device).long()

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        x_pred = self.model(noisy_x, timesteps, latent_obs_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = x
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(x_pred, target)

        return loss
