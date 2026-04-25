import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym.spaces import Box, Discrete
from gym import spaces
import procgen

from abc import ABC, abstractmethod
from typing import Optional, Type, Dict

'''
Feature extractor module from https://huggingface.co/sgoodfriend/ppo-procgen-coinrun-easy/blob/main/shared/module/feature_extractor.py
'''

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Copied from Stable-Baselines 3

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            activation(),
            nn.Conv2d(channels, channels, 3, padding=1),
            activation(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class ConvSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(out_channels, activation),
            ResidualBlock(out_channels, activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class CnnFeatureExtractor(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()


class ImpalaCnn(CnnFeatureExtractor):
    """
    IMPALA-style CNN architecture
    """

    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(in_channels, activation)
        sequences = []
        for out_channels in [16, 32, 32]:
            sequences.append(
                ConvSequence(
                    in_channels, out_channels, activation
                )
            )
            in_channels = out_channels
        sequences.extend(
            [
                activation(),
                nn.Flatten(),
            ]
        )
        self.seq = nn.Sequential(*sequences)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.seq(obs)


class ProcgenFeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_space: gym.Space, # H x W x C
        activation: Type[nn.Module],
        cnn_feature_dim: int = 512,
    ) -> None:
        super().__init__()
        if isinstance(obs_space, Box):
            # Conv2D: (channels, height, width)
            if len(obs_space.shape) == 3:
                cnn = ImpalaCnn(
                    obs_space.shape[-1],
                    activation,
                )

                def preprocess(obs: torch.Tensor) -> torch.Tensor:
                    if len(obs.shape) == 3:
                        obs = obs.unsqueeze(0)
                    return obs.float() / 255.0

                with torch.no_grad():
                    dummy_obs = torch.as_tensor(obs_space.sample()).permute(2, 0, 1)
                    cnn_out = cnn(preprocess(dummy_obs))
                self.preprocess = preprocess
                self.feature_extractor = nn.Sequential(
                    cnn,
                    nn.Linear(cnn_out.shape[1], cnn_feature_dim),
                    activation(),
                )
                self.out_dim = cnn_feature_dim
            elif len(obs_space.shape) == 1:

                def preprocess(obs: torch.Tensor) -> torch.Tensor:
                    if len(obs.shape) == 1:
                        obs = obs.unsqueeze(0)
                    return obs.float()

                self.preprocess = preprocess
                self.feature_extractor = nn.Flatten()
                self.out_dim = get_flattened_obs_dim(obs_space)
            else:
                raise ValueError(f"Unsupported observation space: {obs_space}")
        else:
            raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.preprocess:
            obs = self.preprocess(obs)
        return self.feature_extractor(obs)
    
def load_feature_extractor(path: str, 
                           obs_space: gym.Space, 
                           activation: Type[nn.Module] = nn.ReLU,
                           cnn_feature_dim: int = 256) -> ProcgenFeatureExtractor:
    
    feature_extractor = ProcgenFeatureExtractor(obs_space, activation, cnn_feature_dim)

    state_dict = torch.load(path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_feature_extractor."):
            new_key = key[len("_feature_extractor."):]  # removes the prefix
            new_state_dict[new_key] = value

    feature_extractor.load_state_dict(new_state_dict)
    feature_extractor.eval()
    print('done loading feature extractor')

    return feature_extractor

if __name__ == "__main__":
    path = "/home/rexgliu/archive/tmp/procgen_encoder_ckpts/coinrun_easy_feature_extractor_original.pt"
    env = gym.make("procgen:procgen-coinrun-v0")
    feature_extractor = load_feature_extractor(path, env.observation_space)
    torch.save(feature_extractor.state_dict(), "coinrun_easy_feature_extractor.pt")
    print('done main')