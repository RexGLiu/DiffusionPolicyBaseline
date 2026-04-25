import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Type, Sequence

'''
Feature extractor module from https://huggingface.co/sgoodfriend/ppo-procgen-coinrun-easy/blob/main/shared/module/feature_extractor.py
'''


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




class ProcgenEncoder(nn.Module):
    """
    Custom pre-trained observation encoder for Procgen.
    Define your architecture here and set checkpoint_path in the config to load weights.
    """

    def __init__(self,
                 observation_space: Sequence[int] = [3,64,64], 
                 embed_dim: int = 256,
                 checkpoint_path: str = None,
                 state_dict_key: str = None):
        """
        embed_dim:       output latent dimensionality
        checkpoint_path: path to .pt/.ckpt file; if None, random init is used
        state_dict_key:  dotted key into the checkpoint dict,
                         e.g. "encoder" or "model.encoder" — None tries common conventions
        """
        super().__init__()
        self.embed_dim = embed_dim

        # architecture
        activation = nn.ReLU

        # Conv2D: (channels, height, width)
        cnn = ImpalaCnn(
            observation_space[0],
            activation,
        )

        def preprocess(obs: torch.Tensor) -> torch.Tensor:
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            return obs.float() / 255.0

        with torch.no_grad():
            dummy_obs = torch.randn(1, *observation_space)
            cnn_out = cnn(preprocess(dummy_obs))
        self.preprocess = preprocess
        self.feature_extractor = nn.Sequential(
            cnn,
            nn.Linear(cnn_out.shape[1], embed_dim),
            activation(),
        )
        self.out_dim = embed_dim
    

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if state_dict_key is not None:
                for key in state_dict_key.split('.'):
                    ckpt = ckpt[key]
            elif isinstance(ckpt, dict):
                for key in ('state_dict', 'model', 'encoder'):
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break
            self.load_state_dict(ckpt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, embed_dim)
        """
        if self.preprocess:
            x = self.preprocess(x)
        return self.feature_extractor(x)

    
def load_feature_extractor(path: str, 
                           observation_space = [3,64,64], 
                           cnn_feature_dim: int = 256) -> ProcgenEncoder:
    
    feature_extractor = ProcgenEncoder(observation_space, cnn_feature_dim)

    state_dict = torch.load(path)
    if path.endswith('_original.pt'):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_feature_extractor."):
                new_key = key[len("_feature_extractor."):]  # removes the prefix
                new_state_dict[new_key] = value
        state_dict = new_state_dict

    feature_extractor.load_state_dict(state_dict)
    feature_extractor.eval()

    print('done loading feature extractor')

    return feature_extractor

if __name__ == "__main__":
    # path = "/home/rexgliu/archive/tmp/procgen_encoder_ckpts/coinrun_easy_feature_extractor_original.pt"
    path = "/home/rexgliu/archive/tmp/procgen_encoder_ckpts/coinrun_easy_feature_extractor.pt"
    feature_extractor = load_feature_extractor(path)

    if path.endswith('_original.pt'):
        torch.save(feature_extractor.state_dict(), "coinrun_easy_feature_extractor.pt")
    print('done main')