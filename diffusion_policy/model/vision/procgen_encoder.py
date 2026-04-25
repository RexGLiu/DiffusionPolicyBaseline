import torch
import torch.nn as nn
from typing import Sequence

'''
Feature extractor module adapted from https://huggingface.co/cleanrl/CoinrunHard-v0-cleanba_ppo_envpool_procgen-seed1/blob/main/cleanba_ppo_envpool_procgen.py
'''


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.pad(x, (0, 1, 0, 1))  # (left, right, top, bottom)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        x = self.res1(x)
        x = self.res2(x)
        return x

def _get_flat_size(obs_space, channels):
    x = torch.zeros(1, *obs_space)
    in_channels = obs_space[0]
    convs = nn.ModuleList([
        ConvSequence(c_in, c_out)
        for c_in, c_out in zip([in_channels] + list(channels[:-1]), channels)
    ])
    for conv in convs:
        x = conv(x)
    return x.shape[1] * x.shape[2] * x.shape[3]


class ProcgenEncoder(nn.Module):
    """
    Custom pre-trained observation encoder for Procgen.
    Define your architecture here and set checkpoint_path in the config to load weights.
    """

    def __init__(self,
                 observation_space: Sequence[int] = [3,64,64], 
                 channels: Sequence[int] = (16, 32, 32), 
                 embed_dim=256,
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
        convs = []
        c_in = observation_space[0]
        for c_out in channels:
            convs.append(ConvSequence(c_in, c_out))
            c_in = c_out
        self.convs = nn.ModuleList(convs)
        flat_size = _get_flat_size(observation_space, channels)
        self.fc = nn.Linear(flat_size, embed_dim)

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
        x = x.float() / 255.0
        for conv in self.convs:
            x = conv(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = torch.relu(x)
        return x





if __name__ == "__main__":
    obs_space = [3,64,64]
    f_extractor = ProcgenEncoder(obs_space, [16,32,32], 64)
    x = torch.zeros([2,*obs_space])
    f_extractor(x)
    print('done')
