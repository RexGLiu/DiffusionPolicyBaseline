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


class ProcgenFeatureExtractor(nn.Module):
    def __init__(self, obs_space: Sequence[int] = [3,64,64], channels: Sequence[int] = (16, 32, 32), emb_dim=256):
        super().__init__()
        convs = []
        c_in = obs_space[0]
        for c_out in channels:
            convs.append(ConvSequence(c_in, c_out))
            c_in = c_out
        self.convs = nn.ModuleList(convs)
        flat_size = _get_flat_size(obs_space, channels)
        self.fc = nn.Linear(flat_size, emb_dim)

    def forward(self, x):
        '''
            x: (B, C, H, W)
        '''
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
    f_extractor = ProcgenFeatureExtractor(obs_space, [16,32,32], 64)
    x = torch.zeros([2,*obs_space])
    f_extractor(x)
    print('done')