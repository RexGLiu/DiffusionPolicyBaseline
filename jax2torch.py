"""
Usage:
Small script to convert a trained jax to torch.
"""

import numpy as np
import torch
import flax.serialization

from diffusion_policy.model.vision.procgen_encoder import ProcgenFeatureExtractor
from jax_model.jax_procgen_encoder import JaxEncoder

import inspect


def load_jax_checkpoint(path: str) -> dict:
    """Load raw JAX checkpoint — works with both msgpack and orbax."""
    try:
        # Try orbax first (modern)
        import orbax.checkpoint as ocp
        checkpointer = ocp.StandardCheckpointer()
        return checkpointer.restore(path)
    except Exception:
        # Fall back to msgpack
        with open(path, 'rb') as f:
            return flax.serialization.from_bytes(None, f.read())


def conv_kernel(w: np.ndarray) -> torch.Tensor:
    """Flax conv: (H, W, C_in, C_out) → PyTorch: (C_out, C_in, H, W)"""
    return torch.from_numpy(np.array(w).transpose(3, 2, 0, 1))


def dense_kernel(w: np.ndarray) -> torch.Tensor:
    """Flax dense: (C_in, C_out) → PyTorch: (C_out, C_in)"""
    return torch.from_numpy(np.array(w).T)


def bias(b: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(b))


def port_residual_block(jax_params: dict) -> dict:
    """
    Flax names sub-Convs as 'Conv_0' and 'Conv_1'
    since @nn.compact numbers them in call order.
    """
    return {
        'conv1.weight': conv_kernel(jax_params['Conv_0']['kernel']),
        'conv1.bias':   bias(jax_params['Conv_0']['bias']),
        'conv2.weight': conv_kernel(jax_params['Conv_1']['kernel']),
        'conv2.bias':   bias(jax_params['Conv_1']['bias']),
    }


def port_conv_sequence(jax_params: dict, idx: int) -> dict:
    """
    Inside ConvSequence, @nn.compact names the modules as:
      Conv_0        → the initial conv
      ResidualBlock_0, ResidualBlock_1  → the two residual blocks
    """
    sd = {}
    # Initial conv
    sd['conv.weight'] = conv_kernel(jax_params['Conv_0']['kernel'])
    sd['conv.bias']   = bias(jax_params['Conv_0']['bias'])
    # Residual blocks
    for rb_idx in range(2):
        rb_params = jax_params[f'ResidualBlock_{rb_idx}']
        for k, v in port_residual_block(rb_params).items():
            sd[f'res{rb_idx + 1}.{k}'] = v
    return sd


def port_feature_extractor(jax_params: dict) -> dict:
    # Unwrap the full agent checkpoint down to the feature extractor params
    p = jax_params['1']['0']['params']

    sd = {}

    for i in range(3):
        cs_params = p[f'ConvSequence_{i}']
        for k, v in port_conv_sequence(cs_params, i).items():
            sd[f'convs.{i}.{k}'] = v

    sd['fc.weight'] = dense_kernel(p['Dense_0']['kernel'])
    sd['fc.bias']   = bias(p['Dense_0']['bias'])

    return sd


def load_ported_model(
    checkpoint_path: str,
    model: torch.nn.Module,
    strict: bool = True,
) -> torch.nn.Module:
    jax_params = load_jax_checkpoint(checkpoint_path)
    state_dict = port_feature_extractor(jax_params)

    # Verify shapes match before loading
    current_sd = model.state_dict()
    mismatches = []
    for k, v in state_dict.items():
        if k not in current_sd:
            mismatches.append(f"  MISSING in pytorch model: {k}")
        elif v.shape != current_sd[k].shape:
            mismatches.append(f"  SHAPE MISMATCH {k}: jax={tuple(v.shape)} vs torch={tuple(current_sd[k].shape)}")
    if mismatches:
        raise ValueError("Weight porting errors:\n" + "\n".join(mismatches))

    model.load_state_dict(state_dict, strict=strict)
    return model


def test_equivalence(jax_params, jax_model, torch_model, in_channels=3, h=64, w=64, atol=1e-2):
    torch_model.eval()

    # Create a random input
    np.random.seed(42)
    x_np = np.random.rand(1, in_channels, h, w).astype(np.float32)

    # JAX forward — model expects NCHW and transposes internally
    jax_out = jax_model.apply({'params': jax_params}, x_np)
    jax_out_np = np.array(jax_out)
    print('jax:', jax_out_np.shape)

    # PyTorch forward
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np))
    torch_out_np = torch_out.numpy()
    print('torch:', torch_out_np.shape)


    jax_fc_kernel = np.array(jax_params['Dense_0']['kernel'])  # (2048, 256)
    torch_fc_weight = torch_model.fc.weight.detach().numpy()    # (256, 2048)

    print("JAX kernel shape:  ", jax_fc_kernel.shape)
    print("Torch weight shape:", torch_fc_weight.shape)
    print("Values match:", np.allclose(jax_fc_kernel.T, torch_fc_weight, atol=1e-6))

    # Also check bias
    jax_fc_bias = np.array(jax_params['Dense_0']['bias'])
    torch_fc_bias = torch_model.fc.bias.detach().numpy()
    print("Bias match:", np.allclose(jax_fc_bias, torch_fc_bias, atol=1e-6))

    # Compare
    max_diff = np.abs(jax_out_np - torch_out_np).max()
    mean_diff = np.abs(jax_out_np - torch_out_np).mean()
    print(f"Max diff:  {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    print('jax: ', jax_out_np.shape)
    print('torch: ', torch_out_np.shape)
    jax_out_np = jax_out_np.flatten()
    torch_out_np = torch_out_np.flatten()
    diff = np.abs(jax_out_np - torch_out_np)
    idx = np.where(diff > atol)[0]

    print(f"Number of mismatches: {len(idx)} out of {jax_out_np.size} elements")
    print("First few mismatches:")
    for i in idx[:10]:
        print(i, torch_out_np[i], jax_out_np[i])



    # mask = (jax_out_np != 0) | (torch_out_np != 0)
    # indices = np.where(mask)[0]
    # filtered_jax_out = jax_out_np[indices]
    # filtered_torch_out = torch_out_np[indices]

    # max_diff = np.abs(filtered_jax_out - filtered_torch_out).max()
    # mean_diff = np.abs(filtered_jax_out - filtered_torch_out).mean()
    # print(f"Max diff:  {max_diff:.6f}")
    # print(f"Mean diff: {mean_diff:.6f}")

    # diff = np.abs(filtered_jax_out - filtered_torch_out)
    # idx = np.where(diff > atol)[0]
    # print(f"Number of mismatches: {len(idx)} out of {filtered_torch_out.size} elements")
    # print("First few mismatches:")
    # for i in idx[:10]:
    #     print(i, filtered_torch_out[i], filtered_jax_out[i])

    # print("First non-zero elements:")
    # for i in range(10):
    #     print(i, filtered_torch_out[i], filtered_jax_out[i])


    # print("First non-zero elements:")
    # for i in indices[:20]:  # first 20 non-zero elements
    #     print(f"{i:5d}  jax={jax_out_np[i]:.8f}  torch={torch_out_np[i]:.8f}  diff={abs(jax_out_np[i] - torch_out_np[i]):.8f}")
    
    # top_indices = np.argsort(diff)[::-1]
    # print("Top few mismatches:")
    # for i in top_indices[0:100]:
    #     print(i, torch_out_np[i], jax_out_np[i])


    match = np.allclose(jax_out_np, torch_out_np, atol=atol)
    print(f"Match (atol={atol}): {match}")
    return match


def print_tree(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            print_tree(v, prefix + "  ")
        else:
            arr = np.array(v)
            print(f"{prefix}{k}: {arr.shape} {arr.dtype}")



def main():
    jax_model_path = "jax_model/cleanba_ppo_envpool_procgen.cleanrl_model"

    # 1. Load checkpoint
    with open(jax_model_path, "rb") as f:
        raw = flax.serialization.msgpack_restore(f.read())
    # print_tree(raw)
    jax_params = raw['1']['0']['params']

    # 2. Load ported PyTorch model
    torch_model = ProcgenFeatureExtractor()
    torch_model = load_ported_model(jax_model_path, torch_model)

    # # 3. Test
    jax_model = JaxEncoder()
    test_equivalence(jax_params, jax_model, torch_model)

    torch.save(torch_model.state_dict(), "coinrun_hard_clean_rl_feature_extractor.pt")

    print('end')




if __name__ == "__main__":
    main()
