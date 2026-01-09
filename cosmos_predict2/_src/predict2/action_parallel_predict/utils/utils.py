import torch
from typing import List, Dict

def print_module_params(module, module_name="Module", max_elements=5):
    print(f"\n=== Parameters of {module_name} ===")
    has_params = False
    for name, param in module.named_parameters():
        has_params = True
        vals = param.data.detach().flatten()[:max_elements].cpu().numpy()
        print(f"  {name}: {vals}{'' if len(param.data.flatten()) <= max_elements else ' ...'}")
    
    if not has_params:
        print("  (no parameters found)")
    print("=" * (len(module_name) + 24))
    
def unnormalize_action(
    normalized_action: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    normalization_type: str = "mean_std",
    continuous_dims: List[int] = list(range(6)),
    action_dim: int = 7,
) -> torch.Tensor:
    """
    Unnormalize the continuous part of the action back to original scale.
    
    Supports arbitrary batch shapes: [..., action_dim]
    
    Args:
        normalized_action (torch.Tensor): Normalized action tensor of shape [..., action_dim].
        stats (dict): Dictionary containing stats ('min', 'max') or ('mean', 'std'), each of shape [len(continuous_dims)].
        normalization_type (str): "min_max" or "mean_std".
        continuous_dims (List[int]): Indices of continuous dimensions (e.g., [0,1,2,3,4,5]).
        action_dim (int): Total action dimension (e.g., 7). Used to reconstruct full action if needed.

    Returns:
        torch.Tensor: Unnormalized action tensor of same shape as input.
    """
    # Save original shape
    orig_shape = normalized_action.shape
    device = normalized_action.device

    # Flatten all but last dimension: [..., D] -> [N, D]
    flat_action = normalized_action.view(-1, orig_shape[-1])  # [N, D]

    # Clone to avoid in-place modification
    unnormalized_flat = flat_action.clone()

    # Extract continuous part: [N, len(continuous_dims)]
    cont_norm = flat_action[:, continuous_dims]

    # Move stats to device and ensure float32
    if normalization_type == "min_max":
        mins = stats["min"][0].to(device=device, dtype=flat_action.dtype)
        maxs = stats["max"][0].to(device=device, dtype=flat_action.dtype)
        # Reverse: [-1, 1] -> [0, 1] -> original
        cont_norm_01 = (cont_norm + 1.0) / 2.0
        cont_unnorm = cont_norm_01 * (maxs - mins + 1e-8) + mins
    elif normalization_type == "mean_std":
        mean = stats["mean"][0].to(device=device, dtype=flat_action.dtype)
        std = stats["std"][0].to(device=device, dtype=flat_action.dtype)
        cont_unnorm = cont_norm * std + mean
    else:
        raise ValueError(f"Unsupported normalization_type: {normalization_type}")

    # Write back continuous dims
    unnormalized_flat[:, continuous_dims] = cont_unnorm

    # Reshape back to original shape
    unnormalized = unnormalized_flat.view(orig_shape)
    return unnormalized