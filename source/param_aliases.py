# source/param_aliases.py
"""
Parameter name aliases for backwards compatibility and spelling variants.

This module provides mappings between alternative parameter names and their
canonical names used in the codebase. Use these to normalize user input.
"""
from typing import Dict

# Mapping from alternative names to canonical names
PARAM_ALIASES: Dict[str, str] = {
    # British/American spelling variants
    "color_match_videos": "colour_match_videos",

    # Legacy parameter names (deprecated but still supported)
    "video_guide_path": "video_guide",
    "video_mask_path": "video_mask",
    "image_guide_path": "image_guide",
    "image_mask_path": "image_mask",

    # Common typos/variants
    "negative": "negative_prompt",
    "neg_prompt": "negative_prompt",

    # Shortened forms
    "steps": "num_inference_steps",
    "cfg": "guidance_scale",
    "cfg_scale": "guidance_scale",
}


def normalize_param_name(param_name: str) -> str:
    """
    Convert a parameter name to its canonical form.

    Args:
        param_name: The parameter name (possibly an alias)

    Returns:
        The canonical parameter name
    """
    return PARAM_ALIASES.get(param_name, param_name)


def normalize_params(params: dict) -> dict:
    """
    Normalize all parameter names in a dictionary.

    Args:
        params: Dictionary of parameters with possibly aliased names

    Returns:
        New dictionary with canonical parameter names
    """
    return {normalize_param_name(k): v for k, v in params.items()}
