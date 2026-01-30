"""
WGP monkeypatches for headless operation.

These patches adapt WGP (Wan2GP) for headless/programmatic use without
modifying the upstream Wan2GP repository files directly.

Each patch is documented with:
- Why it's needed
- What it modifies
- When it should be applied
"""

import os
from typing import Optional, TYPE_CHECKING

from source.logging_utils import model_logger

if TYPE_CHECKING:
    import types


def apply_qwen_model_routing_patch(wgp_module: "types.ModuleType", wan_root: str) -> bool:
    """
    Patch WGP to route Qwen-family models to their dedicated handler.

    Why: WGP's load_wan_model doesn't natively handle Qwen models.
    This routes Qwen-family models (qwen_image_edit, etc.) to the
    dedicated Qwen handler in models/qwen/qwen_handler.py.

    Args:
        wgp_module: The imported wgp module
        wan_root: Path to Wan2GP root directory

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        _orig_load_wan_model = wgp_module.load_wan_model

        def _patched_load_wan_model(
            model_filename,
            model_type,
            base_model_type,
            model_def,
            quantizeTransformer=False,
            dtype=None,
            VAE_dtype=None,
            mixed_precision_transformer=False,
            save_quantized=False
        ):
            # Check if this is a Qwen-family model
            try:
                base = wgp_module.get_base_model_type(base_model_type)
            except Exception:
                base = base_model_type

            if isinstance(base, str) and "qwen" in base.lower():
                model_logger.debug("[QWEN_LOAD] Routing to Qwen family loader via monkeypatch")
                from models.qwen.qwen_handler import family_handler as _qwen_handler
                pipe_processor, pipe = _qwen_handler.load_model(
                    model_filename=model_filename,
                    model_type=model_type,
                    base_model_type=base_model_type,
                    model_def=model_def,
                    quantizeTransformer=quantizeTransformer,
                    text_encoder_quantization=wgp_module.text_encoder_quantization,
                    dtype=dtype,
                    VAE_dtype=VAE_dtype,
                    mixed_precision_transformer=mixed_precision_transformer,
                    save_quantized=save_quantized,
                )
                return pipe_processor, pipe

            # Fallback to original WAN loader
            return _orig_load_wan_model(
                model_filename, model_type, base_model_type, model_def,
                quantizeTransformer=quantizeTransformer, dtype=dtype, VAE_dtype=VAE_dtype,
                mixed_precision_transformer=mixed_precision_transformer, save_quantized=save_quantized
            )

        wgp_module.load_wan_model = _patched_load_wan_model
        return True

    except Exception as e:
        model_logger.debug(f"[QWEN_LOAD] Failed to apply load_wan_model patch: {e}")
        return False


def apply_qwen_lora_directory_patch(wgp_module: "types.ModuleType", wan_root: str) -> bool:
    """
    Patch get_lora_dir to redirect Qwen models to loras_qwen/ directory.

    Why: Qwen models need their own LoRA directory (loras_qwen/) since they
    use a different LoRA format than WAN models.

    Args:
        wgp_module: The imported wgp module
        wan_root: Path to Wan2GP root directory

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        _orig_get_lora_dir = wgp_module.get_lora_dir

        def _patched_get_lora_dir(model_type: str):
            try:
                mt = (model_type or "").lower()
                if "qwen" in mt:
                    qwen_dir = os.path.join(wan_root, "loras_qwen")
                    if os.path.isdir(qwen_dir):
                        return qwen_dir
            except Exception:
                pass
            return _orig_get_lora_dir(model_type)

        wgp_module.get_lora_dir = _patched_get_lora_dir
        return True

    except Exception as e:
        model_logger.debug(f"[QWEN_LOAD] Failed to apply get_lora_dir patch: {e}")
        return False


def apply_lora_multiplier_parser_patch(wgp_module: "types.ModuleType") -> bool:
    """
    Harmonize LoRA multiplier parsing across pipelines.

    Why: Use the 3-phase capable parser so Qwen pipeline (which expects
    phase3/shared) receives a compatible slists_dict. This is backward
    compatible for 2-phase models.

    Args:
        wgp_module: The imported wgp module

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        from shared.utils import loras_mutipliers as _shared_lora_utils
        wgp_module.parse_loras_multipliers = _shared_lora_utils.parse_loras_multipliers
        wgp_module.preparse_loras_multipliers = _shared_lora_utils.preparse_loras_multipliers
        return True

    except Exception as e:
        model_logger.debug(f"[QWEN_LOAD] Failed to apply lora parser patch: {e}")
        return False


def apply_qwen_inpainting_lora_patch() -> bool:
    """
    Disable Qwen's built-in inpainting LoRA (preload_URLs) in headless mode.

    Why: Qwen models have a built-in inpainting LoRA that gets auto-loaded.
    In headless mode, this can cause issues. Disabled by default unless
    HEADLESS_WAN2GP_ENABLE_QWEN_INPAINTING_LORA=1 is set.

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        from models.qwen import qwen_main as _qwen_main
        _orig_qwen_get_loras_transformer = _qwen_main.model_factory.get_loras_transformer

        def _patched_qwen_get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs):
            try:
                if os.environ.get("HEADLESS_WAN2GP_ENABLE_QWEN_INPAINTING_LORA", "0") != "1":
                    return [], []
            except Exception:
                # If env check fails, fall back to disabled behavior
                return [], []
            return _orig_qwen_get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs)

        _qwen_main.model_factory.get_loras_transformer = _patched_qwen_get_loras_transformer
        return True

    except Exception as e:
        model_logger.debug(f"[QWEN_LOAD] Failed to apply Qwen inpainting LoRA patch: {e}")
        return False


def apply_all_wgp_patches(wgp_module: "types.ModuleType", wan_root: str) -> dict:
    """
    Apply all WGP patches for headless operation.

    Args:
        wgp_module: The imported wgp module
        wan_root: Path to Wan2GP root directory

    Returns:
        Dict with patch names as keys and success status as values
    """
    results = {}

    results["qwen_model_routing"] = apply_qwen_model_routing_patch(wgp_module, wan_root)
    results["qwen_lora_directory"] = apply_qwen_lora_directory_patch(wgp_module, wan_root)
    results["lora_multiplier_parser"] = apply_lora_multiplier_parser_patch(wgp_module)
    results["qwen_inpainting_lora"] = apply_qwen_inpainting_lora_patch()

    # Log summary
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    if successful:
        model_logger.debug(f"[WGP_PATCHES] Applied: {', '.join(successful)}")
    if failed:
        model_logger.debug(f"[WGP_PATCHES] Failed (non-fatal): {', '.join(failed)}")

    return results
