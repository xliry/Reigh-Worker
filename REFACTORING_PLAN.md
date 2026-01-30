# Headless-Wan2GP Refactoring Plan

## Overview

This plan addresses all identified code quality issues in a systematic, dependency-aware order. Each phase builds on the previous, minimizing risk while maximizing impact.

**Estimated Scope:** ~50 individual changes across 6 phases
**Risk Mitigation:** Each phase ends with a verification step

---

## Phase 1: Safe Cleanup (Low Risk, High Visibility)

Remove dead code, fix obvious bugs, and correct misleading comments. These changes have zero functional impact but improve code clarity immediately.

### 1.1 Remove Dead Code

#### 1.1.1 Delete HeadlessAPI stub class
**File:** `headless_model_management.py` (lines 1367-1391)
**Action:** Delete the entire `HeadlessAPI` class - it's never instantiated or used
```python
# DELETE THIS ENTIRE CLASS
class HeadlessAPI:
    """Simple HTTP API for submitting tasks to the queue..."""
    def __init__(self, queue: HeadlessTaskQueue, port: int = 8080):
        ...
```
Also remove references in `main()` function (lines 1421-1422, 1427, 1436, 1487)

#### 1.1.2 Remove unused imports related to HeadlessAPI
**File:** `headless_model_management.py`
**Action:** Check if `signal` import is still needed after HeadlessAPI removal

#### 1.1.3 Delete empty TODO functions
**File:** `headless_model_management.py`
- `_save_queue_state` (lines 1355-1359) - contains only `pass`
- `_load_queue_state` (lines 1361-1364) - contains only `pass`
**Action:** Either implement or delete with TODO comment explaining why

### 1.2 Fix Obvious Bugs

#### 1.2.1 Fix requests/httpx inconsistency
**File:** `source/db_operations.py` (line 603)
**Bug:** Uses `requests.post` but `requests` is not imported (should be `httpx`)
```python
# BEFORE
resp = requests.post(edge_url, json=payload, headers=headers, timeout=30)

# AFTER
resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)
```

### 1.3 Fix Misleading Comments

#### 1.3.1 Fix wrong file path in header
**File:** `source/db_operations.py` (line 1)
```python
# BEFORE
# source/task_handlers/db_operations.py

# AFTER
# source/db_operations.py
```

#### 1.3.2 Remove outdated comments about file locations
**Action:** Grep for any other wrong path comments and fix them

### 1.4 Remove Hardcoded Credentials from Defaults

#### 1.4.1 Remove Supabase anon key from argparse defaults
**File:** `worker.py` (lines 232-234)
```python
# BEFORE
parser.add_argument("--supabase-anon-key", type=str,
    default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")

# AFTER
parser.add_argument("--supabase-anon-key", type=str, default=None,
    help="Supabase anon key (can also be set via SUPABASE_ANON_KEY env var)")
```
**Also:** Update the code that reads this to fall back to env var

#### 1.4.2 Remove hardcoded worker ID fallback
**File:** `source/db_operations.py` (lines 641-642)
```python
# BEFORE
if not worker_id:
    worker_id = "gpu-20250723_221138-afa8403b"

# AFTER
if not worker_id:
    raise ValueError("worker_id is required for get_oldest_queued_task_supabase")
```

### Phase 1 Verification
```bash
# Run after Phase 1 completion
python -c "from headless_model_management import HeadlessTaskQueue; print('Import OK')"
python -c "from source.db_operations import get_oldest_queued_task_supabase; print('Import OK')"
python worker.py --help  # Verify argparse still works
```

---

## Phase 2: Centralize Definitions (Medium Risk)

Create single sources of truth for duplicated definitions.

### 2.1 Create Task Type Registry

#### 2.1.1 Create new centralized module
**New File:** `source/task_types.py`
```python
"""
Centralized task type definitions.

This is the SINGLE SOURCE OF TRUTH for all task type information.
"""
from typing import Dict, Set, FrozenSet

# All task types that go directly to the WGP queue (no special handling)
DIRECT_QUEUE_TASK_TYPES: FrozenSet[str] = frozenset({
    "wan_2_2_t2i", "vace", "vace_21", "vace_22", "flux", "t2v", "t2v_22",
    "i2v", "i2v_22", "hunyuan", "ltxv", "generate_video",
    "qwen_image_edit", "qwen_image_hires", "qwen_image_style",
    "image_inpaint", "annotated_image_edit",
    "qwen_image", "qwen_image_2512", "z_image_turbo",
    "z_image_turbo_i2i"
})

# Task types that WGP handles (for post-processing organization)
WGP_GENERATION_TASK_TYPES: FrozenSet[str] = frozenset({
    "vace", "vace_21", "vace_22",
    "flux",
    "t2v", "t2v_22", "wan_2_2_t2i",
    "i2v", "i2v_22",
    "hunyuan",
    "ltxv",
    "qwen_image_edit", "qwen_image_style", "image_inpaint", "annotated_image_edit",
    "inpaint_frames",
    "generate_video",
    "z_image_turbo", "z_image_turbo_i2i"
})

# Orchestrator task types (get higher priority)
ORCHESTRATOR_TASK_TYPES: FrozenSet[str] = frozenset({
    "travel_orchestrator",
    "join_clips_orchestrator",
    "edit_video_orchestrator"
})

# Task type to default model mapping
TASK_TYPE_TO_DEFAULT_MODEL: Dict[str, str] = {
    "generate_video": "t2v",
    "vace": "vace_14B_cocktail_2_2",
    "vace_21": "vace_14B",
    "vace_22": "vace_14B_cocktail_2_2",
    "wan_2_2_t2i": "t2v_2_2",
    "flux": "flux",
    "t2v": "t2v",
    "t2v_22": "t2v_2_2",
    "i2v": "i2v_14B",
    "i2v_22": "i2v_2_2",
    "hunyuan": "hunyuan",
    "ltxv": "ltxv_13B",
    "join_clips_segment": "wan_2_2_vace_lightning_baseline_2_2_2",
    "inpaint_frames": "wan_2_2_vace_lightning_baseline_2_2_2",
    "qwen_image_edit": "qwen_image_edit_20B",
    "qwen_image_hires": "qwen_image_edit_20B",
    "qwen_image_style": "qwen_image_edit_20B",
    "image_inpaint": "qwen_image_edit_20B",
    "annotated_image_edit": "qwen_image_edit_20B",
    "qwen_image": "qwen_image_edit_20B",
    "qwen_image_2512": "qwen_image_2512_20B",
    "z_image_turbo": "z_image",
    "z_image_turbo_i2i": "z_image_img2img",
}

# Image-to-image task types (empty prompt is acceptable)
IMG2IMG_TASK_TYPES: FrozenSet[str] = frozenset({
    "z_image_turbo_i2i",
    "qwen_image_edit",
    "qwen_image_style",
    "image_inpaint"
})

def get_default_model(task_type: str) -> str:
    """Get default model for a task type."""
    return TASK_TYPE_TO_DEFAULT_MODEL.get(task_type, "t2v")

def is_direct_queue_task(task_type: str) -> bool:
    """Check if task type goes directly to WGP queue."""
    return task_type in DIRECT_QUEUE_TASK_TYPES

def is_orchestrator_task(task_type: str) -> bool:
    """Check if task type is an orchestrator."""
    return task_type in ORCHESTRATOR_TASK_TYPES

def is_wgp_generation_task(task_type: str) -> bool:
    """Check if task type is handled by WGP generation."""
    return task_type in WGP_GENERATION_TASK_TYPES
```

#### 2.1.2 Update worker.py to use centralized definitions
**File:** `worker.py`
```python
# BEFORE (lines 100-111)
wgp_task_types = {
    "vace", "vace_21", "vace_22",
    ...
}

# AFTER
from source.task_types import WGP_GENERATION_TASK_TYPES
# Use WGP_GENERATION_TASK_TYPES instead of local set
```

#### 2.1.3 Update task_registry.py to use centralized definitions
**File:** `source/task_registry.py`
```python
# BEFORE (lines 56-64)
DIRECT_QUEUE_TASK_TYPES = {
    "wan_2_2_t2i", "vace", ...
}

# AFTER
from source.task_types import DIRECT_QUEUE_TASK_TYPES
# Remove local definition
```

#### 2.1.4 Update task_conversion.py to use centralized definitions
**File:** `source/task_conversion.py`
```python
# BEFORE (lines 248-277)
task_type_to_model = {
    "generate_video": "t2v",
    ...
}
model = task_type_to_model.get(task_type, "t2v")

# AFTER
from source.task_types import get_default_model, IMG2IMG_TASK_TYPES
model = get_default_model(task_type)
```

### 2.2 Centralize LoRA Directory Configuration

#### 2.2.1 Create LoRA path configuration
**New File:** `source/lora_paths.py`
```python
"""
Centralized LoRA directory configuration.

Single source of truth for where LoRA files are stored.
"""
from pathlib import Path
from typing import List

def get_wan_dir() -> Path:
    """Get the Wan2GP directory path."""
    return Path(__file__).parent.parent / "Wan2GP"

def get_all_lora_dirs() -> List[Path]:
    """Get all possible LoRA directories for searching."""
    wan_dir = get_wan_dir()
    repo_root = wan_dir.parent

    return [
        wan_dir / "loras",
        wan_dir / "loras" / "wan",
        wan_dir / "loras_i2v",
        wan_dir / "loras_hunyuan",
        wan_dir / "loras_hunyuan" / "1.5",
        wan_dir / "loras_hunyuan_i2v",
        wan_dir / "loras_flux",
        wan_dir / "loras_qwen",
        wan_dir / "loras_ltxv",
        wan_dir / "loras_kandinsky5",
        repo_root / "loras",
        repo_root / "loras" / "wan",
    ]

def get_lora_dir_for_model(model_type: str) -> Path:
    """Get the appropriate LoRA directory for a model type."""
    wan_dir = get_wan_dir()
    model_lower = (model_type or "").lower()

    if "wan" in model_lower or "vace" in model_lower:
        return wan_dir / "loras" / "wan"
    elif "hunyuan" in model_lower:
        if "i2v" in model_lower:
            return wan_dir / "loras_hunyuan_i2v"
        elif "1_5" in model_lower or "1.5" in model_lower:
            return wan_dir / "loras_hunyuan" / "1.5"
        else:
            return wan_dir / "loras_hunyuan"
    elif "flux" in model_lower:
        return wan_dir / "loras_flux"
    elif "qwen" in model_lower:
        return wan_dir / "loras_qwen"
    elif "ltx" in model_lower:
        return wan_dir / "loras_ltxv"
    elif "kandinsky" in model_lower:
        return wan_dir / "loras_kandinsky5"
    else:
        return wan_dir / "loras"
```

#### 2.2.2 Update lora_utils.py to use centralized paths
**File:** `source/lora_utils.py`
Replace inline path lists with imports from `lora_paths.py`

### Phase 2 Verification
```bash
# Verify new modules import correctly
python -c "from source.task_types import DIRECT_QUEUE_TASK_TYPES; print(len(DIRECT_QUEUE_TASK_TYPES))"
python -c "from source.lora_paths import get_all_lora_dirs; print(len(get_all_lora_dirs()))"

# Verify consumers still work
python -c "from source.task_registry import TaskRegistry; print('OK')"
python -c "from source.task_conversion import db_task_to_generation_task; print('OK')"
```

---

## Phase 3: Naming Standardization (Low-Medium Risk)

Establish and apply consistent naming conventions.

### 3.1 Document and Keep "sm_" Legacy

#### 3.1.1 Add documentation for task_handlers
**File:** `source/task_handlers/README.md` (new file)
```markdown
# SM Functions

"SM" stands for "State Machine" - these are orchestrator and multi-step task handlers
that manage complex workflows with multiple sub-tasks.

## Module Overview
- `travel_between_images.py` - Travel video orchestration
- `join_clips.py` - Video clip joining
- `edit_video_orchestrator.py` - Video editing workflows
- `magic_edit.py` - AI-powered image editing
```

#### 3.1.2 Remove sm_ import aliases (they add confusion, not clarity)
**File:** `source/task_handlers/travel_between_images.py`
```python
# BEFORE (lines 32-59)
from ..common_utils import (
    generate_unique_task_id as sm_generate_unique_task_id,
    get_video_frame_count_and_fps as sm_get_video_frame_count_and_fps,
    ...
)

# AFTER - use original names
from ..common_utils import (
    generate_unique_task_id,
    get_video_frame_count_and_fps,
    ...
)
```
Then update all usages from `sm_function_name` to `function_name`

### 3.2 Standardize Spelling (American English)

#### 3.2.1 Rename colour to color throughout
**Files to update:**
- `worker.py`: `colour_match_videos` → `color_match_videos`
- `source/task_registry.py`: same parameter
- `source/task_handlers/travel_between_images.py`: same parameter

**Approach:**
1. Add deprecation alias for backwards compatibility
2. Update internal usage to new name
3. Log warning if old name is used

```python
# In worker.py argparse
parser.add_argument("--color-match-videos", action="store_true")
parser.add_argument("--colour-match-videos", action="store_true",
    dest="color_match_videos", help="Deprecated: use --color-match-videos")
```

### 3.3 Standardize Parameter Naming

#### 3.3.1 Create parameter name aliases module
**New File:** `source/param_aliases.py`
```python
"""
Parameter name aliases for backwards compatibility.

When parameters are renamed, add the old name here so existing
API consumers continue to work.
"""

# Maps old parameter name -> new canonical name
PARAMETER_ALIASES = {
    # Legacy orchestrator payload names
    "full_orchestrator_payload": "orchestrator_details",
    "independent_segments": None,  # Removed, logic inverted to chain_segments

    # Spelling standardization
    "colour_match_videos": "color_match_videos",

    # Denoising parameter variants
    "denoise_strength": "denoising_strength",
    "strength": "denoising_strength",
}

def normalize_params(params: dict) -> dict:
    """Apply parameter aliases, returning dict with canonical names."""
    result = dict(params)
    for old_name, new_name in PARAMETER_ALIASES.items():
        if old_name in result and new_name and new_name not in result:
            result[new_name] = result[old_name]
    return result
```

### Phase 3 Verification
```bash
# Check no imports are broken
python -c "from source.task_handlers.travel_between_images import _handle_travel_orchestrator_task; print('OK')"

# Check aliases work
python -c "from source.param_aliases import normalize_params; print(normalize_params({'colour_match_videos': True}))"
```

---

## Phase 4: Major Refactoring (Higher Risk)

Break up giant functions into focused, testable components.

### 4.1 Refactor `_handle_travel_segment_via_queue`

This 900+ line function needs to become 5-10 smaller functions.

#### 4.1.1 Create TravelSegmentHandler class
**New File:** `source/travel_segment_handler.py`

```python
"""
Travel segment handling - broken into focused methods.

This replaces the monolithic _handle_travel_segment_via_queue function.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

from source.logging_utils import headless_logger
from source.params.structure_guidance import StructureGuidanceConfig


@dataclass
class TravelSegmentContext:
    """All context needed for processing a travel segment."""
    task_id: str
    segment_idx: int
    task_params: Dict[str, Any]
    main_output_dir_base: Path
    task_queue: Any  # HeadlessTaskQueue
    dprint: Callable
    color_match_videos: bool = False
    mask_active_frames: bool = True
    is_standalone: bool = False


@dataclass
class ResolvedSegmentParams:
    """Parameters resolved from multiple sources with precedence applied."""
    model_name: str
    prompt: str
    negative_prompt: str
    resolution: Tuple[int, int]
    total_frames: int
    seed: int
    processing_dir: Path
    debug_enabled: bool
    travel_mode: str
    use_svi: bool
    # Images
    start_ref_path: Optional[str] = None
    end_ref_path: Optional[str] = None
    svi_predecessor_video: Optional[str] = None


class TravelSegmentHandler:
    """
    Handles travel segment generation via the WGP queue.

    Replaces the monolithic _handle_travel_segment_via_queue function
    with focused, testable methods.
    """

    def __init__(self, context: TravelSegmentContext):
        self.ctx = context
        self.params = None  # Set by resolve_parameters()

    def handle(self) -> Tuple[bool, Optional[str]]:
        """
        Main entry point - orchestrates the full segment handling flow.

        Returns:
            Tuple of (success, output_path_or_error)
        """
        try:
            # Step 1: Resolve parameters from multiple sources
            self.params = self._resolve_parameters()

            # Step 2: Resolve images (start/end refs, SVI predecessor)
            self._resolve_images()

            # Step 3: Create guide/mask videos if needed
            guide_mask_result = self._create_guide_mask_videos()

            # Step 4: Build WGP generation parameters
            generation_params = self._build_generation_params(guide_mask_result)

            # Step 5: Apply mode-specific parameters (SVI, Uni3C, etc.)
            self._apply_mode_specific_params(generation_params)

            # Step 6: Submit to queue and wait
            return self._submit_and_wait(generation_params)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Travel segment {self.ctx.task_id}: Exception: {str(e)}"

    def _resolve_parameters(self) -> ResolvedSegmentParams:
        """
        Resolve parameters from multiple sources with proper precedence.

        Precedence: individual_params > segment_params > orchestrator_details
        """
        # Extract from context
        segment_params = self.ctx.task_params
        orchestrator_details = (
            segment_params.get("orchestrator_details") or
            segment_params.get("full_orchestrator_payload") or
            {}
        )
        individual_params = segment_params.get("individual_segment_params", {})

        # Helper for precedence
        def get_param(key, default=None, prefer_truthy=False):
            for source in [individual_params, segment_params, orchestrator_details]:
                if source and key in source:
                    value = source[key]
                    if value is not None:
                        if prefer_truthy and not isinstance(value, bool) and not value:
                            continue
                        return value
            return default

        # Resolve each parameter
        model_name = segment_params.get("model_name") or orchestrator_details.get("model_name", "")

        # Prompts - enhanced takes precedence
        enhanced_prompt = get_param("enhanced_prompt")
        base_prompt = get_param("base_prompt", " ")
        prompt = enhanced_prompt if enhanced_prompt and enhanced_prompt.strip() else base_prompt

        # Resolution
        res_str = segment_params.get("parsed_resolution_wh") or orchestrator_details.get("parsed_resolution_wh", "1280x720")
        # Parse and snap to grid (import helpers as needed)

        # ... continue for all parameters ...

        return ResolvedSegmentParams(
            model_name=model_name,
            prompt=prompt,
            negative_prompt=get_param("negative_prompt", " "),
            resolution=(1280, 720),  # Placeholder - implement parsing
            total_frames=get_param("num_frames") or 49,
            seed=get_param("seed_to_use", 12345),
            processing_dir=self.ctx.main_output_dir_base,
            debug_enabled=get_param("debug_mode_enabled", False),
            travel_mode=orchestrator_details.get("model_type", "vace"),
            use_svi=get_param("use_svi", False),
        )

    def _resolve_images(self) -> None:
        """Resolve start/end reference images and SVI predecessor."""
        # SVI mode uses predecessor video
        if self.params.use_svi and self.ctx.segment_idx > 0:
            self._resolve_svi_predecessor()

        # Resolve start/end images based on mode
        self._resolve_start_end_images()

    def _resolve_svi_predecessor(self) -> None:
        """Fetch and process SVI predecessor video for chaining."""
        # Implementation moved from main function
        pass

    def _resolve_start_end_images(self) -> None:
        """Resolve start and end reference images."""
        # Implementation moved from main function
        pass

    def _create_guide_mask_videos(self) -> Dict[str, Any]:
        """Create guide and mask videos if needed for VACE/structure guidance."""
        # Check if we need to run TravelSegmentProcessor
        structure_config = StructureGuidanceConfig.from_params(self.ctx.task_params)

        if self.params.travel_mode == "vace" or structure_config.has_guidance:
            # Run processor and return results
            pass

        return {}

    def _build_generation_params(self, guide_mask_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build the base WGP generation parameters."""
        return {
            "model_name": self.params.model_name,
            "negative_prompt": self.params.negative_prompt,
            "resolution": f"{self.params.resolution[0]}x{self.params.resolution[1]}",
            "video_length": self.params.total_frames,
            "seed": self.params.seed,
        }

    def _apply_mode_specific_params(self, params: Dict[str, Any]) -> None:
        """Apply SVI, Uni3C, or other mode-specific parameters."""
        if self.params.use_svi:
            self._apply_svi_params(params)

        # Check for Uni3C
        # ...

    def _apply_svi_params(self, params: Dict[str, Any]) -> None:
        """Apply SVI-specific generation parameters."""
        params["svi2pro"] = True
        params["video_prompt_type"] = "I"
        params["sliding_window_overlap"] = 4
        # ... rest of SVI setup

    def _submit_and_wait(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Submit task to queue and wait for completion."""
        from headless_model_management import GenerationTask

        params["_source_task_type"] = "travel_segment"
        generation_task = GenerationTask(
            id=self.ctx.task_id,
            model=self.params.model_name,
            prompt=self.params.prompt,
            parameters=params
        )

        self.ctx.task_queue.submit_task(generation_task)

        # Wait loop
        max_wait = 1800
        elapsed = 0
        while elapsed < max_wait:
            status = self.ctx.task_queue.get_task_status(self.ctx.task_id)
            if status is None:
                return False, f"Task status became None"

            if status.status == "completed":
                return self._handle_completion(status.result_path)
            elif status.status == "failed":
                return False, f"Generation failed: {status.error_message}"

            import time
            time.sleep(2)
            elapsed += 2

        return False, "Generation timeout"

    def _handle_completion(self, result_path: str) -> Tuple[bool, Optional[str]]:
        """Handle successful completion, including chaining if needed."""
        if self.ctx.is_standalone:
            return True, result_path

        # Run chaining for orchestrator mode
        # ...
        return True, result_path


# Wrapper function for backwards compatibility
def _handle_travel_segment_via_queue(
    task_params_dict,
    main_output_dir_base: Path,
    task_id: str,
    colour_match_videos: bool,
    mask_active_frames: bool,
    task_queue,
    dprint_func,
    is_standalone: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Backwards-compatible wrapper for TravelSegmentHandler.

    DEPRECATED: Use TravelSegmentHandler directly for new code.
    """
    context = TravelSegmentContext(
        task_id=task_id,
        segment_idx=task_params_dict.get("segment_index", 0),
        task_params=task_params_dict,
        main_output_dir_base=main_output_dir_base,
        task_queue=task_queue,
        dprint=dprint_func,
        color_match_videos=colour_match_videos,
        mask_active_frames=mask_active_frames,
        is_standalone=is_standalone,
    )

    handler = TravelSegmentHandler(context)
    return handler.handle()
```

#### 4.1.2 Update task_registry.py to use new handler
**File:** `source/task_registry.py`
```python
# Replace the giant function with import
from source.travel_segment_handler import _handle_travel_segment_via_queue
```

### 4.2 Simplify WGP Initialization in headless_wgp.py

#### 4.2.1 Extract monkeypatching to separate module
**New File:** `source/wgp_patches.py`
```python
"""
WGP monkeypatches for headless operation.

These patches adapt WGP for headless/programmatic use.
Each patch is documented with why it's needed.
"""

def apply_qwen_family_patches(wgp_module):
    """
    Patch WGP to support Qwen model family.

    Why: WGP's load_wan_model doesn't natively handle Qwen models.
    This routes Qwen-family models to their dedicated handler.
    """
    _orig_load_wan_model = wgp_module.load_wan_model

    def _patched_load_wan_model(model_filename, model_type, base_model_type, model_def, **kwargs):
        # ... patch implementation ...
        pass

    wgp_module.load_wan_model = _patched_load_wan_model


def apply_lora_directory_patches(wgp_module, wan_root: str):
    """
    Patch LoRA directory resolution for Qwen models.

    Why: Qwen models need their own LoRA directory (loras_qwen/).
    """
    # ... implementation ...
    pass


def apply_all_patches(wgp_module, wan_root: str):
    """Apply all WGP patches for headless operation."""
    apply_qwen_family_patches(wgp_module)
    apply_lora_directory_patches(wgp_module, wan_root)
```

#### 4.2.2 Simplify WanOrchestrator.__init__
**File:** `headless_wgp.py`
Replace inline patching with:
```python
from source.wgp_patches import apply_all_patches
# ... later in __init__ ...
apply_all_patches(wgp, self.wan_root)
```

### Phase 4 Verification
```bash
# Run existing tests
pytest tests/ -v

# Test new handler directly
python -c "from source.travel_segment_handler import TravelSegmentHandler; print('OK')"

# Integration test (if available)
python create_test_task.py --task-type travel_segment --dry-run
```

---

## Phase 5: Simplify Complex Code (Medium Risk)

Reduce complexity in key areas without changing behavior.

### 5.1 Simplify Logging Function Redefinition

#### 5.1.1 Use decorators instead of manual redefinition
**File:** `source/logging_utils.py`

```python
# BEFORE (lines 489-522) - confusing redefinition pattern
_original_essential = essential
def essential(component: str, message: str, task_id: Optional[str] = None):
    _original_essential(component, message, task_id)
    _intercept_log(...)

# AFTER - use a decorator pattern
def with_interception(func):
    """Decorator to add log interception to logging functions."""
    @functools.wraps(func)
    def wrapper(component: str, message: str, task_id: Optional[str] = None):
        result = func(component, message, task_id)
        _intercept_log(func.__name__.upper(), f"{component}: {message}", task_id)
        return result
    return wrapper

# Apply to all logging functions
essential = with_interception(essential)
success = with_interception(success)
warning = with_interception(warning)
error = with_interception(error)
```

### 5.2 Simplify ALSA Suppression

#### 5.2.1 Extract to utility function
**File:** `worker.py`

```python
# BEFORE (lines 22-37) - nested try/except mess
try:
    import ctypes
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(...)
    def py_error_handler(...):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except:
        pass
except:
    pass

# AFTER - move to utility module
# source/platform_utils.py
def suppress_alsa_errors():
    """Suppress ALSA error messages on Linux."""
    try:
        import ctypes
        handler_type = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
        )
        null_handler = handler_type(lambda *args: None)
        libasound = ctypes.cdll.LoadLibrary('libasound.so.2')
        libasound.snd_lib_error_set_handler(null_handler)
    except (OSError, AttributeError):
        pass  # ALSA not available or not on Linux

# In worker.py
from source.platform_utils import suppress_alsa_errors
suppress_alsa_errors()
```

### 5.3 Reduce SVI2Pro Patching Complexity

#### 5.3.1 Create SVI patch helper
**File:** `source/svi_patches.py`
```python
"""
SVI (Stable Video Infinity) model patching utilities.

SVI requires patching model definitions at runtime to enable
sliding window generation with overlapped latents.
"""

@dataclass
class SVIPatchState:
    """Tracks which patches have been applied for cleanup."""
    model_key: str
    models_def_patched: bool = False
    wan_model_patched: bool = False
    original_svi2pro: Any = None
    original_sliding_window: Any = None


def apply_svi_patches(wgp_module, model_key: str, task_id: str, logger) -> SVIPatchState:
    """
    Apply SVI patches to model definition.

    Returns state object for later restoration.
    """
    state = SVIPatchState(model_key=model_key)

    # Patch models_def
    if model_key in wgp_module.models_def:
        state.original_svi2pro = wgp_module.models_def[model_key].get("svi2pro")
        wgp_module.models_def[model_key]["svi2pro"] = True
        wgp_module.models_def[model_key]["sliding_window"] = True
        wgp_module.models_def[model_key]["sliding_window_defaults"] = {"overlap_default": 4}
        state.models_def_patched = True

    # Patch wan_model.model_def if loaded
    if hasattr(wgp_module, 'wan_model') and wgp_module.wan_model:
        model_def = getattr(wgp_module.wan_model, 'model_def', None)
        if model_def:
            state.original_sliding_window = model_def.get("sliding_window")
            model_def["svi2pro"] = True
            model_def["sliding_window"] = True
            model_def["sliding_window_defaults"] = {"overlap_default": 4}
            model_def["svi_empty_frames_mode"] = "zeros"
            state.wan_model_patched = True

    logger.info(f"[SVI2PRO] Applied patches to {model_key}", task_id=task_id)
    return state


def restore_svi_patches(wgp_module, state: SVIPatchState, task_id: str, logger):
    """Restore model definition to pre-patch state."""
    if state.models_def_patched and state.model_key in wgp_module.models_def:
        if state.original_svi2pro is None:
            wgp_module.models_def[state.model_key].pop("svi2pro", None)
        else:
            wgp_module.models_def[state.model_key]["svi2pro"] = state.original_svi2pro
        wgp_module.models_def[state.model_key].pop("sliding_window", None)

    # Similar for wan_model.model_def...

    logger.info(f"[SVI2PRO] Restored patches for {state.model_key}", task_id=task_id)
```

### Phase 5 Verification
```bash
# Test logging still works
python -c "from source.logging_utils import essential; essential('TEST', 'message'); print('OK')"

# Test SVI patches
python -c "from source.svi_patches import SVIPatchState; print('OK')"
```

---

## Phase 6: Code Quality Improvements (Lower Risk)

Polish and documentation improvements.

### 6.1 Add Type Hints to Key Modules

Priority order:
1. `source/task_types.py` (new, already typed)
2. `source/db_operations.py` - public functions
3. `source/task_registry.py` - handler signatures
4. `headless_model_management.py` - HeadlessTaskQueue methods

### 6.2 Add Module-Level Docstrings

Every file should have a docstring explaining:
- What the module does
- Key classes/functions
- Usage examples (for complex modules)

### 6.3 Create Architecture Documentation

**New File:** `docs/ARCHITECTURE.md`
```markdown
# Headless-Wan2GP Architecture

## Overview
[High-level description]

## Component Diagram
[ASCII diagram of major components]

## Data Flow
[How tasks flow through the system]

## Key Design Decisions
[Why certain patterns were chosen]
```

### Phase 6 Verification
```bash
# Type check with mypy (if configured)
mypy source/task_types.py

# Documentation generation test
python -c "import source.task_types; help(source.task_types)"
```

---

## Implementation Schedule

### Day 1: Phase 1 (Safe Cleanup)
- 1.1 Remove dead code (1 hour)
- 1.2 Fix bugs (30 min)
- 1.3 Fix comments (30 min)
- 1.4 Remove hardcoded creds (1 hour)
- Verification (30 min)

### Day 2: Phase 2 (Centralization)
- 2.1 Create task_types.py (2 hours)
- 2.1.2-2.1.4 Update consumers (2 hours)
- 2.2 Create lora_paths.py (1 hour)
- Verification (30 min)

### Day 3: Phase 3 (Naming)
- 3.1 Document task_handlers (1 hour)
- 3.1.2 Remove sm_ aliases (2 hours)
- 3.2 Standardize spelling (1 hour)
- 3.3 Create param_aliases.py (1 hour)
- Verification (30 min)

### Day 4-5: Phase 4 (Major Refactoring)
- 4.1 Create TravelSegmentHandler (4-6 hours)
- 4.1.2 Update task_registry.py (1 hour)
- 4.2 Create wgp_patches.py (2 hours)
- 4.2.2 Simplify WanOrchestrator (2 hours)
- Verification (1-2 hours)

### Day 6: Phase 5 (Simplification)
- 5.1 Simplify logging (2 hours)
- 5.2 Simplify ALSA suppression (1 hour)
- 5.3 Create svi_patches.py (2 hours)
- Verification (1 hour)

### Day 7: Phase 6 (Polish)
- 6.1 Add type hints (3 hours)
- 6.2 Add docstrings (2 hours)
- 6.3 Create architecture docs (2 hours)

---

## Risk Mitigation

### Before Starting
1. Create a backup branch: `git checkout -b pre-refactor-backup`
2. Ensure all tests pass: `pytest tests/ -v`
3. Document current behavior for critical paths

### During Refactoring
1. Commit after each sub-task
2. Run verification after each phase
3. Keep backwards compatibility wrappers initially

### If Something Breaks
1. Each phase is independent - can revert single phase
2. Wrapper functions maintain old API signatures
3. Git bisect can identify breaking commits

---

## Success Criteria

After completion:
- [ ] No function exceeds 200 lines
- [ ] All task type definitions come from single source
- [ ] No duplicated path/directory lists
- [ ] All public functions have type hints
- [ ] All modules have docstrings
- [ ] Tests still pass
- [ ] No security credentials in code
