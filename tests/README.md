# Test Suite

**69 passed, 2 skipped** across headless + GPU tests.

## Quick Run

```bash
# Headless only (no GPU, ~4s)
python -m pytest tests/test_ltx2_pose_smoke.py tests/test_ltx2_headless.py tests/test_task_conversion_headless.py -v

# GPU tests (requires model weights + test media in Wan2GP/)
python -m pytest tests/test_ic_lora_gpu.py -v -s

# Everything
python -m pytest tests/test_ltx2_pose_smoke.py tests/test_ltx2_headless.py tests/test_task_conversion_headless.py tests/test_ic_lora_gpu.py -v -s
```

**GPU test prerequisites:** Place `vid1.mp4` and `img1.png` in the `Wan2GP/` directory. Model weights download automatically on first run via `preload_URLs`.

## Test Summary

| File | Tests | What it covers |
|------|-------|----------------|
| `test_ltx2_pose_smoke.py` | 8 | MediaPipe pose extraction, control signal wiring, video_prompt_type mapping |
| `test_ltx2_headless.py` | 29 | LTX-2 model detection, v10.x param passthrough, image/audio bridging, travel toggle |
| `test_task_conversion_headless.py` | 30 | DB task → GenerationTask pipeline, phase config parsing, model override priority |
| `test_ic_lora_gpu.py` | 1+1skip | IC-LoRA pose workflow end-to-end (GPU), union control LoRA loading |

### Other test files (pre-existing)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_service_health.py` | 12 | Task type set sizes, handler coverage, JSON config validity |
| `test_all_services_headless.py` | 67 | Registration for all 24 task types, model detection methods, helper functions |
| `test_wan_headless.py` | 20 | T2V/I2V/VACE/Flux/Hunyuan smoke, model switching rotation |
| `test_ltx_headless.py` | 18 | LTXv detection, cross-model switching (LTX2 ↔ LTXv ↔ T2V) |

## Bug Fixes Found During Testing

| Bug | Fix | File |
|-----|-----|------|
| `apply_changes` import crash | Replaced with `get_default_settings` + `set_model_settings` API | `orchestrator.py` |
| 12 missing v10.x params (`alt_prompt`, `duration_seconds`, `audio_scale`, `self_refiner_*`, etc.) | Added to both passthrough and normal mode builders | `wgp_params.py` |
| Union control LoRA 404 (wrong filename) | Corrected to `ltx-2-19b-ic-lora-union-control-ref0.5.safetensors` | `ltx2_19B.json` |
| `NoneType / int` crash on control_net_weight2 | Moved default assignment outside `if is_vace:` block | `orchestrator.py` |

## IC-LoRA Pipeline Insight

IC-LoRA weights (pose, depth, canny, union) load as standard LoRA adapters and work. However the dedicated `ICLoraPipeline` — with reference downscale and video conditioning guide injection (ComfyUI equivalent: `LTXICLoRALoaderModelOnly` + `LTXAddVideoICLoRAGuide`) — is **not active**. The config defaults to `two_stage` pipeline.

To enable: add `"ltx2_pipeline": "ic_lora"` in `Wan2GP/defaults/ltx2_19B.json` model definition.

## Key Source Files

| File | Role |
|------|------|
| `source/models/wgp/orchestrator.py` | WanOrchestrator — model init, generate() dispatch |
| `source/models/wgp/generators/wgp_params.py` | Parameter dict builders (passthrough + normal mode) |
| `source/task_types.py` | Task type sets, model mappings |
| `source/task_conversion.py` | DB task → GenerationTask, phase config parsing |
| `Wan2GP/defaults/ltx2_19B.json` | LTX-2 19B model config (URLs, LoRAs, preloads) |
| `Wan2GP/models/ltx2/ltx2.py` | LTX-2 model class, pipeline selection logic |
