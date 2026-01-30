# source/task_handlers/__init__.py
"""Specialized task handlers for complex multi-step operations.

This package contains handlers for tasks that involve multiple generation steps,
external service calls, or complex video processing pipelines.

Handlers included:
- travel_between_images: Multi-segment travel video generation with SVI chaining
- join_clips: Video clip joining with AI-generated transitions
- join_clips_orchestrator: High-level coordination for join clips workflow
- edit_video_orchestrator: Video editing workflow coordination
- magic_edit: AI-powered image editing via Replicate API
- inpaint_frames: Frame-level video inpainting using VACE
- create_visualization: Debug visualization generation

Note: This package was formerly called "sm_functions" (Steerable Motion).
"""
