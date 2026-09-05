# ComfyUI FluxFlow Plugin - Changelog

All notable changes to the FluxFlow ComfyUI plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Not yet released — work in progress toward v0.10.0.

### Added (Experimental)
- **AMD ROCm/gfx1151 support (experimental, unvalidated)**: `get_device_auto`/
  `parse_device` now delegate to `fluxflow.utils.device`, which distinguishes
  ROCm from real NVIDIA CUDA. No node-visible behavior change — ROCm-build
  PyTorch already routed through the existing CUDA code path. See
  `docs/ROCM.md` in fluxflow-core.
- NPU (XDNA) acceleration was evaluated and deferred; not implemented.

### Changed (BREAKING — workflow re-wiring required)

- **New `FLUXFLOW_TEXT` ComfyUI type** replaces `FLUXFLOW_CONDITIONING`
  - `FluxFlowTextEncode` and `FluxFlowTextEncodeNegative` now return
    `((text_seq, text_mask),)` typed as `("FLUXFLOW_TEXT",)` — a per-token
    embedding tuple rather than a pooled vector.
  - `FluxFlowSampler` input renamed `conditioning` → `text` (positive) and the
    optional negative input is `negative_text`, both typed `FLUXFLOW_TEXT`.
  - The pooled `FLUXFLOW_CONDITIONING` type is removed from the v0.10.0 nodes.
- **Per-token vs pooled dispatch** in the sampler via
  `fluxflow.models.pipeline._flow_processor_takes_pertoken_text`:
  v0.10.0 flow processors receive `(packed, text_seq, text_mask, timesteps)`;
  legacy v0.6/0.7/0.8 processors receive a masked-mean-pooled `[B, E]` tensor
  via `_masked_mean_pool` for backwards compatibility.

### Migration

- **Workflows saved with v0.8.x will fail to load** with a clear ComfyUI
  type-mismatch error where TextEncode previously wired into
  `FluxFlowSampler.conditioning`.
- Re-wire `FluxFlowTextEncode.text` → `FluxFlowSampler.text` (and the
  negative encoder into `negative_text` if using CFG).
- See `TROUBLESHOOTING.md` ("Workflow won't load after v0.10.0 upgrade") for
  the diagnostic and the full re-wiring procedure.

### Known follow-ups

- CFG null condition currently uses `zeros_like(text_seq)` with the positive
  mask rather than an encoded empty prompt; this matches the legacy null path
  but is intentionally not the encoded-empty-prompt path used elsewhere in
  fluxflow-core. Tracked for a future release.

### Updated

- **`fluxflow` dependency** pinned to the
  `fluxflow-core@feature/model-v0.10.0` branch during development (per-token
  text encoder and v0.10.0 flow processor are required); this will become
  `fluxflow>=0.10.0` after the PyPI release.

### Fixed
- **Train/inference max-length mismatch**: `FluxFlowTextEncode` hardcoded
  `max_length=512` while `fluxflow-training` actually caps captions at 32
  tokens with no override and no truncation logging — the model was only
  ever trained on the first ~32 tokens, but the node silently allowed much
  longer prompts. Now imports a shared `DEFAULT_MAX_TEXT_LENGTH` constant
  from `fluxflow-core` (falls back to `32` if the installed `fluxflow-core`
  predates the constant), so this node can't silently diverge from
  training's actual max length again.
- `FluxFlowModelLoader` could not load v0.10.0 checkpoints at all: the
  legacy-detection heuristic checked for v0.8.0 (`pillar_cross_attn`/
  `film_p0`) and v0.7.0 markers first, and v0.10.0's `FluxTransformerBlock_v100`
  contains `film_p0_text`/`film_p0_time` (inherited naming), so v0.10.0
  checkpoints always hit the "v0.8.0 detected but requires metadata" error
  path. The node now checks `fluxflow.models.detect_architecture_version`
  first and constructs a proper v0.10.0 model (previously only v0.3.0 had a
  construct-and-load fallback; v0.7.0/v0.8.0 just errored).
- The `config_info` version string always showed a stale/default value
  (`getattr(pipeline, "version", ...)` — no such attribute is ever set) —
  now re-detected from the checkpoint's own state-dict keys for display.
- **Dtype mismatch when running inference with a non-fp32 model load**:
  `FluxFlowModelLoader`'s `dtype` option (fp16/bf16) cast the model's
  weights, but `FluxFlowVAEEncode`, `FluxFlowVAEDecode`, and
  `FluxFlowSampler` only moved their input tensors to the model's *device*,
  never its *dtype* — raising `RuntimeError: expected scalar type Float but
  found BFloat16/Half` on the first VAE/flow-processor call. These nodes now
  cast floating-point tensors (image/latent/text_seq) to match the loaded
  model's dtype at the same point they're moved to its device; integer/bool
  tensors (e.g. `text_mask`) are left untouched. New `to_model_dtype()`
  helper in `nodes/utils.py`.
- **Dtype mismatch missed by the above fix — per-step timestep batch**:
  `FluxFlowSampler`'s denoising loop still built `t_batch` hardcoded to
  `torch.float32` on every step, so a bf16/fp16 model load still raised
  `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and
  BFloat16/Half` in the flow processor's time-embedding MLP
  (`time_mlp(time_emb)`). `t_batch` is now cast with the same
  `to_model_dtype()` helper right after construction.
- **Decoded images returned in the model's inference dtype instead of
  float32**: `FluxFlowVAEDecode` handed ComfyUI's `SaveImage`/`PreviewImage`
  nodes an image tensor still in whatever dtype the model decoded in
  (fp16/bf16 with the `dtype` load option above), but ComfyUI's `IMAGE`
  contract is always float32 — its own `nodes.py` does
  `image.cpu().numpy()`, and numpy has no bfloat16 support, raising
  `TypeError: Got unsupported ScalarType BFloat16`. `flux_image_to_comfy` in
  `nodes/utils.py` now always casts its output to float32 regardless of
  input dtype. `comfy_image_to_flux` (the encode direction) is intentionally
  left unchanged, since ComfyUI always supplies float32 images already.

### Added
- Optional `text_encoder_path` input on `FluxFlowModelLoader`, overriding
  both the auto-discovered sibling `text_encoder.safetensors` and the
  checkpoint's bundled copy.
- Optional `dtype` input (`fp32`/`fp16`/`bf16`, default `fp32`) on
  `FluxFlowModelLoader`, matching the fp16/bf16 options already available
  for training. Weights still load in their native dtype; the cast is
  applied at the existing device-move step, so `fp32` behaves exactly as
  before.

## [0.8.0] - 2026-02-21

### Added
- **v0.8.0 checkpoint detection** in `FluxFlowModelLoader`
  - Detects pillar-attention weights (`pillar_cross_attn`, `film_p0` keys) in state dict
  - Returns a clear error message directing users to use versioned loading for v0.8.0 models
  - Prevents silent architecture mismatches when loading v0.8.0 checkpoints

### Changed
- **Updated `fluxflow` dependency** to `>=0.8.0`
- Version bumped to 0.8.0

<!-- Versions 0.2.0–0.7.x were internal development iterations aligned with fluxflow-core
     architecture changes. No separate ComfyUI plugin releases were made for those versions. -->

## [0.1.0] - 2025-01-13

### Added
- Initial release: 6 complete nodes for ComfyUI integration
- 14 scheduler options (DPM++, Euler, DDIM, LCM, etc.)
- Automatic configuration detection from checkpoints (VAE dim, flow dim, text embed dim)
- Complete VAE encode/decode support
- Text conditioning with DistilBERT
- Comprehensive documentation (README, INSTALL, guides)
- Native ComfyUI tensor format integration
- Type hints and error handling throughout
- Reproducible generation with seed control
- Context conditioning toggle
- Multiple prediction types (v_prediction, epsilon, sample)
- **Color-coded connectors for FluxFlow custom types** via JavaScript extension (`src/comfyui_fluxflow/web/fluxflow_types.js`)
  - Purple: FLUXFLOW_MODEL, Green: FLUXFLOW_TEXT_ENCODER, Dark Green: FLUXFLOW_TOKENIZER
  - Amber: FLUXFLOW_CONDITIONING, Blue: FLUXFLOW_LATENT

### Fixed
- Fixed workflow reload error with ComfyUI-Impact-Pack
  - Changed `use_context` from BOOLEAN to COMBO type (`["true", "false"]`)
  - Resolves "Cannot delete property 'value' of #<BooleanWidget2>" error
  - Workflows now save and reload correctly with Impact-Pack installed
- **FluxFlowModelLoader** now persists checkpoint path in saved workflows
  - Added `multiline: False` and `dynamicPrompts: False` to checkpoint_path and tokenizer_name
  - Checkpoint path now saves and restores correctly when reloading workflows
