# ComfyUI FluxFlow Plugin - Changelog

All notable changes to the FluxFlow ComfyUI plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

_No unreleased changes._


## [0.10.0] - 2026-06-14

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

- **`fluxflow` dependency** bumped to `>=0.10.0` (per-token text encoder and
  v0.10.0 flow processor are required).


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
