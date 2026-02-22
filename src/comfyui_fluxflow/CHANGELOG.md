# ComfyUI FluxFlow Plugin - Changelog

All notable changes to the FluxFlow ComfyUI plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

_No unreleased changes._


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
