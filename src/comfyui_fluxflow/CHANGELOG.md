# ComfyUI FluxFlow Plugin - Changelog

## [0.1.0] - 2025-01-13

### Added
- **Color-coded connectors for FluxFlow custom types**
  - JavaScript extension (`web/fluxflow_types.js`) registers custom type colors
  - Visual identification for different data types:
    - Purple: FLUXFLOW_MODEL
    - Green: FLUXFLOW_TEXT_ENCODER
    - Dark Green: FLUXFLOW_TOKENIZER
    - Amber: FLUXFLOW_CONDITIONING
    - Blue: FLUXFLOW_LATENT
  - Enhanced debugging with comprehensive console logging
  - Added `WEB_EXTENSION_README.md` documenting extension capabilities and limitations

### Fixed
- **Critical**: Fixed workflow reload error with ComfyUI-Impact-Pack
  - Changed `use_context` from BOOLEAN to COMBO type (["true", "false"])
  - Resolves "Cannot delete property 'value' of #<BooleanWidget2>" error
  - Workflows now save and reload correctly with Impact-Pack installed
- **FluxFlowModelLoader now persists checkpoint path in saved workflows**
  - Added `multiline: False` and `dynamicPrompts: False` to checkpoint_path
  - Added placeholder text for better UX
  - Checkpoint path now saves and restores correctly when reloading workflows
  - Applied same fix to tokenizer_name input

### Known Limitations
- **Auto-suggest does NOT work for custom types** (ComfyUI core limitation)
  - ComfyUI only supports auto-suggest for built-in types (IMAGE, LATENT, MODEL, etc.)
  - Custom types like FLUXFLOW_MODEL are not recognized by ComfyUI's type matching
  - Users must add FluxFlow nodes via right-click menu or search
  - This is intentional - custom types provide type safety and prevent incompatible connections
  - See `WEB_EXTENSION_README.md` for detailed explanation and alternatives

### Previous Changes

### Added
- Initial release
- 6 complete nodes for ComfyUI integration
- 14 scheduler options (DPM++, Euler, DDIM, LCM, etc.)
- Automatic configuration detection from checkpoints
- Complete VAE encode/decode support
- Text conditioning with DistilBERT
- Comprehensive documentation (README, INSTALL, guides)
- Native ComfyUI tensor format integration
- Type hints and error handling throughout

### Features
- Auto-detects: VAE dim, flow dim, text embed dim, architecture params
- Supports any FluxFlow model size
- Reproducible generation with seed control
- Context conditioning toggle
- Multiple prediction types (v_prediction, epsilon, sample)

