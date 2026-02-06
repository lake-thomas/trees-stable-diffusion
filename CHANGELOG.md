# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-06

### Added
- Initial release of Trees SD package
- Support for Stable Diffusion 1.5 and 3.5 fine-tuning with LoRA
- Dataset loaders for iNaturalist and Autoarborist formats
- Unified training interface with comprehensive configuration options
- Command-line interface (`trees-sd-train`) for easy training
- Example configuration files for different scenarios
- Example Python scripts demonstrating various use cases
- Comprehensive README with installation and usage instructions
- Basic test structure for package validation
- Utilities for dataset validation and training time estimation

### Features
- LoRA-based parameter-efficient fine-tuning
- Support for both SD1.5 and SD3.5 models
- Flexible dataset handling (JSON metadata or direct image scanning)
- Hyperparameter configuration via CLI or YAML files
- Mixed precision training support (fp16, bf16)
- XFormers memory-efficient attention support
- Gradient accumulation for large effective batch sizes
- Regular checkpoint saving during training
- Accelerate integration for distributed training
