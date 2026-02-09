# Test Suite Documentation

This document provides comprehensive documentation for the test suite, with a focus on the newly added tests for data validation and error handling.

## Overview

The test suite is organized to validate different aspects of the Trees Stable Diffusion package:

1. **Basic Tests** (`test_basic.py`) - Package structure and imports
2. **Evaluation Tests** (`test_evaluation.py`) - Image quality evaluation
3. **Generation Tests** (`test_generation.py`) - Image generation utilities
4. **Data Validation Tests** (`test_data_validation.py`) - NEW: Comprehensive data type and validation tests
5. **Error Handling Tests** (`test_error_handling.py`) - NEW: Dataset loading and error handling tests

## Running Tests

### Run All Tests
```bash
cd /path/to/trees-stable-diffusion
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
# Run data validation tests
python -m pytest tests/test_data_validation.py -v

# Run error handling tests
python -m pytest tests/test_error_handling.py -v

# Run evaluation tests
python -m pytest tests/test_evaluation.py -v
```

### Run Specific Test Classes
```bash
# Run only corrupted image tests
python -m pytest tests/test_data_validation.py::TestCorruptedImages -v

# Run only edge case tests
python -m pytest tests/test_data_validation.py::TestEdgeCases -v
```

### Run Specific Tests
```bash
# Run a single test
python -m pytest tests/test_data_validation.py::TestImageDataTypes::test_supported_image_formats -v
```

## New Test Modules

### test_data_validation.py

This module contains comprehensive tests for data validation, covering data types, corrupted images, and variable ranges.

#### Test Classes and Coverage

**1. TestImageDataTypes** - Image format and data type handling
- ✅ `test_supported_image_formats` - Validates all supported image formats (JPEG, PNG, BMP, WebP)
- ✅ `test_collect_image_paths_filters_by_extension` - Ensures only valid image files are collected
- ✅ `test_case_insensitive_extension_matching` - Validates case-insensitive extension handling
- ✅ `test_rgba_to_rgb_conversion` - Tests RGBA to RGB conversion

**2. TestCorruptedImages** - Corrupted and invalid image handling
- ✅ `test_empty_file_raises_error` - Validates handling of empty files
- ✅ `test_truncated_image_file` - Tests truncated/incomplete image files
- ✅ `test_invalid_image_data` - Tests files with invalid image data
- ✅ `test_non_image_file_with_image_extension` - Tests non-image files with image extensions

**3. TestVariableRanges** - Variable range validation
- ✅ `test_very_small_images` - Tests handling of very small images (1x1 to 8x8 pixels)
- ✅ `test_very_large_images` - Tests handling of large images (up to 2048x2048)
- ✅ `test_non_square_aspect_ratios` - Tests various aspect ratios (2:1, 1:2, 4:3, 16:9, etc.)
- ✅ `test_rgb_value_extremes` - Tests extreme RGB values (pure black, white, primary colors)
- ✅ `test_basic_stats_with_varied_dimensions` - Tests statistics computation with varied dimensions

**4. TestEdgeCases** - Edge case handling
- ✅ `test_empty_directory_raises_error` - Empty directory detection
- ✅ `test_nonexistent_directory_raises_error` - Nonexistent directory handling
- ✅ `test_directory_with_only_unsupported_files` - Directory with no valid images
- ✅ `test_single_image_in_directory` - Single image handling
- ✅ `test_nested_directory_structure` - Nested directory traversal
- ✅ `test_mismatched_real_and_generated_counts` - Different set sizes
- ✅ `test_special_characters_in_filenames` - Special characters in filenames
- ✅ `test_zero_dimensions_image` - Zero-dimension image edge case
- ✅ `test_rgb_mean_computation_correctness` - RGB mean value accuracy

**5. TestBatchProcessing** - Batch processing validation
- ✅ `test_various_batch_sizes` - Tests with different batch sizes (1, 2, 5, 10, 32)
- ✅ `test_device_parameter_validation` - Device parameter acceptance
- ✅ `test_num_workers_parameter` - Number of workers parameter

**Total: 25 tests covering comprehensive data validation scenarios**

### test_error_handling.py

This module contains tests for error handling, dataset loading, and parameter validation.

#### Test Classes and Coverage

**1. TestDatasetParameterValidation** - Parameter validation
- ⚠️ `test_invalid_dataset_type_raises_error` - Invalid dataset type rejection
- ⚠️ `test_valid_dataset_types_accepted` - Valid dataset type acceptance
- ⚠️ `test_max_size_parameter_validation` - Max size parameter validation

**2. TestMetadataValidation** - Metadata file validation
- ⚠️ `test_valid_inaturalist_metadata` - Valid iNaturalist metadata loading
- ⚠️ `test_malformed_json_metadata` - Malformed JSON handling
- ⚠️ `test_metadata_with_missing_image_files` - Missing image file handling
- ⚠️ `test_valid_autoarborist_annotations` - Valid Autoarborist annotations
- ⚠️ `test_empty_metadata_file` - Empty metadata file handling
- ⚠️ `test_metadata_with_missing_fields` - Missing field handling

**3. TestFileIOErrors** - File I/O error handling
- ⚠️ `test_nonexistent_data_directory` - Nonexistent directory error handling
- ⚠️ `test_permission_denied_scenarios` - Permission error handling (skipped on most systems)
- ⚠️ `test_read_only_directory` - Read-only directory handling

**4. TestDatasetLoadingEdgeCases** - Dataset loading edge cases
- ⚠️ `test_no_metadata_fallback_to_scanning` - Fallback to directory scanning
- ⚠️ `test_mixed_image_formats_in_dataset` - Mixed format handling
- ⚠️ `test_dataset_with_subdirectories` - Subdirectory handling
- ⚠️ `test_dataset_getitem_returns_expected_keys` - Return value validation
- ⚠️ `test_image_resizing_behavior` - Image resizing validation
- ⚠️ `test_caption_generation_fallback` - Caption generation fallback

**5. TestInputSanitization** - Input sanitization
- ⚠️ `test_string_path_conversion` - Path object and string handling
- ⚠️ `test_dataset_type_case_insensitivity` - Case-insensitive type matching

**6. TestGenerationHelpers** - Generation helper functions
- ⚠️ `test_normalize_dataset_type` - Dataset type normalization
- ⚠️ `test_generation_prompts_type_validation` - Generation prompt validation
- ⚠️ `test_negative_prompt_validation` - Negative prompt validation

**Total: 23 tests (⚠️ may be skipped if torch is not available)**

## Test Coverage Summary

### What is Tested

✅ **Image Data Types**
- JPEG, PNG, BMP, WebP format support
- Case-insensitive extension matching
- RGBA to RGB conversion

✅ **Corrupted/Invalid Images**
- Empty files
- Truncated files
- Invalid image data
- Non-image files with image extensions

✅ **Variable Ranges**
- Image dimensions: 1x1 to 2048x2048 pixels
- Various aspect ratios: square, wide, tall, 4:3, 16:9
- RGB values: 0-255 (black to white, primary colors)
- Batch sizes: 1 to 32

✅ **Edge Cases**
- Empty directories
- Nonexistent directories
- Single image datasets
- Nested directory structures
- Mismatched dataset sizes
- Special characters in filenames
- Zero-dimension images

✅ **Batch Processing**
- Various batch sizes
- Device parameter handling
- Worker count parameter

⚠️ **Dataset Loading** (requires torch)
- Invalid dataset types
- Metadata validation
- Fallback to directory scanning
- Image resizing
- Caption generation

⚠️ **Error Handling** (requires torch)
- File I/O errors
- Missing files
- Malformed metadata
- Parameter validation

### Dependencies

**Core Dependencies (always required):**
- pytest
- Pillow (PIL)
- numpy

**Optional Dependencies (for full test coverage):**
- torch (for dataset loading tests)
- torchvision (for FID computation)
- torchmetrics (for FID computation)

## Test Design Principles

1. **Independence**: Each test is independent and can run in isolation
2. **Clarity**: Test names clearly describe what is being tested
3. **Documentation**: Comprehensive docstrings explain test purpose
4. **Isolation**: Tests use temporary directories (pytest's `tmp_path`)
5. **Robustness**: Tests handle environments where torch is not available
6. **Coverage**: Tests cover normal cases, edge cases, and error conditions

## Adding New Tests

When adding new tests:

1. **Choose the appropriate test file:**
   - `test_data_validation.py` - Data types, corruption, ranges
   - `test_error_handling.py` - Error conditions, dataset loading
   - `test_evaluation.py` - Evaluation metrics
   - `test_generation.py` - Generation utilities

2. **Follow existing patterns:**
   - Use descriptive test names starting with `test_`
   - Group related tests in classes
   - Add comprehensive docstrings
   - Use `tmp_path` fixture for file operations
   - Handle missing dependencies gracefully with `pytest.skip()`

3. **Example test structure:**
```python
def test_descriptive_name(self, tmp_path):
    """Clear description of what is being tested."""
    # Arrange: Set up test data
    test_file = tmp_path / "test.jpg"
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(test_file)
    
    # Act: Perform the operation
    result = some_function(test_file)
    
    # Assert: Verify expected behavior
    assert result == expected_value
```

## Continuous Integration

These tests are designed to run in CI/CD environments:

- Tests are isolated and don't require external resources
- Tests use temporary directories that are automatically cleaned up
- Tests handle missing optional dependencies gracefully
- Tests have reasonable timeouts for CI execution

## Known Limitations

1. **torch-dependent tests**: Many dataset loading tests require torch, which may not be available in all environments
2. **FID computation**: FID tests require torchmetrics and torchvision
3. **Permission tests**: Some file permission tests are skipped on systems where they cannot be reliably tested

## Future Improvements

Potential areas for test expansion:

- [ ] Performance benchmarks for large datasets
- [ ] Memory usage tests
- [ ] Concurrent processing tests
- [ ] Network-based dataset loading tests
- [ ] Integration tests with actual model training
- [ ] GPU-specific tests (when available)

## Troubleshooting

### Tests are being skipped
- Check if torch is installed: `python -c "import torch"`
- Install optional dependencies: `pip install torch torchvision torchmetrics`

### Tests fail with import errors
- Ensure the package is installed: `pip install -e .`
- Check Python path: `echo $PYTHONPATH`

### Tests fail with PIL errors
- Update Pillow: `pip install --upgrade Pillow`
- Check supported formats: `python -c "from PIL import Image; print(Image.OPEN)"`

## Contact

For questions or issues with the test suite, please open an issue on the GitHub repository.
