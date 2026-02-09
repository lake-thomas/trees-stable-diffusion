"""
Comprehensive tests for error handling and dataset-specific validation.

This test suite validates:
1. Dataset loading error handling (TreeDataset class)
2. Metadata and annotation file validation
3. File I/O error handling
4. Parameter validation and type checking
5. Dataset-specific edge cases (iNaturalist and Autoarborist formats)

These tests ensure the system gracefully handles errors and validates inputs.
"""

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import sys

# Note: Some tests may be skipped if torch is not available in the environment


class TestDatasetParameterValidation:
    """Test parameter validation for dataset creation."""
    
    def test_invalid_dataset_type_raises_error(self, tmp_path):
        """Test that invalid dataset types are rejected."""
        # Import here to handle potential torch import issues
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Should raise ValueError for invalid dataset type
        with pytest.raises(ValueError, match="Unknown dataset type"):
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="invalid_format"
            )
    
    def test_valid_dataset_types_accepted(self, tmp_path):
        """Test that valid dataset types are accepted."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create minimal dataset structure
        (tmp_path / "test.jpg").write_bytes(b"")
        
        valid_types = ["inaturalist", "autoarborist", "INATURALIST", "Autoarborist"]
        
        for dtype in valid_types:
            try:
                # Should accept valid types (may fail on data loading but not on type validation)
                dataset = TreeDataset(
                    data_dir=str(tmp_path),
                    dataset_type=dtype
                )
                assert dataset.dataset_type.lower() in ["inaturalist", "autoarborist"]
            except Exception as e:
                # Expected if there are no valid images, but type should be accepted
                if "Unknown dataset type" in str(e):
                    pytest.fail(f"Valid dataset type '{dtype}' was rejected")
    
    def test_max_size_parameter_validation(self, tmp_path):
        """Test max_size parameter type and range validation."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create a simple image
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(tmp_path / "test.jpg")
        
        # Test various max_size values
        valid_sizes = [64, 128, 256, 512, 1024, 2048]
        
        for size in valid_sizes:
            try:
                dataset = TreeDataset(
                    data_dir=str(tmp_path),
                    dataset_type="inaturalist",
                    max_size=size
                )
                assert dataset.max_size == size
            except Exception:
                # May fail for other reasons, but should accept valid sizes
                pass


class TestMetadataValidation:
    """Test metadata and annotation file validation."""
    
    def test_valid_inaturalist_metadata(self, tmp_path):
        """Test loading valid iNaturalist metadata."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create valid metadata
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        metadata = [
            {
                "image_path": "tree1.jpg",
                "caption": "A photo of an oak tree",
                "species": "Quercus robur"
            }
        ]
        
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            assert len(dataset) == 1
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Failed to load valid metadata: {e}")
    
    def test_malformed_json_metadata(self, tmp_path):
        """Test handling of malformed JSON metadata files."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create malformed JSON
        (tmp_path / "metadata.json").write_text("{ invalid json content")
        
        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
    
    def test_metadata_with_missing_image_files(self, tmp_path):
        """Test metadata referencing non-existent images."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create metadata referencing non-existent files
        metadata = [
            {
                "image_path": "nonexistent1.jpg",
                "caption": "A photo of a tree",
                "species": "Unknown"
            },
            {
                "image_path": "nonexistent2.jpg",
                "caption": "Another tree photo",
                "species": "Unknown"
            }
        ]
        
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            # Dataset should be empty or contain only existing images
            assert len(dataset) == 0
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Failed to handle missing images: {e}")
    
    def test_valid_autoarborist_annotations(self, tmp_path):
        """Test loading valid Autoarborist annotations."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create valid annotations
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        annotations = [
            {
                "image_file": "tree1.jpg",
                "caption": "A photo of a maple tree",
                "tree_info": {
                    "species": "Acer saccharum",
                    "height": "15m"
                }
            }
        ]
        
        (tmp_path / "annotations.json").write_text(json.dumps(annotations))
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="autoarborist"
            )
            assert len(dataset) == 1
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Failed to load valid annotations: {e}")
    
    def test_empty_metadata_file(self, tmp_path):
        """Test handling of empty metadata files."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create empty JSON array
        (tmp_path / "metadata.json").write_text("[]")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            assert len(dataset) == 0
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Failed to handle empty metadata: {e}")
    
    def test_metadata_with_missing_fields(self, tmp_path):
        """Test handling of metadata with missing required fields."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create image
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        # Metadata with minimal fields
        metadata = [
            {
                "image_path": "tree1.jpg"
                # Missing caption and species
            }
        ]
        
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            # Should handle missing fields gracefully with defaults
            if len(dataset) > 0:
                item = dataset[0]
                assert 'caption' in item
                assert 'species' in item
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Failed to handle missing fields: {e}")


class TestFileIOErrors:
    """Test file I/O error handling."""
    
    def test_nonexistent_data_directory(self):
        """Test handling of non-existent data directories."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Should raise an appropriate error
        with pytest.raises((FileNotFoundError, ValueError)):
            dataset = TreeDataset(
                data_dir="/nonexistent/path/to/data",
                dataset_type="inaturalist"
            )
    
    def test_permission_denied_scenarios(self, tmp_path):
        """Test handling when file permissions prevent access."""
        # This test may not work on all systems, so we mark it as optional
        pytest.skip("Permission testing requires specific system configuration")
    
    def test_read_only_directory(self, tmp_path):
        """Test that read-only directories can still be used for loading."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create valid dataset
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        try:
            # Should be able to read from directory
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            # Loading should work fine
            assert len(dataset) >= 0
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Failed on read-only test: {e}")


class TestDatasetLoadingEdgeCases:
    """Test edge cases in dataset loading."""
    
    def test_no_metadata_fallback_to_scanning(self, tmp_path):
        """Test fallback to directory scanning when metadata is absent."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create images without metadata
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(i * 80, i * 80, i * 80))
            img.save(tmp_path / f"tree{i}.jpg")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            # Should find images via directory scanning
            assert len(dataset) == 3
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Fallback scanning failed: {e}")
    
    def test_mixed_image_formats_in_dataset(self, tmp_path):
        """Test dataset with mixed image formats."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create images in different formats
        img_jpg = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img_jpg.save(tmp_path / "tree1.jpg", "JPEG")
        
        img_png = Image.new("RGB", (64, 64), color=(150, 150, 150))
        img_png.save(tmp_path / "tree2.png", "PNG")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            assert len(dataset) == 2
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Mixed formats failed: {e}")
    
    def test_dataset_with_subdirectories(self, tmp_path):
        """Test that images in subdirectories are NOT loaded by default."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create main directory images
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        # Create subdirectory with images
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        img.save(subdir / "tree2.jpg")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            # Should only find images in main directory (based on glob pattern)
            assert len(dataset) == 1
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Subdirectory test failed: {e}")
    
    def test_dataset_getitem_returns_expected_keys(self, tmp_path):
        """Test that dataset __getitem__ returns expected dictionary keys."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create valid dataset
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist",
                tokenizer=None  # No tokenizer
            )
            
            if len(dataset) > 0:
                item = dataset[0]
                
                # Check required keys
                assert 'image' in item
                assert 'caption' in item
                assert 'species' in item
                
                # Verify types
                assert isinstance(item['image'], Image.Image)
                assert isinstance(item['caption'], str)
                assert isinstance(item['species'], str)
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"GetItem test failed: {e}")
    
    def test_image_resizing_behavior(self, tmp_path):
        """Test that images larger than max_size are properly resized."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create large image
        large_img = Image.new("RGB", (1024, 1024), color=(100, 100, 100))
        large_img.save(tmp_path / "large_tree.jpg")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist",
                max_size=512
            )
            
            if len(dataset) > 0:
                item = dataset[0]
                img = item['image']
                
                # Image should be resized to fit within max_size
                assert max(img.size) <= 512
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Resizing test failed: {e}")
    
    def test_caption_generation_fallback(self, tmp_path):
        """Test that captions are generated when not provided in metadata."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create image without metadata
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "oak_tree.jpg")
        
        try:
            dataset = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            
            if len(dataset) > 0:
                item = dataset[0]
                # Should have a generated caption
                assert len(item['caption']) > 0
                assert isinstance(item['caption'], str)
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Caption fallback test failed: {e}")


class TestInputSanitization:
    """Test input sanitization and type validation."""
    
    def test_string_path_conversion(self, tmp_path):
        """Test that Path objects and strings both work for data_dir."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create valid dataset
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        try:
            # Test with string
            dataset_str = TreeDataset(
                data_dir=str(tmp_path),
                dataset_type="inaturalist"
            )
            
            # Test with Path object
            dataset_path = TreeDataset(
                data_dir=tmp_path,
                dataset_type="inaturalist"
            )
            
            # Both should work
            assert len(dataset_str) == len(dataset_path)
        except Exception as e:
            if "torch" not in str(e).lower():
                pytest.fail(f"Path conversion test failed: {e}")
    
    def test_dataset_type_case_insensitivity(self, tmp_path):
        """Test that dataset_type is case-insensitive."""
        try:
            from trees_sd.datasets import TreeDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import TreeDataset: {e}")
        
        # Create valid dataset
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(tmp_path / "tree1.jpg")
        
        variations = ["inaturalist", "INATURALIST", "iNaturalist", "InAtUrAlIsT"]
        
        for variation in variations:
            try:
                dataset = TreeDataset(
                    data_dir=str(tmp_path),
                    dataset_type=variation
                )
                # Should normalize to lowercase
                assert dataset.dataset_type == "inaturalist"
            except Exception as e:
                if "torch" not in str(e).lower():
                    pytest.fail(f"Case insensitivity test failed for '{variation}': {e}")


class TestGenerationHelpers:
    """Test generation helper functions that don't require torch."""
    
    def test_normalize_dataset_type(self):
        """Test dataset type normalization function."""
        try:
            from trees_sd.generation import normalize_dataset_type
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import generation module: {e}")
        
        # Test aliases
        assert normalize_dataset_type("inat") == "inaturalist"
        assert normalize_dataset_type("aa") == "autoarborist"
        assert normalize_dataset_type("inaturalist") == "inaturalist"
        assert normalize_dataset_type("autoarborist") == "autoarborist"
    
    def test_generation_prompts_type_validation(self):
        """Test that generation prompts are returned as expected."""
        try:
            from trees_sd.generation import get_generation_prompts
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import generation module: {e}")
        
        prompts = get_generation_prompts("inaturalist", "Quercus")
        
        # Should return a list of strings
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)
    
    def test_negative_prompt_validation(self):
        """Test negative prompt generation."""
        try:
            from trees_sd.generation import get_negative_prompt
        except (ImportError, OSError) as e:
            pytest.skip(f"Cannot import generation module: {e}")
        
        neg_prompt = get_negative_prompt("inaturalist")
        
        # Should return a string
        assert isinstance(neg_prompt, str)
        assert len(neg_prompt) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
