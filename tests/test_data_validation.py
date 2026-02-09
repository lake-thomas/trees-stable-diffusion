"""
Comprehensive tests for data validation: data types, corrupted images, and variable ranges.

This test suite validates:
1. Image data type handling (various formats: JPEG, PNG, etc.)
2. Corrupted/invalid image handling
3. Variable ranges (dimensions, RGB values, batch sizes, etc.)
4. Edge cases (empty datasets, missing files, malformed metadata)

These tests ensure robustness when processing tree image datasets from various sources.
"""

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from trees_sd.evaluation.metrics import (
    _collect_image_paths,
    _compute_basic_image_stats,
    evaluate_image_quality,
    VALID_EXTENSIONS,
)


class TestImageDataTypes:
    """Test handling of various image data types and formats."""
    
    def test_supported_image_formats(self, tmp_path):
        """Test that all supported image formats are properly handled."""
        formats = [
            ("test.jpg", "JPEG"),
            ("test.jpeg", "JPEG"),
            ("test.png", "PNG"),
            ("test.bmp", "BMP"),
            ("test.webp", "WebP"),
        ]
        
        for filename, format_type in formats:
            img_path = tmp_path / filename
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(img_path, format=format_type)
            
            # Verify file is created and can be loaded
            assert img_path.exists()
            loaded_img = Image.open(img_path)
            assert loaded_img.mode == "RGB"
            assert loaded_img.size == (64, 64)
    
    def test_collect_image_paths_filters_by_extension(self, tmp_path):
        """Test that only valid image extensions are collected."""
        # Create files with various extensions
        valid_files = ["img1.jpg", "img2.png", "img3.jpeg"]
        invalid_files = ["file.txt", "data.json", "script.py", "readme.md"]
        
        for filename in valid_files + invalid_files:
            file_path = tmp_path / filename
            if filename in valid_files:
                img = Image.new("RGB", (32, 32), color=(0, 0, 0))
                img.save(file_path)
            else:
                file_path.write_text("not an image")
        
        # Collect image paths
        paths = _collect_image_paths(str(tmp_path))
        
        # Verify only valid image files are collected
        assert len(paths) == len(valid_files)
        for path in paths:
            assert path.suffix.lower() in VALID_EXTENSIONS
    
    def test_case_insensitive_extension_matching(self, tmp_path):
        """Test that image extension matching is case-insensitive."""
        extensions = [".JPG", ".Jpg", ".jpg", ".PNG", ".png", ".JPEG"]
        
        for i, ext in enumerate(extensions):
            img_path = tmp_path / f"test{i}{ext}"
            img = Image.new("RGB", (32, 32), color=(i * 40, i * 40, i * 40))
            img.save(img_path, format="JPEG" if "jpg" in ext.lower() or "jpeg" in ext.lower() else "PNG")
        
        paths = _collect_image_paths(str(tmp_path))
        assert len(paths) == len(extensions)
    
    def test_rgba_to_rgb_conversion(self, tmp_path):
        """Test that RGBA images are properly converted to RGB."""
        img_path = tmp_path / "rgba_image.png"
        
        # Create RGBA image with transparency
        img = Image.new("RGBA", (64, 64), color=(100, 150, 200, 255))
        img.save(img_path, "PNG")
        
        # Load and verify conversion happens in evaluation
        paths = [img_path]
        stats = _compute_basic_image_stats(paths, paths)
        
        # Should successfully process RGBA image
        assert stats["real_count"] == 1.0
        assert stats["generated_count"] == 1.0


class TestCorruptedImages:
    """Test handling of corrupted and invalid image files."""
    
    def test_empty_file_raises_error(self, tmp_path):
        """Test that empty files are handled gracefully."""
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_bytes(b"")
        
        # Should raise an error when trying to process
        with pytest.raises(Exception):  # PIL raises various exceptions
            Image.open(empty_file).convert("RGB")
    
    def test_truncated_image_file(self, tmp_path):
        """Test handling of truncated/incomplete image files."""
        img_path = tmp_path / "truncated.jpg"
        
        # Create a valid image first
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(img_path, "JPEG")
        
        # Truncate the file
        with open(img_path, "rb") as f:
            data = f.read()
        
        truncated_data = data[:len(data) // 2]  # Keep only half
        img_path.write_bytes(truncated_data)
        
        # Should raise an error or warning
        with pytest.raises(Exception):
            img = Image.open(img_path)
            img.load()  # Force loading
    
    def test_invalid_image_data(self, tmp_path):
        """Test handling of files with invalid image data."""
        bad_file = tmp_path / "invalid.jpg"
        bad_file.write_text("This is not an image file!")
        
        # Should raise an error
        with pytest.raises(Exception):
            Image.open(bad_file).convert("RGB")
    
    def test_non_image_file_with_image_extension(self, tmp_path):
        """Test handling of non-image files with image extensions."""
        fake_img = tmp_path / "fake.png"
        fake_img.write_text("Just some text pretending to be an image")
        
        with pytest.raises(Exception):
            Image.open(fake_img).convert("RGB")


class TestVariableRanges:
    """Test validation of variable ranges (dimensions, RGB values, sizes, etc.)."""
    
    def test_very_small_images(self, tmp_path):
        """Test handling of very small image dimensions."""
        small_sizes = [(1, 1), (2, 2), (4, 4), (8, 8)]
        
        for i, size in enumerate(small_sizes):
            img_path = tmp_path / f"small_{i}.jpg"
            img = Image.new("RGB", size, color=(50, 100, 150))
            img.save(img_path, "JPEG")
            
            # Verify can be loaded
            loaded = Image.open(img_path)
            assert loaded.size == size
    
    def test_very_large_images(self, tmp_path):
        """Test handling of large image dimensions."""
        # Test moderately large images (very large would be memory intensive)
        large_sizes = [(1024, 1024), (2048, 1024), (1024, 2048)]
        
        for i, size in enumerate(large_sizes):
            img_path = tmp_path / f"large_{i}.png"
            img = Image.new("RGB", size, color=(100, 100, 100))
            img.save(img_path, "PNG")
            
            loaded = Image.open(img_path)
            assert loaded.size == size
    
    def test_non_square_aspect_ratios(self, tmp_path):
        """Test handling of various aspect ratios."""
        aspect_ratios = [
            (100, 50),   # 2:1
            (50, 100),   # 1:2
            (300, 100),  # 3:1
            (100, 300),  # 1:3
            (640, 480),  # 4:3
            (1920, 1080), # 16:9
        ]
        
        for i, size in enumerate(aspect_ratios):
            img_path = tmp_path / f"aspect_{i}.jpg"
            img = Image.new("RGB", size, color=(i * 30, i * 30, i * 30))
            img.save(img_path, "JPEG")
            
            loaded = Image.open(img_path)
            assert loaded.size == size
    
    def test_rgb_value_extremes(self, tmp_path):
        """Test handling of extreme RGB values."""
        test_colors = [
            (0, 0, 0),       # Pure black
            (255, 255, 255), # Pure white
            (255, 0, 0),     # Pure red
            (0, 255, 0),     # Pure green
            (0, 0, 255),     # Pure blue
            (128, 128, 128), # Mid gray
        ]
        
        for i, color in enumerate(test_colors):
            img_path = tmp_path / f"color_{i}.png"
            img = Image.new("RGB", (32, 32), color=color)
            img.save(img_path, "PNG")
            
            # Verify color is preserved
            loaded = Image.open(img_path)
            arr = np.array(loaded)
            mean_color = arr.mean(axis=(0, 1))
            
            # Allow some tolerance for lossy compression
            for j, expected in enumerate(color):
                assert abs(mean_color[j] - expected) < 5
    
    def test_basic_stats_with_varied_dimensions(self, tmp_path):
        """Test basic statistics computation with varied image dimensions."""
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Create images with different dimensions
        sizes = [(64, 64), (128, 128), (256, 256), (128, 256)]
        
        for i, size in enumerate(sizes):
            real_img = Image.new("RGB", size, color=(50 + i * 20, 100, 150))
            gen_img = Image.new("RGB", size, color=(60 + i * 20, 110, 160))
            
            real_img.save(real_dir / f"real_{i}.png")
            gen_img.save(gen_dir / f"gen_{i}.png")
        
        # Compute stats
        metrics = evaluate_image_quality(
            real_dir=str(real_dir),
            generated_dir=str(gen_dir),
            metrics=["basic"],
        )
        
        # Verify counts
        assert metrics["real_count"] == len(sizes)
        assert metrics["generated_count"] == len(sizes)
        
        # Verify average dimensions are computed
        assert metrics["real_avg_width"] > 0
        assert metrics["real_avg_height"] > 0
        assert metrics["generated_avg_width"] > 0
        assert metrics["generated_avg_height"] > 0


class TestEdgeCases:
    """Test edge cases: empty datasets, missing files, malformed metadata."""
    
    def test_empty_directory_raises_error(self, tmp_path):
        """Test that empty directories are properly detected."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No supported image files found"):
            _collect_image_paths(str(empty_dir))
    
    def test_nonexistent_directory_raises_error(self, tmp_path):
        """Test that nonexistent directories raise appropriate errors."""
        fake_dir = tmp_path / "does_not_exist"
        
        with pytest.raises(ValueError, match="does not exist"):
            _collect_image_paths(str(fake_dir))
    
    def test_directory_with_only_unsupported_files(self, tmp_path):
        """Test directory containing only unsupported file types."""
        test_dir = tmp_path / "unsupported"
        test_dir.mkdir()
        
        # Create non-image files
        (test_dir / "file.txt").write_text("text")
        (test_dir / "data.json").write_text("{}")
        (test_dir / "script.py").write_text("print('hello')")
        
        with pytest.raises(ValueError, match="No supported image files found"):
            _collect_image_paths(str(test_dir))
    
    def test_single_image_in_directory(self, tmp_path):
        """Test handling of directory with single image."""
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(single_dir / "only_one.jpg", "JPEG")
        
        paths = _collect_image_paths(str(single_dir))
        assert len(paths) == 1
    
    def test_nested_directory_structure(self, tmp_path):
        """Test that images in nested directories are found."""
        root_dir = tmp_path / "root"
        root_dir.mkdir()
        
        # Create nested structure
        (root_dir / "level1").mkdir()
        (root_dir / "level1" / "level2").mkdir()
        
        # Add images at different levels
        for level in ["", "level1", "level1/level2"]:
            dir_path = root_dir / level if level else root_dir
            img = Image.new("RGB", (32, 32), color=(50, 50, 50))
            img.save(dir_path / f"img_{level.replace('/', '_')}.jpg", "JPEG")
        
        paths = _collect_image_paths(str(root_dir))
        assert len(paths) == 3
    
    def test_mismatched_real_and_generated_counts(self, tmp_path):
        """Test handling when real and generated sets have different sizes."""
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Create different numbers of images
        for i in range(5):
            img = Image.new("RGB", (32, 32), color=(i * 50, i * 50, i * 50))
            img.save(real_dir / f"real_{i}.jpg", "JPEG")
        
        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(i * 80, i * 80, i * 80))
            img.save(gen_dir / f"gen_{i}.jpg", "JPEG")
        
        # Should still compute statistics
        metrics = evaluate_image_quality(
            real_dir=str(real_dir),
            generated_dir=str(gen_dir),
            metrics=["basic"],
        )
        
        assert metrics["real_count"] == 5.0
        assert metrics["generated_count"] == 3.0
    
    def test_special_characters_in_filenames(self, tmp_path):
        """Test handling of filenames with special characters."""
        special_names = [
            "image with spaces.jpg",
            "image-with-dashes.png",
            "image_with_underscores.jpg",
            "image.multiple.dots.png",
        ]
        
        for name in special_names:
            img_path = tmp_path / name
            img = Image.new("RGB", (32, 32), color=(100, 100, 100))
            img.save(img_path)
        
        paths = _collect_image_paths(str(tmp_path))
        assert len(paths) == len(special_names)
    
    def test_zero_dimensions_image(self, tmp_path):
        """Test that images with zero dimensions can be created but are unusual."""
        # PIL actually allows creating 0-dimension images in newer versions
        # This is an edge case that may cause issues in processing
        img_zero = Image.new("RGB", (0, 0))
        assert img_zero.size == (0, 0)
        
        # Test that such images have zero pixels
        arr = np.array(img_zero)
        assert arr.size == 0
        
        # These would be problematic in real use cases but PIL allows them
        img_partial_zero = Image.new("RGB", (64, 0))
        assert img_partial_zero.size == (64, 0)
    
    def test_rgb_mean_computation_correctness(self, tmp_path):
        """Test that RGB mean values are computed correctly."""
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Create image with known color
        known_color = (100, 150, 200)
        img = Image.new("RGB", (16, 16), color=known_color)
        img.save(real_dir / "test.png")
        img.save(gen_dir / "test.png")
        
        metrics = evaluate_image_quality(
            real_dir=str(real_dir),
            generated_dir=str(gen_dir),
            metrics=["basic"],
        )
        
        # Verify mean RGB values (normalized to 0-1)
        expected_r = known_color[0] / 255.0
        expected_g = known_color[1] / 255.0
        expected_b = known_color[2] / 255.0
        
        assert abs(metrics["real_mean_r"] - expected_r) < 0.01
        assert abs(metrics["real_mean_g"] - expected_g) < 0.01
        assert abs(metrics["real_mean_b"] - expected_b) < 0.01


class TestBatchProcessing:
    """Test batch processing and parameter validation."""
    
    def test_various_batch_sizes(self, tmp_path):
        """Test evaluation with various batch sizes."""
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Create 10 test images
        for i in range(10):
            img = Image.new("RGB", (32, 32), color=(i * 25, i * 25, i * 25))
            img.save(real_dir / f"real_{i}.jpg")
            img.save(gen_dir / f"gen_{i}.jpg")
        
        batch_sizes = [1, 2, 5, 10, 32]
        
        for batch_size in batch_sizes:
            # Should work with any reasonable batch size
            metrics = evaluate_image_quality(
                real_dir=str(real_dir),
                generated_dir=str(gen_dir),
                metrics=["basic"],
                batch_size=batch_size,
            )
            
            assert metrics["real_count"] == 10.0
            assert metrics["generated_count"] == 10.0
    
    def test_device_parameter_validation(self, tmp_path):
        """Test that device parameter is accepted."""
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(real_dir / "test.jpg")
        img.save(gen_dir / "test.jpg")
        
        # Should accept device parameter (even if not used for basic metrics)
        metrics = evaluate_image_quality(
            real_dir=str(real_dir),
            generated_dir=str(gen_dir),
            metrics=["basic"],
            device="cpu",
        )
        
        assert metrics["real_count"] == 1.0
    
    def test_num_workers_parameter(self, tmp_path):
        """Test that num_workers parameter is accepted."""
        real_dir = tmp_path / "real"
        gen_dir = tmp_path / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(real_dir / "test.jpg")
        img.save(gen_dir / "test.jpg")
        
        # Should accept num_workers parameter
        metrics = evaluate_image_quality(
            real_dir=str(real_dir),
            generated_dir=str(gen_dir),
            metrics=["basic"],
            num_workers=0,
        )
        
        assert metrics["real_count"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
