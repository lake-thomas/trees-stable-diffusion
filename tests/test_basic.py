"""
Basic tests for Trees SD package structure
"""

import pytest
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_package_import():
    """Test that the package can be imported"""
    import trees_sd
    assert trees_sd.__version__ == "0.1.0"


def test_dataset_module_import():
    """Test that dataset module can be imported"""
    from trees_sd.datasets import TreeDataset, create_dataset
    assert TreeDataset is not None
    assert create_dataset is not None


def test_training_module_import():
    """Test that training module can be imported"""
    from trees_sd.training import LoRATrainer, train_model
    assert LoRATrainer is not None
    assert train_model is not None


def test_cli_module_import():
    """Test that CLI module can be imported"""
    from trees_sd import cli
    assert cli.main is not None


def test_dataset_type_validation():
    """Test dataset type validation"""
    from trees_sd.datasets import TreeDataset
    
    # Should accept valid dataset types
    # Note: These will fail at runtime without actual data, 
    # but should pass the initialization
    try:
        dataset = TreeDataset(
            data_dir="/nonexistent",
            dataset_type="inaturalist"
        )
    except FileNotFoundError:
        pass  # Expected since directory doesn't exist
    
    try:
        dataset = TreeDataset(
            data_dir="/nonexistent",
            dataset_type="autoarborist"
        )
    except FileNotFoundError:
        pass  # Expected since directory doesn't exist
    
    # Should reject invalid dataset type
    with pytest.raises(ValueError):
        dataset = TreeDataset(
            data_dir="/nonexistent",
            dataset_type="invalid_type"
        )


def test_model_version_validation():
    """Test model version validation in trainer"""
    from trees_sd.training import LoRATrainer
    
    # Should accept valid model versions
    trainer_15 = LoRATrainer(model_version="sd1.5")
    assert trainer_15.model_version == "sd1.5"
    
    trainer_35 = LoRATrainer(model_version="sd3.5")
    assert trainer_35.model_version == "sd3.5"
    
    trainer_refiner = LoRATrainer(model_version="sdxl-refiner")
    assert trainer_refiner.model_version == "sdxl-refiner"
    
    # Should reject invalid model version
    with pytest.raises(ValueError):
        trainer = LoRATrainer(model_version="invalid_version")


def test_config_files_exist():
    """Test that example config files exist"""
    config_dir = Path(__file__).parent.parent / "trees_sd" / "configs"
    
    assert (config_dir / "sd15_inaturalist.yaml").exists()
    assert (config_dir / "sd35_inaturalist.yaml").exists()
    assert (config_dir / "sd15_autoarborist.yaml").exists()
    assert (config_dir / "sdxl_refiner_inaturalist.yaml").exists()


def test_example_scripts_exist():
    """Test that example scripts exist"""
    examples_dir = Path(__file__).parent.parent / "examples"
    
    assert (examples_dir / "train_sd15_inaturalist.py").exists()
    assert (examples_dir / "train_sd35_autoarborist.py").exists()
    assert (examples_dir / "compare_models.py").exists()
    assert (examples_dir / "train_sdxl_refiner.py").exists()


def test_unified_script_layout():
    """Test that generation moved into package and legacy folder removed"""
    repo_root = Path(__file__).parent.parent

    assert (repo_root / "download_models.py").exists()
    assert (repo_root / "make_genus_splits.py").exists()
    assert (repo_root / "trees_sd" / "generation" / "generator.py").exists()
    assert (repo_root / "trees_sd" / "generation" / "prompts.py").exists()
    assert (repo_root / "trees_sd" / "generate_cli.py").exists()
    assert not (repo_root / "sd-code").exists()
    assert not (repo_root / "sd1.5-code").exists()
    assert not (repo_root / "sd3.5-code").exists()


def test_sdxl_refiner_default_model():
    """Test SDXL refiner default model path"""
    from trees_sd.training import LoRATrainer
    
    trainer = LoRATrainer(model_version="sdxl-refiner")
    assert trainer.pretrained_model_name_or_path == "stabilityai/stable-diffusion-xl-refiner-1.0"


def test_sdxl_refiner_parameters():
    """Test SDXL refiner-specific parameters"""
    from trees_sd.training import LoRATrainer
    
    # Test default refiner parameters
    trainer = LoRATrainer(
        model_version="sdxl-refiner",
        refiner_strength=0.5,
        use_refiner=True
    )
    assert trainer.refiner_strength == 0.5
    assert trainer.use_refiner is True


def test_dataset_dual_tokenizer_support():
    """Test that dataset supports dual tokenizers for SDXL models"""
    from trees_sd.datasets import TreeDataset
    
    # Should support second tokenizer
    try:
        dataset = TreeDataset(
            data_dir="/nonexistent",
            dataset_type="inaturalist",
            tokenizer=None,
            tokenizer_2=None
        )
    except FileNotFoundError:
        pass  # Expected since directory doesn't exist


def test_generator_sdxl_refiner_support():
    """Test that generator supports SDXL refiner"""
    from trees_sd.generation.generator import ImageGenerator
    
    generator = ImageGenerator(model_version="sdxl-refiner")
    assert generator.model_version == "sdxl-refiner"
    assert generator.base_model_id == "stabilityai/stable-diffusion-xl-refiner-1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
