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
    
    # Should reject invalid model version
    with pytest.raises(ValueError):
        trainer = LoRATrainer(model_version="invalid_version")


def test_config_files_exist():
    """Test that example config files exist"""
    config_dir = Path(__file__).parent.parent / "trees_sd" / "configs"
    
    assert (config_dir / "sd15_inaturalist.yaml").exists()
    assert (config_dir / "sd35_inaturalist.yaml").exists()
    assert (config_dir / "sd15_autoarborist.yaml").exists()


def test_example_scripts_exist():
    """Test that example scripts exist"""
    examples_dir = Path(__file__).parent.parent / "examples"
    
    assert (examples_dir / "train_sd15_inaturalist.py").exists()
    assert (examples_dir / "train_sd35_autoarborist.py").exists()
    assert (examples_dir / "compare_models.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
