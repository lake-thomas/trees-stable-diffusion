"""
Dataset loaders for tree images from iNaturalist and Autoarborist
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import json


class TreeDataset(Dataset):
    """Base dataset class for tree images"""
    
    def __init__(
        self,
        data_dir: str,
        dataset_type: str = "inaturalist",
        caption_column: str = "text",
        image_column: str = "image",
        max_size: int = 512,
        tokenizer=None,
    ):
        """
        Args:
            data_dir: Directory containing the dataset
            dataset_type: Either 'inaturalist' or 'autoarborist'
            caption_column: Column name for captions
            image_column: Column name for images
            max_size: Maximum image size
            tokenizer: Tokenizer for text encoding
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type.lower()
        self.caption_column = caption_column
        self.image_column = image_column
        self.max_size = max_size
        self.tokenizer = tokenizer
        
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset based on type"""
        if self.dataset_type == "inaturalist":
            return self._load_inaturalist()
        elif self.dataset_type == "autoarborist":
            return self._load_autoarborist()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_inaturalist(self) -> List[Dict[str, Any]]:
        """Load iNaturalist formatted data"""
        data = []
        
        # Look for metadata file
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            for item in metadata:
                image_path = self.data_dir / item.get('image_path', item.get('file_name', ''))
                if image_path.exists():
                    caption = item.get('caption', item.get('description', f"A photo of a {item.get('species', 'tree')}"))
                    data.append({
                        'image_path': str(image_path),
                        'caption': caption,
                        'species': item.get('species', 'unknown'),
                    })
        else:
            # Fallback: scan directory for images
            for img_file in self.data_dir.glob("*.jpg"):
                data.append({
                    'image_path': str(img_file),
                    'caption': f"A photo of a tree",
                    'species': img_file.stem,
                })
                
        return data
    
    def _load_autoarborist(self) -> List[Dict[str, Any]]:
        """Load Autoarborist formatted data"""
        data = []
        
        # Look for annotations file
        annotations_file = self.data_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
                
            for item in annotations:
                image_path = self.data_dir / item.get('image_file', item.get('filename', ''))
                if image_path.exists():
                    tree_info = item.get('tree_info', {})
                    species = tree_info.get('species', 'tree')
                    caption = item.get('caption', f"A photo of a {species}")
                    data.append({
                        'image_path': str(image_path),
                        'caption': caption,
                        'species': species,
                        'tree_info': tree_info,
                    })
        else:
            # Fallback: scan directory for images
            for img_file in self.data_dir.glob("*.jpg"):
                data.append({
                    'image_path': str(img_file),
                    'caption': f"A photo of a tree",
                    'species': img_file.stem,
                })
                
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Resize if needed
        if max(image.size) > self.max_size:
            image.thumbnail((self.max_size, self.max_size), Image.LANCZOS)
        
        result = {
            'image': image,
            'caption': item['caption'],
            'species': item.get('species', 'unknown'),
        }
        
        # Tokenize caption if tokenizer provided
        if self.tokenizer is not None:
            result['input_ids'] = self.tokenizer(
                item['caption'],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).input_ids[0]
        
        return result


def create_dataset(
    data_dir: str,
    dataset_type: str = "inaturalist",
    max_size: int = 512,
    tokenizer=None,
) -> TreeDataset:
    """
    Factory function to create a tree dataset
    
    Args:
        data_dir: Directory containing the dataset
        dataset_type: Either 'inaturalist' or 'autoarborist'
        max_size: Maximum image size
        tokenizer: Tokenizer for text encoding
        
    Returns:
        TreeDataset instance
    """
    return TreeDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        max_size=max_size,
        tokenizer=tokenizer,
    )
