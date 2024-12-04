# src/data/data_loader.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .dataset import QADataset

class DataLoaderFactory:
    """Factory class for creating data loaders"""
    
    def __init__(self):
        self.tokenizer = None
    
    def create_dataloaders(self, dataset_name, data_path, batch_size, 
                          tokenizer_name="bert-base-uncased", num_workers=4):
        """
        Create train and validation data loaders.
        
        Args:
            dataset_name (str): Name of the dataset (unused, kept for compatibility)
            data_path (str): Path to the data directory
            batch_size (int): Batch size for the data loaders
            tokenizer_name (str): Name of the tokenizer to use
            num_workers (int): Number of workers for data loading
            
        Returns:
            dict: Dictionary containing train and validation data loaders
        """
        # Initialize tokenizer if not already initialized
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Create dataset instances without dataset_name parameter
        train_dataset = QADataset(
            data_path=data_path,
            split='train',
            tokenizer=self.tokenizer
        )
        
        val_dataset = QADataset(
            data_path=data_path,
            split='validation',
            tokenizer=self.tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn
        )
        
        return {
            'train': train_loader,
            'validation': val_loader
        }
        
def create_dataloaders(self, dataset_name, data_path, batch_size, 
                      tokenizer_name="bert-base-uncased", num_workers=4):
    """
    Create train and validation data loaders.
    
    Args:
        dataset_name (str): Name of the dataset (unused, kept for compatibility)
        data_path (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        tokenizer_name (str): Name of the tokenizer to use
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Dictionary containing train and validation data loaders
    """
    # Initialize tokenizer if not already initialized
    if self.tokenizer is None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    # Create dataset instances
    train_dataset = QADataset(
        data_path=data_path,
        split='train',
        tokenizer=self.tokenizer
    )
    
    val_dataset = QADataset(
        data_path=data_path,
        split='validation',
        tokenizer=self.tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    return {
        'train': train_loader,
        'validation': val_loader
    }