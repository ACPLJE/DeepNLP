import unittest
import os
import torch
from src.data.data_loader import DataLoaderFactory
from src.data.dataset import QADataset
from src.utils.logger import setup_logger

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger('test_data_loader')
        cls.dataset_config = {
            'name': 'squad',
            'data_dir': 'data/squad',
            'max_seq_length': 384,
            'doc_stride': 128,
            'max_query_length': 64,
            'batch_size': 8
        }

    def setUp(self):
        self.data_loader_factory = DataLoaderFactory()
        
    def test_create_data_loader(self):
        data_loader = self.data_loader_factory.create_data_loader(
            dataset_name=self.dataset_config['name'],
            data_dir=self.dataset_config['data_dir'],
            batch_size=self.dataset_config['batch_size']
        )
        self.assertIsNotNone(data_loader)
        
    def test_batch_format(self):
        data_loader = self.data_loader_factory.create_data_loader(
            dataset_name=self.dataset_config['name'],
            data_dir=self.dataset_config['data_dir'],
            batch_size=self.dataset_config['batch_size']
        )
        batch = next(iter(data_loader))
        
        required_keys = ['input_ids', 'attention_mask', 'token_type_ids', 
                        'start_positions', 'end_positions']
        for key in required_keys:
            self.assertIn(key, batch)
            self.assertTrue(torch.is_tensor(batch[key]))
            
    def test_sequence_length(self):
        data_loader = self.data_loader_factory.create_data_loader(
            dataset_name=self.dataset_config['name'],
            data_dir=self.dataset_config['data_dir'],
            batch_size=self.dataset_config['batch_size']
        )
        batch = next(iter(data_loader))
        
        max_length = self.dataset_config['max_seq_length']
        self.assertTrue(all(batch['input_ids'].size(1) <= max_length))
        self.assertTrue(all(batch['attention_mask'].size(1) <= max_length))
        
    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            self.data_loader_factory.create_data_loader(
                dataset_name='invalid_dataset',
                data_dir='data/invalid',
                batch_size=8
            )

if __name__ == '__main__':
    unittest.main()