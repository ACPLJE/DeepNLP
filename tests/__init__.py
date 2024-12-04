import os
import sys
import unittest
import torch
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

class TestBase(unittest.TestCase):
    """Base class for all test cases in the project."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment before all tests."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Common test configurations
        cls.test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.base_config = {
            'max_seq_length': 384,
            'doc_stride': 128,
            'max_query_length': 64,
            'batch_size': 8,
            'learning_rate': 3e-5,
            'num_epochs': 3
        }
        
    def setUp(self):
        """Set up test environment before each test."""
        pass
        
    @staticmethod
    def get_test_data_path(dataset_name):
        """Get path to test data directory."""
        return os.path.join(project_root, 'tests', 'test_data', dataset_name)
    
    @staticmethod
    def create_dummy_batch(batch_size=4, seq_length=128):
        """Create a dummy batch for testing."""
        return {
            'input_ids': torch.randint(0, 30522, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length),
            'token_type_ids': torch.zeros(batch_size, seq_length),
            'start_positions': torch.randint(0, seq_length-1, (batch_size,)),
            'end_positions': torch.randint(0, seq_length-1, (batch_size,))
        }
    
    @staticmethod
    def get_model_config(model_type):
        """Get model configuration for testing."""
        configs = {
            'bert': {
                'model_name': 'bert-base-uncased',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12
            },
            'distilbert': {
                'model_name': 'distilbert-base-uncased',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 6
            },
            'roberta': {
                'model_name': 'roberta-base',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12
            }
        }
        return configs.get(model_type, None)
    
    def assertTensorEqual(self, tensor1, tensor2, msg=None):
        """Assert that two tensors are equal."""
        self.assertTrue(torch.all(torch.eq(tensor1, tensor2)), msg)
    
    def assertTensorShape(self, tensor, expected_shape, msg=None):
        """Assert that a tensor has the expected shape."""
        self.assertEqual(tuple(tensor.shape), tuple(expected_shape), msg)
    
    def assertModelOutput(self, output, batch_size, seq_length):
        """Assert that model output has the correct format and shapes."""
        self.assertIn('start_logits', output)
        self.assertIn('end_logits', output)
        self.assertTensorShape(output['start_logits'], (batch_size, seq_length))
        self.assertTensorShape(output['end_logits'], (batch_size, seq_length))