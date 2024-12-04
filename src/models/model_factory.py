# src/models/model_factory.py
from transformers import AutoModel, AutoConfig
from .base_model import BaseModel
from .context_aware_distillation import ContextAwareDistillationModel

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    def create_model(self, model_type, model_config):
        """
        Create a model instance based on the specified type and configuration.
        
        Args:
            model_type (str): Type of model to create ('bert', 'distilbert', 'roberta')
            model_config (dict): Model configuration parameters
            
        Returns:
            BaseModel: Instance of the specified model
        
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type.lower() == 'bert':
            base_model = AutoModel.from_pretrained(model_config['model_name'])
            model = BaseModel(
                base_model=base_model,
                hidden_size=model_config['hidden_size'],
                num_labels=2  # start and end position
            )
            
        elif model_type.lower() == 'distilbert':
            base_model = AutoModel.from_pretrained(model_config['model_name'])
            model = BaseModel(
                base_model=base_model,
                hidden_size=model_config['hidden_size'],
                num_labels=2
            )
            
        elif model_type.lower() == 'roberta':
            base_model = AutoModel.from_pretrained(model_config['model_name'])
            model = BaseModel(
                base_model=base_model,
                hidden_size=model_config['hidden_size'],
                num_labels=2
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model

    def create_distillation_model(self, teacher_model, student_model, temperature=2.0):
        """
        Create a distillation model combining teacher and student models.
        
        Args:
            teacher_model (BaseModel): Teacher model instance
            student_model (BaseModel): Student model instance
            temperature (float): Temperature for softmax in distillation
            
        Returns:
            ContextAwareDistillation: Distillation model instance
        """
        return ContextAwareDistillationModel(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=temperature
        )