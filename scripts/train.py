# scripts/train.py
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import setup_logger
from src import ModelFactory, DataLoaderFactory, DistillationTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to base config file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config file(s)')
    parser.add_argument('--dataset_config', type=str, required=True, help='Path to dataset config file')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('train')
    logger.info('Starting training process...')
    
    try:
        # Load configurations
        logger.info('Loading configurations...')
        base_config = load_config(args.config)
        dataset_config = load_config(args.dataset_config)
        model_configs = [load_config(path.strip()) for path in args.model_config.split(',')]
        
        # Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')

        # Create data loader

        logger.info('Creating data loaders...')
        data_loader_factory = DataLoaderFactory()
        data_loaders = data_loader_factory.create_dataloaders(
            dataset_name=dataset_config['name'],
            data_path=dataset_config['data_path'],
            batch_size=base_config['training']['batch_size']
        )
        train_loader = data_loaders['train']
        val_loader = data_loaders['validation']
        

        logger.info('Setting up models and trainer...')
        model_factory = ModelFactory()
        if len(model_configs) == 2:  # Distillation
            logger.info('Initializing distillation training...')
            teacher_model = model_factory.create_model('bert', model_configs[0])
            student_model = model_factory.create_model('distilbert', model_configs[1])
            
            # learning rate를 float로 변환
            learning_rate = float(base_config['training']['learning_rate'])
            
            optimizer = torch.optim.AdamW(
                student_model.parameters(), 
                lr=learning_rate  # float로 변환된 learning rate 사용
            )
            
            criterion = torch.nn.CrossEntropyLoss()
        
         
            trainer = DistillationTrainer(
                teacher_model=teacher_model.to(device),
                student_model=student_model.to(device),
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=base_config
            )
        else:  # Single model training
            logger.info('Initializing single model training...')
            model = model_factory.create_model(model_configs[0]['type'], model_configs[0])
            trainer = BaseTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,  # Added validation loader
                config=base_config,
                device=device
            )

        # Start training
        logger.info('Starting training...')
        trainer.train()
        logger.info('Training completed successfully!')
        
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    main()