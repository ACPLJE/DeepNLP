# scripts/evaluate.py
import argparse
import yaml
import torch
from pathlib import Path
from src.models import ModelFactory
from src.data import DataLoaderFactory
from src.utils.logger import Logger
from src.utils.evaluation import Evaluator
from src.utils.visualization import Visualizer
from transformers import AutoTokenizer



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the distillation model')
    parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--dataset_config', type=str, required=True, help='Path to dataset config file')  # 추가
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_outputs', help='Output directory')
    parser.add_argument('--split', type=str, default='validation', choices=['validation', 'test'],
                      help='Dataset split to evaluate on')
    return parser.parse_args()

def main():
    # Parse arguments and load config
    args = parse_args()
    model_config = load_config(args.config)
    dataset_config = load_config(args.dataset_config)  # dataset config 로드
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config['output_dir'] = str(output_dir)
    
    # Initialize logger
    logger = Logger("evaluation")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    # Create data loader
    data_loader_factory = DataLoaderFactory()  # 인스턴스 생성
    dataloaders = data_loader_factory.create_dataloaders(
        dataset_name=dataset_config['name'],
        data_path=dataset_config['data_path'],
        batch_size=32,  # 또는 config에서 가져올 수 있음
        tokenizer_name=model_config['model_name']
    )
    eval_dataloader = dataloaders['validation']
    
    # Create model
    model_type = Path(args.config).stem.split('_')[0]  # 'distilbert_config.yaml' -> 'distilbert'
    model = ModelFactory.create_model(
        model_type=model_type,
        model_config=model_config
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Initialize evaluator and visualizer
    evaluator = Evaluator(model, tokenizer, device)
    visualizer = Visualizer(config)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    try:
        results = evaluator.evaluate(eval_dataloader)
        
        # Log results
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.2f}")
        
        # Generate visualizations
        if config.get('generate_visualizations', False):
            logger.info("Generating visualizations...")
            with torch.no_grad():
                batch = next(iter(eval_dataloader))
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = model(**batch)
                
                # Visualize embeddings
                visualizer.visualize_token_embeddings(
                    outputs['student_hidden'],
                    outputs['teacher_hidden'],
                    step='final'
                )
                
                # Visualize attention maps
                if 'attention_weights' in outputs:
                    visualizer.visualize_attention_maps(
                        outputs['attention_weights'][0],
                        tokenizer.convert_ids_to_tokens(batch['input_ids'][0]),
                        step='final'
                    )
        
        # Save results
        results_path = output_dir / 'evaluation_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    
    logger.info("Evaluation completed")

if __name__ == '__main__':
    main()