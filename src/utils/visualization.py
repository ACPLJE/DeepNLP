# src/utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import wandb
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(config.output_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def visualize_token_embeddings(self, student_embeddings, teacher_embeddings, step):
        """Visualize token embeddings using t-SNE and PCA"""
        # Convert to numpy arrays
        student_np = student_embeddings.detach().cpu().numpy()
        teacher_np = teacher_embeddings.detach().cpu().numpy()
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        student_tsne = tsne.fit_transform(student_np)
        teacher_tsne = tsne.fit_transform(teacher_np)
        
        plt.figure(figsize=(15, 5))
        
        # t-SNE plot
        plt.subplot(121)
        plt.scatter(student_tsne[:, 0], student_tsne[:, 1], alpha=0.5, label='Student')
        plt.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], alpha=0.5, label='Teacher')
        plt.title('t-SNE Visualization of Token Embeddings')
        plt.legend()
        
        # PCA visualization
        pca = PCA(n_components=2)
        student_pca = pca.fit_transform(student_np)
        teacher_pca = pca.fit_transform(teacher_np)
        
        # PCA plot
        plt.subplot(122)
        plt.scatter(student_pca[:, 0], student_pca[:, 1], alpha=0.5, label='Student')
        plt.scatter(teacher_pca[:, 0], teacher_pca[:, 1], alpha=0.5, label='Teacher')
        plt.title('PCA Visualization of Token Embeddings')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'embeddings_step_{step}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            wandb.log({"token_embeddings": wandb.Image(plot_path)}, step=step)
            
    def visualize_attention_maps(self, attention_weights, tokens, step):
        """Visualize attention maps"""
        plt.figure(figsize=(10, 10))
        sns.heatmap(attention_weights.detach().cpu().numpy(),
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='viridis')
        plt.title('Attention Weights Heatmap')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'attention_map_step_{step}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            wandb.log({"attention_map": wandb.Image(plot_path)}, step=step)
            
    def visualize_loss_curves(self, train_losses, eval_losses, steps):
        """Plot training and evaluation loss curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_losses, label='Training Loss')
        plt.plot(steps, eval_losses, label='Evaluation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Curves')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'loss_curves.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            wandb.log({"loss_curves": wandb.Image(plot_path)})
            
    def visualize_metric_comparison(self, student_metrics, teacher_metrics, baseline_metrics):
        """Compare model performance metrics"""
        metrics = ['exact_match', 'f1']
        models = ['Student', 'Teacher', 'Baseline']
        
        values = [
            [student_metrics[m] for m in metrics],
            [teacher_metrics[m] for m in metrics],
            [baseline_metrics[m] for m in metrics]
        ]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            plt.bar(x + i*width, values[i], width, label=model)
            
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison')
        plt.xticks(x + width, metrics)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'metric_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            wandb.log({"metric_comparison": wandb.Image(plot_path)})
            
    def visualize_embedding_distances(self, student_embeddings, teacher_embeddings):
        """Visualize distances between student and teacher embeddings"""
        distances = torch.norm(student_embeddings - teacher_embeddings, dim=-1)
        
        plt.figure(figsize=(10, 5))
        plt.hist(distances.detach().cpu().numpy(), bins=50)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Student-Teacher Embedding Distances')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'embedding_distances.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            wandb.log({"embedding_distances": wandb.Image(plot_path)})