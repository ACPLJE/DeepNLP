# setup.py
from setuptools import setup, find_packages

setup(
    name="context_aware_distillation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "datasets>=1.6.0",
        "numpy>=1.19.0",
        "tqdm>=4.61.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "wandb>=0.12.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.7",
)