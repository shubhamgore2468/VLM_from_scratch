"""Setup script for VLM package."""

from setuptools import setup, find_packages

setup(
    name="vlm",
    version="0.1.0",
    description="Vision Language Model Training with SigLIP + Qwen2.5",
    author="Shubham Gore",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
    },
)