#!/usr/bin/env python
"""
Example script for running VLM training on Kaggle.

Usage:
    1. git clone your VLM repo
    2. pip install -e VLM/
    3. Run this script or import from VLM package
"""

import os
import torch

# ============================================================================
# STEP 1: Prepare Dataset (run once)
# ============================================================================

def prepare_dataset():
    """Download and prepare LLaVA dataset."""
    from VLM.data import prepare_vqa_samples, save_vqa_dataset
    
    print("Preparing VQA dataset...")
    samples = prepare_vqa_samples(max_samples=20000)
    save_vqa_dataset(
        samples,
        output="/kaggle/working/",
        dataset="llava",
        split="train",
    )
    print("Dataset ready!")


# ============================================================================
# STEP 2: Training
# ============================================================================

def run_training():
    """Run VLM training."""
    from VLM import TrainingConfig, train
    
    config = TrainingConfig(
        # Data
        data_path="/kaggle/working/vqa_llava_train.json",
        output_dir="/kaggle/working/checkpoints",
        
        # Training
        epochs=1,
        micro_batch_size=4,
        grad_accum_steps=8,
        lr=2e-4,
        
        # Optimization
        use_amp=True,
        gradient_checkpointing=True,
        
        # Logging
        log_interval=10,
        save_interval=500,
    )
    
    print(f"Effective batch size: {config.effective_batch_size}")
    train(config)


# ============================================================================
# STEP 3: Inference
# ============================================================================

def run_inference():
    """Test inference with trained model."""
    from PIL import Image
    from VLM import VLMInference
    
    vlm = VLMInference(
        projector_checkpoint="/kaggle/input/models/foxtrot22/customvlm/pytorch/default/1/final.pt",
        device="cuda",
    )
    
    # Test on COCO image
    test_image = Image.open(
        "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/test2017/000000000001.jpg"
    ).convert("RGB")
    
    question = "What is in this image?"
    answer = vlm.generate(test_image, question, max_new_tokens=50)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")


# ============================================================================
# STEP 4: Benchmark KV-Cache
# ============================================================================

def benchmark_kv_cache():
    """Benchmark KV-cache speedup."""
    import time
    from PIL import Image
    from VLM import VLMInference
    
    vlm = VLMInference(
        projector_checkpoint="/kaggle/input/models/foxtrot22/customvlm/pytorch/default/1/final.pt",
        device="cuda",
    )
    
    test_image = Image.open(
        "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/test2017/000000000001.jpg"
    ).convert("RGB")
    question = "Describe this image in detail."
    
    # Warmup
    _ = vlm.generate_naive(test_image, question, max_new_tokens=5)
    _ = vlm.generate_with_kv_cache(test_image, question, max_new_tokens=5)
    
    print("Tokens | Naive (s) | KV (s) | Speedup")
    print("-" * 45)
    
    for max_tokens in [50, 100, 150, 200]:
        # Naive
        torch.cuda.synchronize()
        start = time.time()
        _ = vlm.generate_naive(test_image, question, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        t_naive = time.time() - start
        
        # KV-cache
        torch.cuda.synchronize()
        start = time.time()
        _ = vlm.generate_with_kv_cache(test_image, question, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        t_kv = time.time() - start
        
        speedup = t_naive / t_kv
        print(f"{max_tokens:6d} | {t_naive:9.2f} | {t_kv:6.2f} | {speedup:.2f}x")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM Training Pipeline")
    parser.add_argument("--step", choices=["prepare", "train", "inference", "benchmark", "all"], 
                        default="all", help="Which step to run")
    args = parser.parse_args()
    
    if args.step in ["prepare", "all"]:
        prepare_dataset()
    
    if args.step in ["train", "all"]:
        run_training()
    
    if args.step in ["inference", "all"]:
        run_inference()
    
    if args.step in ["benchmark", "all"]:
        benchmark_kv_cache()