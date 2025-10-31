#!/usr/bin/env python3
"""
Quantize PokeeResearch-7B model using AutoAWQ

This script creates AWQ quantized weights that can be used with vLLM.
Reduces model size from ~14GB to ~4-5GB, freeing up memory for larger context windows.

Usage:
    # Use both GPUs automatically (recommended)
    python scripts/quantize_awq.py \
        --model PokeeAI/pokee_research_7b \
        --output ./quantized/pokee-research-7b-awq \
        --bits 4
    
    # Use specific GPU(s)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/quantize_awq.py \
        --model PokeeAI/pokee_research_7b \
        --output ./quantized/pokee-research-7b-awq \
        --bits 4

Requirements:
    pip install autoawq transformers datasets
"""

import argparse
import os
from pathlib import Path

try:
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: AutoAWQ not installed. Install with: pip install autoawq")
    exit(1)


def quantize_model(
    model_path: str,
    output_path: str,
    bits: int = 4,
    zero_point: bool = True,
    calib_dataset: str = "wikitext",
    device_map: str = "auto",
):
    """Quantize a model using AutoAWQ.
    
    Args:
        model_path: HuggingFace model path or local path
        output_path: Directory to save quantized model
        bits: Quantization bits (4 or 8)
        zero_point: Use zero-point quantization
        calib_dataset: Calibration dataset (wikitext, c4, etc.)
        device_map: Device mapping strategy ("auto", "cuda:0", "cuda:1", or custom dict)
    """
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if device_map == "auto" and gpu_count > 1:
            print(f"\n🚀 Using device_map='auto' - model will be split across {gpu_count} GPUs")
            print(f"   This will utilize both GPUs for faster quantization!")
    else:
        print("⚠️  No CUDA GPUs detected - quantization will run on CPU (very slow!)")
        device_map = "cpu"
    
    print(f"\nLoading model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Quantization: {bits}-bit AWQ")
    print(f"Device map: {device_map}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    print("\nLoading model (this may take a while)...")
    print("Model will be automatically split across available GPUs...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Display device placement
    if torch.cuda.is_available():
        print("\nModel device placement:")
        for name, param in list(model.named_parameters())[:5]:  # Show first 5 layers
            print(f"  {name}: {param.device}")
        print("  ... (model split across GPUs)")
    
    # Quantize model
    print(f"Quantizing model to {bits}-bit AWQ...")
    print("This will take 1-4 hours depending on hardware...")
    
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": zero_point,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM",
        },
        calib_data=calib_dataset,
    )
    
    # Save quantized model
    print(f"Saving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Quantization complete! Model saved to {output_path}")
    print(f"\nNext steps:")
    print(f"1. Update .env: MODEL={output_path}")
    print(f"2. Update .env: QUANTIZATION=awq")
    print(f"3. Restart vLLM server")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize model using AutoAWQ")
    parser.add_argument(
        "--model_path",
        "--model",
        type=str,
        dest="model_path",
        required=True,
        help="Model path (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--quantized_model_path",
        "--output",
        type=str,
        dest="quantized_model_path",
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (4 or 8)",
    )
    parser.add_argument(
        "--zero-point",
        action="store_true",
        default=True,
        help="Use zero-point quantization",
    )
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default="wikitext",
        help="Calibration dataset (wikitext, c4, etc.)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy: 'auto' (use all GPUs), 'cuda:0', 'cuda:1', or 'cpu'",
    )
    
    args = parser.parse_args()
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.quantized_model_path,
        bits=args.bits,
        zero_point=args.zero_point,
        calib_dataset=args.calib_dataset,
        device_map=args.device_map,
    )

