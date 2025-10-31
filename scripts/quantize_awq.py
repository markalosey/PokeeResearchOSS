#!/usr/bin/env python3
"""
Quantize PokeeResearch-7B model using AutoAWQ

This script creates AWQ quantized weights that can be used with vLLM.
Reduces model size from ~14GB to ~4-5GB, freeing up memory for larger context windows.

Usage:
    python scripts/quantize_awq.py \
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
):
    """Quantize a model using AutoAWQ.
    
    Args:
        model_path: HuggingFace model path or local path
        output_path: Directory to save quantized model
        bits: Quantization bits (4 or 8)
        zero_point: Use zero-point quantization
        calib_dataset: Calibration dataset (wikitext, c4, etc.)
    """
    print(f"Loading model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Quantization: {bits}-bit AWQ")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    print("Loading model (this may take a while)...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    
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
        "--model",
        type=str,
        required=True,
        help="Model path (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--output",
        type=str,
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
    
    args = parser.parse_args()
    
    quantize_model(
        model_path=args.model,
        output_path=args.output,
        bits=args.bits,
        zero_point=args.zero_point,
        calib_dataset=args.calib_dataset,
    )

