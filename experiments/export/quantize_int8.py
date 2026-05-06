import os
import sys
import numpy as np
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path, output_path, optimize=True):

    print(f"Quantizing model: {input_path}")
    print(f"Output path: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,  
        reduce_range=False 
    )
    
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100
    
    print(f"\n✓ Quantization complete!")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Size reduction: {reduction:.1f}%")
    
    metadata = {
        "source_model": str(input_path),
        "quantization_type": "dynamic",
        "weight_type": "QInt8",
        "per_channel": True,
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "reduction_percent": reduction
    }
    
    metadata_path = str(output_path).replace('.onnx', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")
    
    return output_path

def validate_quantized_model(original_path, quantized_path, n_samples=10):
    """
    Quick validation that quantized model produces similar outputs.
    """
    import onnxruntime as ort
    
    print("\nValidating quantized model...")
    
    original_session = ort.InferenceSession(str(original_path))
    quantized_session = ort.InferenceSession(str(quantized_path))
    
    input_shape = original_session.get_inputs()[0].shape
    seq_len = input_shape[1] if len(input_shape) > 1 else 512
    
    np.random.seed(42)
    errors = []
    
    for i in range(n_samples):
        input_ids = np.random.randint(0, 1000, size=(1, seq_len), dtype=np.int64)
        token_type_ids = np.zeros((1, seq_len), dtype=np.int64)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        
        original_outputs = original_session.run(None, inputs)[0]
        quantized_outputs = quantized_session.run(None, inputs)[0]
        
        mse = np.mean((original_outputs - quantized_outputs) ** 2)
        max_diff = np.max(np.abs(original_outputs - quantized_outputs))
        
        errors.append({"mse": mse, "max_diff": max_diff})
    
    avg_mse = np.mean([e["mse"] for e in errors])
    avg_max_diff = np.mean([e["max_diff"] for e in errors])
    
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average max difference: {avg_max_diff:.6f}")
    
    if avg_max_diff < 0.1: 
        print("  Validation passed - quantization maintains accuracy")
        return True
    else:
        print("  Warning: Large differences detected")
        return False

def main():
    """CLI interface for quantization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument(
        "input_model",
        help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input>_int8.onnx)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate quantized model accuracy"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip graph optimization"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input_model)
        args.output = str(input_path.parent / f"{input_path.stem}_int8.onnx")
    
    output_path = quantize_model(
        args.input_model,
        args.output,
        optimize=not args.no_optimize
    )
    
    if args.validate:
        validate_quantized_model(args.input_model, output_path)

if __name__ == "__main__":
    main()