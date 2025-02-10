import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from transformers import AutoModelForCausalLM 


def save_weights(model, filename):
    """Save model weights in .pt format."""
    torch.save(model.state_dict(), filename)
    logging.info(f"Model weights saved to {filename}")

def extract_exp_mantissa(tensors):
    """Extract mantissa and exponent from FP8 tensors."""
    if tensors.dtype != torch.float8_e4m3fn:
        raise ValueError("Input tensor must be of type torch.float8_e4m3fn")
    int_repr = tensors.view(torch.int8).flatten()
    mantissa_mask = 0b00000111  # 3 bits mask for mantissa
    exp_mask = 0b00001111       # 4 bits mask for exponent
    mantissas = int_repr & mantissa_mask
    exponents = (int_repr >> 3) & exp_mask
    return mantissas, exponents

def max_convolution(tensors, window_size):
    """Perform max convolution on tensors."""
    mantissas, exponents = extract_exp_mantissa(tensors)
    mantissas = mantissas.view(1, 1, *tensors.shape).to(torch.float32)
    max_mantissas, max_indices = torch.nn.functional.max_pool2d(mantissas, kernel_size=window_size, stride=1, return_indices=True)
    max_exponents = torch.zeros_like(max_mantissas, dtype=torch.int8, device=max_mantissas.device)
    for i in range(max_indices.shape[2]):
        for j in range(max_indices.shape[3]):
            max_idx = max_indices[0, 0, i, j].item()
            row = max_idx // exponents.shape[-1]
            col = max_idx % exponents.shape[-1]
            max_exponents[0, 0, i, j] = exponents[0, 0, row, col].item()
    return max_mantissas.squeeze(0).squeeze(0).to(torch.int8), max_exponents.squeeze(0).squeeze(0)


def save_quantized_model(model, output_dir, model_name, precision):
    """Save the quantized model to disk."""
    model_path = os.path.join(output_dir, f"{model_name}_{precision}_quantized")
    model.save_pretrained(model_path)
    logging.info(f"Quantized model saved to {model_path}")


def load_quantized_model(output_dir, model_name, precision):
    """Load a quantized model from disk if it exists."""
    model_path = os.path.join(output_dir, f"{model_name}_{precision}_quantized")
    if os.path.exists(model_path):
        logging.info(f"Loading quantized model from {model_path}")
        return AutoModelForCausalLM.from_pretrained(model_path)
    return None

