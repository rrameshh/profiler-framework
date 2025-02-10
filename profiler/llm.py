import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from profiler.utils import save_weights, extract_exp_mantissa, max_convolution, load_quantized_model, save_quantized_model

def get_quantization_config(precision, quant_scheme):
    """Get quantization configuration based on precision and scheme."""
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        return BitsAndBytesConfig(load_in_4bit=True)
    elif precision == "fp8" and quant_scheme == "awq":
        pass
    elif precision == "fp8" and quant_scheme == "gptq":
        pass
    return None

def profile_llm(model_name, precision, quant_scheme, tile_sizes, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Loading model: {model_name} with precision {precision}")
    # Load quantized model if available
    quantized_model = load_quantized_model(output_dir, model_name, precision)
    if quantized_model:
        model = quantized_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = get_quantization_config(precision, quant_scheme)
        if quantization_config:
       
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
            save_quantized_model(model, output_dir, model_name, precision)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

    # model.to(device)
    model_dict = model.state_dict()
    layers = {}
    for name, param in model_dict.items():
        if any(name.endswith(proj) for proj in [
            'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
            'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'
        ]):
            layers[name] = param

    save_weights(model, os.path.join(output_dir, f"{model_name}_weights.pt"))

    for name, param in layers.items():
        logging.info(f"Processing layer: {name}")
        param = param.to(device)

        if precision.lower() == 'fp8' or precision.lower() == 'fp16':
        
            mantissas, exps = extract_exp_mantissa(param)
            if tile_sizes:
                max_mantissas, max_exps = max_convolution(param, window_size=(tile_sizes, tile_sizes))
                np.savetxt(os.path.join(output_dir, f"{name.replace('.', '_')}_tile_{tile_sizes}_mantissa.txt"), max_mantissas.cpu().numpy())
                np.savetxt(os.path.join(output_dir, f"{name.replace('.', '_')}_tile_{tile_sizes}_exponent.txt"), max_exps.cpu().numpy())
            else:
                np.savetxt(os.path.join(output_dir, f"{name.replace('.', '_')}_mantissa.txt"), mantissas.cpu().numpy())
                np.savetxt(os.path.join(output_dir, f"{name.replace('.', '_')}_exponent.txt"), exps.cpu().numpy())


        elif precision.lower() == 'int8' or  precision.lower() == 'int4':
            tensor_numpy = param.cpu().numpy()
            file_name = f"{name.replace('.', '_')}.txt"
            file_path = os.path.join(output_dir, file_name)
            np.savetxt(file_path, tensor_numpy.flatten(), fmt="%.6f")
            logging.info(f"Saved weights for {name} to {file_path}")
