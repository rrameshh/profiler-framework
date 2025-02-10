import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

bar_colors = ["#96DCF8", "#0B76A0", "#084F6A", "#F6C6AD", "#F1791F", "#9D350B"]

def load_data_from_file(file_path):
    """Load data from a text file."""
    if os.path.exists(file_path):
        return np.loadtxt(file_path)
    else:
        logging.warning(f"File not found: {file_path}")
        return None

def plot_mantissa_exponent_distribution(output_dir, model_name, layers):
    """Plot mantissa and exponent distributions for each layer."""
    fig, axs = plt.subplots(1, len(layers), figsize=(9, 3))  # 1 row, N columns
    if len(layers) == 1:
        axs = [axs] 

    for i, layer in enumerate(layers):
        self_attn_mantissa_files = [
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_k_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_q_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_v_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_o_proj_weight.txt"),
        ]
        mlp_mantissa_files = [
            os.path.join(output_dir, f"model_layers_{layer}_mlp_up_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_mlp_down_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_mlp_gate_proj_weight.txt"),
        ]

        self_attn_mantissas = []
        for file in self_attn_mantissa_files:
            data = load_data_from_file(file)
            if data is not None:
                self_attn_mantissas.extend(data)

        mlp_mantissas = []
        for file in mlp_mantissa_files:
            data = load_data_from_file(file)
            if data is not None:
                mlp_mantissas.extend(data)

        if self_attn_mantissas and mlp_mantissas:
            self_attn_mantissas = np.array(self_attn_mantissas)
            mlp_mantissas = np.array(mlp_mantissas)

            self_attn_percentages = np.bincount(self_attn_mantissas.astype(int), minlength=8) / len(self_attn_mantissas) * 100
            mlp_percentages = np.bincount(mlp_mantissas.astype(int), minlength=8) / len(mlp_mantissas) * 100

            mantissa_x = np.arange(8)

            axs[i].bar(mantissa_x - 0.2, self_attn_percentages, width=0.4, label='Self-Attn Mantissa', color=bar_colors[2])
            axs[i].bar(mantissa_x + 0.2, mlp_percentages, width=0.4, label='MLP Mantissa', color=bar_colors[5])

            axs[i].set_title(f'Layer {layer}', fontsize=15)
            axs[i].set_xlabel('')
            axs[i].set_ylabel('Percentage')
            axs[i].set_ylim([0, 100])
            axs[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

            if i == 0:
                axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_mantissa_exponent_distribution.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

def plot_weight_distribution(output_dir, model_name, layers):
    """Plot weight distributions for each layer."""
    fig, axes = plt.subplots(1, len(layers), figsize=(9, 3))  # 1 row, N columns
    if len(layers) == 1:
        axes = [axes]  # Ensure axes is a list even for a single layer

    for idx, layer in enumerate(layers):
        # Load weight files for self-attention and MLP layers
        self_attn_weight_files = [
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_k_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_q_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_v_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_self_attn_o_proj_weight.txt"),
        ]
        mlp_weight_files = [
            os.path.join(output_dir, f"model_layers_{layer}_mlp_up_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_mlp_down_proj_weight.txt"),
            os.path.join(output_dir, f"model_layers_{layer}_mlp_gate_proj_weight.txt"),
        ]

        # Combine all weight values
        self_attn_weights = []
        for file in self_attn_weight_files:
            data = load_data_from_file(file)
            if data is not None:
                self_attn_weights.extend(data)

        mlp_weights = []
        for file in mlp_weight_files:
            data = load_data_from_file(file)
            if data is not None:
                mlp_weights.extend(data)

        if self_attn_weights and mlp_weights:
            ax = axes[idx]
            ax.hist(self_attn_weights, bins=30, color=bar_colors[2], alpha=0.7, label='Self-Attn Weights')
            ax.hist(mlp_weights, bins=30, color=bar_colors[5], alpha=0.7, label='MLP Weights')
            ax.set_title(f'Layer {layer}', fontsize=15)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.75)
            if idx == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_weight_distribution.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.show()