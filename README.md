# LLM/CNN Profiling Framework

A framework for profiling Large Language Models (LLMs) with various precision formats. 
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Module Details](#module-details)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation

### Prerequisites

The framework requires Python 3.8+ and the following packages:

```bash
pip install torch transformers matplotlib numpy bitsandbytes
```

### Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
cd profiler-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python main.py --model_type [llm/cnn] --model_name [model_name] --precision [precision_type]
```

### Command Line Arguments

| Argument | Description | Required | Default | Choices |
|----------|-------------|----------|---------|---------|
| `--model_type` | Type of model to profile | Yes | - | `llm`, `cnn` |
| `--model_name` | Name of the model to profile | Yes | - | - |
| `--precision` | Precision level for profiling | No | `fp32` | `fp32`, `fp16`, `fp8`, `int8`, `int4` |
| `--tile_sizes` | Tile sizes for value distribution analysis | No | `[4, 64, 128]` | - |
| `--output_dir` | Directory to save profiling results | No | `output` | - |
| `--plot` | Enable plotting after profiling | No | `False` | - |
| `--verbose` | Enable verbose logging | No | `False` | - |

### Example Commands

Profile an LLM model:
```bash
python main.py --model_type llm --model_name gpt2 --precision fp16 --plot
```

Profile with custom tile sizes:
```bash
python main.py --model_type llm --model_name bert-base-uncased --precision int8 --tile_sizes 32
```

## Project Structure

```
profiler/
├── main.py                 # Main entry point
├── profiler/
│   ├── llm.py             # LLM profiling functionality
│   ├── plotter.py         # Visualization utilities
│   └── utils.py           # Common utilities
├── output/                # Default output directory
└── README.md
```