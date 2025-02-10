import argparse
import logging
import os
from profiler.llm import profile_llm
from profiler.plotter import plot_mantissa_exponent_distribution, plot_weight_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Unified Framework for Profiling LLMs and CNNs")
    parser.add_argument("--model_type", type=str, required=True, choices=["llm", "cnn"], help="Type of model to profile (llm or cnn)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to profile")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "fp8", "int8", "int4"], help="Precision level for profiling")
    parser.add_argument("--tile_sizes", type=int, nargs="+", default=[4, 64, 128], help="Tile sizes for value distribution analysis")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save profiling results")
    parser.add_argument("--plot", action="store_true", help="Enable plotting after profiling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.model_type == "llm":
            logging.info(f"Profiling LLM: {args.model_name} with precision {args.precision}")
            profile_llm(args.model_name, args.precision, None, args.tile_sizes, args.output_dir)

        if args.plot:
            logging.info(f"Plotting results for {args.model_name}")
            # layers = [0, 14, 28]
            if args.precision == 'fp8' or  args.precision == 'fp16' or args.precision == 'fp32':
                plot_mantissa_exponent_distribution(args.output_dir, args.model_name, layers)
            plot_weight_distribution(args.output_dir, args.model_name, layers)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()