import argparse
from hiertext_generator.HierTextGenerator import HierTextGenerator

def parse_arguments():
    """
    Parse command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Generate HierText dataset with synthetic data.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_samples", help="Number of train samples", type=int, default=1000)
    parser.add_argument("--val_samples", help="Number of validation samples", type=int, default=200)
    parser.add_argument("--test_samples", help="Number of test samples", type=int, default=200)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def main():
    args = parse_arguments()
    generator = HierTextGenerator.from_config(args.config, debug=args.debug)
    split_sizes = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }
    for split, num_samples in split_sizes.items():
        generator.generate_dataset(args.output_dir, split, num_samples=num_samples)

if __name__ == "__main__":
    main()