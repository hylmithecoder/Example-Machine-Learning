import argparse
from scripts.train import train

def main():
    parser = argparse.ArgumentParser(description="Train AI Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    print(f"Using config file: {args.config}")
    train()
    
if __name__ == "__main__":
    main()
