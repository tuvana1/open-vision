"""CLI for SafeDrive Vision."""

import sys
import os
from detector import process_video


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/annotated_{os.path.basename(input_path)}"
    
    # TODO: Add CLI flags for confidence threshold, output format
    process_video(input_path, output_path)


if __name__ == "__main__":
    main()
