"""
Inspect pipeline outputs quickly:

    python debug_inspect.py --events output_videos/events.csv
"""

import argparse
import os
import pandas as pd
import cv2


def main():
    parser = argparse.ArgumentParser(description="Inspect generated artifacts for quick debugging.")
    parser.add_argument("--events", type=str, default="output_videos/events.csv", help="Path to events CSV")
    parser.add_argument("--head", type=int, default=20, help="Number of rows to preview from events CSV")
    parser.add_argument("--debug_frame", type=str, default="debug_first_annotated_frame.jpg",
                        help="Path to the saved debug frame")
    args = parser.parse_args()

    if os.path.exists(args.events):
        df = pd.read_csv(args.events)
        print(f"[debug_inspect] Loaded {len(df)} events from {args.events}")
        with pd.option_context("display.max_columns", None):
            print(df.head(args.head))
    else:
        print(f"[debug_inspect] Events file not found at {args.events}")

    if os.path.exists(args.debug_frame):
        img = cv2.imread(args.debug_frame)
        if img is not None:
            h, w = img.shape[:2]
            print(f"[debug_inspect] Debug frame '{args.debug_frame}' size: {w}x{h}")
        else:
            print(f"[debug_inspect] Failed to read debug frame at {args.debug_frame}")
    else:
        print(f"[debug_inspect] Debug frame not found at {args.debug_frame}")


if __name__ == "__main__":
    main()

