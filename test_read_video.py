"""
Quick sanity check for the video IO stack.

Run from football_analysis/:
    python test_read_video.py --video input_videos/08fd33_4_small.mp4
"""

import argparse
import os

import cv2

from utils import read_video


def main():
    parser = argparse.ArgumentParser(description="Smoke-test video decoding backends.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--max_frames", type=int, default=0, help="Optional limit to stop after N frames")
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    frames = read_video(args.video, max_frames=max_frames)
    if not frames:
        raise RuntimeError("read_video returned no frames. Verify codecs/backends.")

    h, w = frames[0].shape[:2]
    print(f"[test_read_video] Frames decoded: {len(frames)} | Resolution: {w}x{h}")

    preview_path = os.path.join("output_videos", "test_read_video_first_frame.jpg")
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    cv2.imwrite(preview_path, frames[0])
    print(f"[test_read_video] Saved first frame preview to {preview_path}")


if __name__ == "__main__":
    main()

