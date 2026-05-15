"""
Quick visual regression test for the cleaned overlays.

Usage (from football_analysis/):
    python visual_test_small.py
"""

import argparse
import os
import pickle

import cv2

from visualization import draw_visuals, DEFAULT_TEAM_COLORS


def _bbox_center(bbox):
    if not bbox:
        return (None, None)
    try:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            x, y, w, h = x1, y1, x2, y2
            return (x + w / 2.0, y + h / 2.0)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    except Exception:
        return (None, None)


def _load_tracks(stub_path, frame_count):
    with open(stub_path, "rb") as f:
        tracks = pickle.load(f)
    # ensure all keys exist
    tracks.setdefault("players", [{} for _ in range(frame_count)])
    tracks.setdefault("ball", [{} for _ in range(frame_count)])
    tracks.setdefault("referees", [{} for _ in range(frame_count)])
    players = tracks["players"]
    for frame_players in players:
        for pid, info in frame_players.items():
            if "center" not in info or info["center"] is None:
                info["center"] = _bbox_center(info.get("bbox"))
            info.setdefault("center_smoothed", info.get("center"))
            info.setdefault("team", 1 if pid % 2 == 0 else 2)
            info.setdefault("team_color", DEFAULT_TEAM_COLORS.get(info["team"], (200, 200, 200)))
            info.setdefault("speed", 0.0)
            info.setdefault("distance_from_prev", 0.0)
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Render a single frame with the cleaned visualization.")
    parser.add_argument("--frame_path", type=str, default="debug_first_annotated_frame.jpg")
    parser.add_argument("--stub_tracks", type=str, default="stubs/track_stubs.pkl")
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--out", type=str, default="debug_visual_clean.jpg")
    args = parser.parse_args()

    if not os.path.exists(args.frame_path):
        raise FileNotFoundError(f"Frame image not found at {args.frame_path}")
    frame = cv2.imread(args.frame_path)

    frame_count = args.frame_idx + 1
    if os.path.exists(args.stub_tracks):
        tracks = _load_tracks(args.stub_tracks, frame_count)
    else:
        tracks = {"players": [{ }], "ball": [{}], "referees": [{}]}

    camera_movement = [(0.0, 0.0)] * frame_count
    output = draw_visuals(
        frame.copy(),
        tracks,
        frame_idx=min(args.frame_idx, len(tracks["players"]) - 1),
        events=[],
        camera_movement=camera_movement[0],
        team_colors=DEFAULT_TEAM_COLORS,
        draw_trails=False,
    )
    cv2.imwrite(args.out, output)
    print(f"[visual_test_small] Wrote {args.out}")


if __name__ == "__main__":
    main()

