
import argparse
import os
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# Project modules - ensure these imports match your repo structure
from utils import read_video, save_video
from trackers import Tracker
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

from team_assigner import TeamAssigner

# AI events module
from ai_module.events_detector import (
    detect_passes_and_shots,
    detect_passes_by_trajectory,
    merge_event_lists,
    export_events_csv,
)

# visualization
from visualization import draw_visuals

# ---------------------------
# small helpers
# ---------------------------
def resize_frames(frames, target_width=None):
    if not target_width:
        return frames
    out = []
    for f in frames:
        h, w = f.shape[:2]
        new_w = int(target_width)
        new_h = int(h * (new_w / w))
        out.append(cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA))
    return out

def get_video_fps(path: str, fallback=25.0):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return fallback
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and fps > 0 else fallback
    except Exception:
        return fallback

def draw_possession_overlay(frame, possession_text, bg_alpha=0.78):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    rect_h = max(50, int(h * 0.07))
    rect_w = min(int(w * 0.92), 1400)
    x = int((w - rect_w) / 2)
    y = 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + rect_w, y + rect_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, rect_h / 60 * 0.9)
    thickness = 2
    text_size, _ = cv2.getTextSize(possession_text, font, font_scale, thickness)
    tx = x + 12
    ty = y + int((rect_h + text_size[1]) / 2)
    cv2.putText(frame, possession_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def save_video_fallback(frames: List[np.ndarray], out_path: str, fps: float):
    if not frames:
        raise ValueError("No frames to save")
    h, w = frames[0].shape[:2]
    ext = out_path.split('.')[-1].lower()
    if ext in ('mp4', 'm4v'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext in ('avi',):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path} size={(w,h)}")
    for idx, f in enumerate(frames):
        if f is None:
            continue
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h))
        if f.dtype != np.uint8:
            f = (np.clip(f, 0, 255)).astype('uint8')
        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()

# ---------------------------
# Pipeline main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="../input_videos/08fd33_4_small.mp4")
    parser.add_argument("--model", type=str, default="../yolov8s.pt")
    parser.add_argument("--stub_tracks", type=str, default="stubs/track_stubs.pkl")
    parser.add_argument("--stub_camera", type=str, default="stubs/camera_movement_stub.pkl")
    parser.add_argument("--resize_width", type=int, default=720)
    parser.add_argument("--out", type=str, default="../output_videos/output_video_with_possession.mp4")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    print("[INFO] Reading video frames.")
    video_frames = read_video(args.video)
    if args.resize_width and args.resize_width > 0:
        video_frames = resize_frames(video_frames, args.resize_width)
    if not video_frames:
        raise RuntimeError("No frames read from input video. Check path or read_video implementation.")
    fps = args.fps if args.fps else get_video_fps(args.video)
    print(f"[INFO] Using FPS = {fps}")

    # --- Tracker / detection step ---
    print("[INFO] Running tracker/detections (this is the expensive step).")
    tracker = Tracker(args.model)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=args.stub_tracks)

    # Ensure positions fields present
    try:
        tracker.add_position_to_tracks(tracks)
    except Exception:
        # If tracker does not provide add_position_to_tracks, proceed (assume positions exist)
        pass

    # camera movement adjust
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path=args.stub_camera)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # view transformer (if available)
    view_transformer = ViewTransformer()
    try:
        view_transformer.add_transformed_position_to_tracks(tracks)
    except Exception:
        pass

    # interpolate ball positions (if method exists)
    if 'ball' in tracks and hasattr(tracker, "interpolate_ball_positions"):
        try:
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        except Exception:
            pass

    # some modules may compute speed/distance — keep but not used as main drawer
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    try:
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    except Exception:
        pass

    # ------------------------
    # Team assignment & motion stats
    # ------------------------
    team_assigner = TeamAssigner()

    try:
        # robust reference color sampling across several frames
        frames_for_sampling = min(40, len(video_frames))
        if frames_for_sampling >= 4:
            team_assigner.assign_reference_colors_from_video(video_frames[:frames_for_sampling],
                                                             tracks['players'][:frames_for_sampling],
                                                             sample_frames=6)
        # annotate tracks in-place with team and team_color
        team_assigner.assign_teams_to_tracks(tracks, video_frames)
        # compute smoothed centers, speed and distance per-player
        team_assigner.compute_player_motion_stats(tracks, fps=fps, smooth_window=3)
    except Exception as e:
        print(f"[WARN] Robust team assignment failed: {e}. Falling back to single-frame assign.")
        try:
            team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        except Exception:
            pass
        # ensure defaults exist
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                if 'team' not in track:
                    track['team'] = 0
                if 'team_color' not in track:
                    track['team_color'] = team_assigner.team_colors.get(track['team'], (200,200,200))

    # ------------------------
    # Ball possession assignment
    # ------------------------
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if 'ball' not in tracks or frame_num >= len(tracks['ball']) or 1 not in tracks['ball'][frame_num]:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) else 0)
            continue
        ball_bbox = tracks['ball'][frame_num][1].get('bbox')
        try:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        except Exception:
            assigned_player = -1
        if assigned_player != -1 and assigned_player in player_track:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player].get('team', 0))
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) else 0)
    team_ball_control = np.array(team_ball_control)

    # ------------------------
    # Possession summary
    # ------------------------
    valid_mask = (team_ball_control == 1) | (team_ball_control == 2)
    total_valid = int(valid_mask.sum())
    t1_frames = int((team_ball_control == 1).sum())
    t2_frames = int((team_ball_control == 2).sum())
    t1_pct = (t1_frames / total_valid * 100) if total_valid else 0.0
    t2_pct = (t2_frames / total_valid * 100) if total_valid else 0.0
    print("\n=== MATCH POSSESSION SUMMARY ===")
    print(f"Team 1 Possession: {t1_pct:.2f}% ({t1_frames} of {total_valid} frames)")
    print(f"Team 2 Possession: {t2_pct:.2f}% ({t2_frames} of {total_valid} frames)")
    print("================================\n")

    # ------------------------
    # Event detection
    # ------------------------
    print("[INFO] Running event detection (passes & shots).")
    events = detect_passes_and_shots(tracks, team_ball_control, fps=fps,
                                     pass_dist_thresh=100.0, pass_speed_thresh=150.0, shot_speed_thresh=500.0)

    fallback_events = detect_passes_by_trajectory(tracks, max_frame_window=6, max_receiver_dist=140.0, min_ball_travel=8.0)
    merge_event_lists(events, fallback_events)
    print(f"[INFO] Total events detected: {len(events)}")
    os.makedirs(os.path.dirname("output_videos/events.csv") or ".", exist_ok=True)
    export_events_csv(events, "output_videos/events.csv")
    print("[INFO] Exported events CSV")

    # ------------------------
    # Draw annotations (original tracker.draw_annotations if available)
    # ------------------------
    # Use your tracker draw_annotations if present to draw basic bounding boxes and labels first
    try:
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    except Exception:
        # fallback: use raw frames
        output_video_frames = video_frames.copy()

    print(f"[DEBUG] annotated frames returned: {len(output_video_frames)}")
    print(f"[DEBUG] possession array length: {len(team_ball_control)}")

    # Force possession overlay and save first debug frame
    final_frames = []
    for i, frm in enumerate(output_video_frames):
        possession_value = int(team_ball_control[i]) if i < len(team_ball_control) else 0
        if possession_value == 1:
            possession_text = f"POSSESSION: Team 1  ({t1_pct:.1f}%)"
        elif possession_value == 2:
            possession_text = f"POSSESSION: Team 2  ({t2_pct:.1f}%)"
        else:
            possession_text = "POSSESSION: None"
        annotated = draw_possession_overlay(frm, possession_text)
        final_frames.append(annotated)
        if i == 0:
            try:
                cv2.imwrite("debug_first_annotated_frame.jpg", annotated)
                print("[DEBUG] Wrote debug_first_annotated_frame.jpg")
            except Exception as e:
                print(f"[DEBUG] could not write debug image: {e}")

    # ------------------------
    # Visual polish using visualization.draw_visuals
    # ------------------------
    # camera_movement_per_frame is expected as list of (x,y) tuples
    # We will call draw_visuals per frame to draw badges, speed, distance, trails, arrows, legend, and camera overlay.
    final_frames2 = []
    for fi, frm in enumerate(final_frames):
        cam_mv = (camera_movement_per_frame[fi][0], camera_movement_per_frame[fi][1]) if fi < len(camera_movement_per_frame) else (0.0, 0.0)
        # If you have pixel->meter scale (meters_per_pixel) set it here; otherwise leave None and values show in px units
        meters_per_pixel = None
        try:
            out_fr = draw_visuals(
                frm,
                tracks,
                fi,
                events=events,
                camera_movement=cam_mv,
                team_colors=team_assigner.team_colors,
                meters_per_pixel=meters_per_pixel,
                draw_trails=True,
                trail_length=12
            )
        except Exception as e:
            # If visualization fails for any frame, fallback to the raw annotated frame
            print(f"[WARN] draw_visuals failed on frame {fi}: {e}")
            out_fr = frm
        final_frames2.append(out_fr)
    final_frames = final_frames2

    # ------------------------
    # Save video
    # ------------------------
    out_path = args.out
    if not out_path:
        out_path = "../output_videos/output_video_with_possession.mp4"
    print(f"[INFO] Preparing to save {len(final_frames)} frames to {out_path}")
    try:
        save_video(final_frames, out_path)
        print("[INFO] Saved video using utils.save_video")
    except Exception as e:
        print(f"[WARN] utils.save_video failed: {e}. Using fallback writer.")
        try:
            save_video_fallback(final_frames, out_path, fps)
            print("[INFO] Saved video using fallback writer.")
        except Exception as e2:
            print(f"[ERROR] fallback writer failed: {e2}")
            raise

    print("[DONE] Processing complete.")

if __name__ == "__main__":
    main()
