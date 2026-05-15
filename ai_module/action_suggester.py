"""
action_suggester.py

Heuristic real-time pass/action recommendation module.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.neighbors import KDTree


@dataclass
class SuggestionParams:
    search_radius_m: float = 25.0
    openness_radius_m: float = 8.0
    min_pass_prob: float = 0.25
    xT_weight: float = 0.35
    intercept_threshold: float = 0.6
    shot_zone_m: float = 20.0
    dribble_space_m: float = 12.0


def _ensure_np(point):
    if point is None:
        return None
    return np.array(point, dtype=float)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_to_point(origin: np.ndarray, target: np.ndarray) -> float:
    vec = target - origin
    return math.degrees(math.atan2(vec[1], vec[0]))


def _estimate_xt_value(pos: np.ndarray, pitch_size: Tuple[float, float]) -> float:
    """Simple zone-based expected threat proxy."""
    width, height = pitch_size
    if width <= 0 or height <= 0:
        return 0.0
    x_norm = np.clip(pos[0] / width, 0, 1)
    y_norm = np.clip(pos[1] / height, 0, 1)
    # basic weighting: central lanes more valuable, closer to goal more valuable
    centre_bonus = 1 - abs(0.5 - y_norm) * 1.6
    return float(x_norm * centre_bonus)


def _pass_success_probability(distance_m: float, pressure: float, angle_deg: float) -> float:
    # simple logistic using intuitive weights
    angle_penalty = abs(angle_deg) / 90.0
    z = 1.5 - 0.18 * distance_m - 1.2 * pressure - 0.5 * angle_penalty
    return 1 / (1 + math.exp(-z))


def _estimate_intercept_probability(ball_line: Tuple[np.ndarray, np.ndarray],
                                    opponents: List[np.ndarray],
                                    intercept_radius: float) -> float:
    if not opponents:
        return 0.0
    start, end = ball_line
    line_vec = end - start
    line_len_sq = np.dot(line_vec, line_vec) + 1e-6
    worst = 0.0
    for opp in opponents:
        t = np.dot(opp - start, line_vec) / line_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = start + t * line_vec
        d = _distance(opp, closest)
        prob = math.exp(-d / intercept_radius)
        worst = max(worst, prob)
    return worst


def suggest_actions(frame_idx: int,
                    ball_owner_id: int,
                    tracks: Dict[str, List[Dict[int, Dict]]],
                    meters_per_pixel: Optional[float],
                    params: Optional[SuggestionParams] = None,
                    pitch_size_px: Optional[Tuple[int, int]] = None) -> Dict:
    params = params or SuggestionParams()
    px_to_m = meters_per_pixel if meters_per_pixel and meters_per_pixel > 0 else 1.0
    inverse_m = 1.0 / px_to_m
    search_radius_px = params.search_radius_m * inverse_m
    openness_radius_px = params.openness_radius_m * inverse_m

    suggestions = []

    players_frame = tracks['players'][frame_idx]
    owner = players_frame.get(ball_owner_id)
    if not owner:
        return {"frame": frame_idx, "owner": ball_owner_id, "suggestions": []}
    if not owner.get('has_ball'):
        return {"frame": frame_idx, "owner": ball_owner_id, "suggestions": []}

    owner_center = _ensure_np(owner.get('center_smoothed') or owner.get('center'))
    if owner_center is None or owner_center[0] is None:
        return {"frame": frame_idx, "owner": ball_owner_id, "suggestions": []}
    owner_center = np.array(owner_center, dtype=float)
    owner_team = owner.get('team', 0)

    teammate_points = []
    teammate_ids = []
    opponent_points = []

    for pid, info in players_frame.items():
        center = _ensure_np(info.get('center_smoothed') or info.get('center'))
        if center is None or center[0] is None:
            continue
        if info.get('team', 0) == owner_team and pid != ball_owner_id:
            teammate_points.append(center)
            teammate_ids.append(pid)
        elif info.get('team', 0) in (1, 2) and info.get('team') != owner_team:
            opponent_points.append(center)

    if teammate_points:
        team_tree = KDTree(np.vstack(teammate_points))
    else:
        team_tree = None
    if opponent_points:
        opp_tree = KDTree(np.vstack(opponent_points))
    else:
        opp_tree = None

    pitch_size = pitch_size_px if pitch_size_px else (1280, 720)
    owner_xt = _estimate_xt_value(owner_center, pitch_size)

    # Candidate passes
    if team_tree is not None:
        idxs = team_tree.query_radius(owner_center.reshape(1, -1), r=search_radius_px)[0]
        for idx in idxs:
            target_id = teammate_ids[idx]
            target_point = teammate_points[idx]
            dist_px = _distance(owner_center, target_point)
            dist_m = dist_px * px_to_m
            angle = _angle_to_point(owner_center, target_point)
            pressure = 0.0
            if opp_tree is not None:
                pressure = min(1.0, len(opp_tree.query_radius(target_point.reshape(1, -1), r=openness_radius_px)[0]) / 3.0)

            ball_line = (owner_center, target_point)
            intercept_prob = _estimate_intercept_probability(ball_line, opponent_points, openness_radius_px) if opponent_points else 0.0
            xt_target = _estimate_xt_value(target_point, pitch_size)
            xt_delta = xt_target - owner_xt
            pass_prob = _pass_success_probability(dist_m, pressure, angle) * (1 - intercept_prob)

            score = pass_prob + params.xT_weight * xt_delta
            if pass_prob >= params.min_pass_prob:
                suggestions.append({
                    "type": "pass",
                    "target_id": int(target_id),
                    "score": round(score, 3),
                    "confidence": round(pass_prob, 3),
                    "features": {
                        "distance_m": round(dist_m, 2),
                        "angle_deg": round(angle, 1),
                        "pressure": round(pressure, 2),
                        "xt_delta": round(xt_delta, 3),
                        "intercept_prob": round(intercept_prob, 3),
                    },
                    "visual": {
                        "from": owner_center.tolist(),
                        "to": target_point.tolist(),
                    }
                })

    # Non-pass actions
    # Dribble suggestion
    if opp_tree is not None:
        ahead_point = owner_center + np.array([search_radius_px * 0.5, 0])
        opponents_ahead = len(opp_tree.query_radius(ahead_point.reshape(1, -1), r=openness_radius_px * 1.2)[0])
        space_score = max(0.0, 1 - opponents_ahead / 4.0)
    else:
        space_score = 0.8

    if space_score > 0.3:
        suggestions.append({
            "type": "dribble",
            "target_id": None,
            "score": round(space_score * 0.7, 3),
            "confidence": round(space_score, 3),
            "features": {"space_score": round(space_score, 3)},
            "visual": {"from": owner_center.tolist(), "to": (owner_center + np.array([search_radius_px * 0.4, 0])).tolist()}
        })

    # Shot suggestion heuristic
    goal_x = pitch_size[0] if owner_team == 1 else 0
    goal_point = np.array([goal_x, pitch_size[1] / 2], dtype=float)
    goal_dist_px = _distance(owner_center, goal_point)
    goal_dist_m = goal_dist_px * px_to_m
    if goal_dist_m <= params.shot_zone_m:
        shot_pressure = 0.0
        if opp_tree is not None:
            shot_pressure = min(1.0, len(opp_tree.query_radius(owner_center.reshape(1, -1), r=openness_radius_px)[0]) / 3.0)
        shot_score = max(0.0, 1.2 - 0.05 * goal_dist_m - 0.6 * shot_pressure)
        suggestions.append({
            "type": "shot",
            "target_id": None,
            "score": round(shot_score, 3),
            "confidence": round(shot_score, 3),
            "features": {"distance_m": round(goal_dist_m, 2), "pressure": round(shot_pressure, 2)},
            "visual": {"from": owner_center.tolist(), "to": goal_point.tolist()}
        })

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return {"frame": frame_idx, "owner": int(ball_owner_id), "suggestions": suggestions[:3]}


def collect_action_suggestions(tracks: Dict[str, List[Dict[int, Dict]]],
                               meters_per_pixel: Optional[float],
                               params: Optional[SuggestionParams],
                               debug_frame: Optional[int] = None,
                               debug_owner: Optional[int] = None,
                               pitch_size_px: Optional[Tuple[int, int]] = None) -> Dict[int, List[Dict]]:
    suggestions = {}
    total_frames = len(tracks.get('players', []))
    for frame_idx in range(total_frames):
        players_frame = tracks['players'][frame_idx]
        owner_id = next((pid for pid, info in players_frame.items() if info.get('has_ball')), None)
        if owner_id is None:
            continue
        result = suggest_actions(frame_idx, owner_id, tracks, meters_per_pixel, params, pitch_size_px)
        if result["suggestions"]:
            suggestions[frame_idx] = result["suggestions"]
            if debug_frame is not None and frame_idx == debug_frame and (debug_owner is None or debug_owner == owner_id):
                print(f"[action_suggester] Frame {frame_idx}, owner {owner_id}")
                for s in result["suggestions"]:
                    print(f"  -> {s['type']} (score {s['score']:.2f}) features={s['features']}")
    return suggestions


def write_action_suggestions_csv(suggestions: Dict[int, List[Dict]], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for frame_idx, suggs in suggestions.items():
        for s in suggs:
            rows.append({
                "frame": frame_idx,
                "type": s["type"],
                "target_id": s.get("target_id"),
                "score": s["score"],
                "confidence": s.get("confidence"),
                "features": json.dumps(s.get("features", {})),
            })
    if not rows:
        return
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

