"""
visualization.py

Polished match visualization utilities (drop-in).
Requires tracks structure populated by team_assigner.compute_player_motion_stats(...)
and events list from events_detector.detect_passes_and_shots(...).
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

DEFAULT_TEAM_COLORS = {1: (0, 120, 255), 2: (0, 200, 0), 0: (200, 200, 200)}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Tunable overlay knobs
MIN_BADGE_RADIUS = 10
MAX_BADGE_RADIUS = 20
BADGE_SCALE = 0.16
BADGE_FALLBACK = 14
DEFAULT_TRAIL_LEN = 8
TRAIL_ALPHA_MIN = 0.25
TRAIL_ALPHA_MAX = 0.6
PASS_ARROW_THICKNESS = 2
PASS_ARROW_TIP = 0.12
TEXT_OVERLAP_PADDING = 8


def _clamp(value, min_v, max_v):
    return max(min_v, min(max_v, value))


def _center_from_bbox(bb):
    if not bb:
        return (None, None)
    try:
        if len(bb) == 4:
            x1, y1, x2, y2 = bb
            if x2 <= x1 or y2 <= y1:
                x, y, w, h = x1, y1, x2, y2
                cx = x + w / 2.0
                cy = y + h / 2.0
            else:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
            return (float(cx), float(cy))
    except Exception:
        pass
    return (None, None)


def _draw_shadowed_text(img, text, org, font_scale=0.55, thickness=1,
                        text_color=(255, 255, 255), shadow_color=(0, 0, 0), shadow_thickness=None):
    x, y = org
    shadow_thickness = shadow_thickness if shadow_thickness is not None else max(1, thickness + 1)
    cv2.putText(img, text, (x, y), FONT, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, font_scale, text_color, thickness, cv2.LINE_AA)


def _bbox_center_and_width(bb):
    if not bb:
        return (None, None), None
    try:
        if len(bb) == 4:
            x1, y1, x2, y2 = bb
            if x2 <= x1 or y2 <= y1:
                x, y, w, h = x1, y1, x2, y2
                cx = x + w / 2.0
                cy = y + h / 2.0
                return (float(cx), float(cy)), float(w)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return (float(cx), float(cy)), float(x2 - x1)
    except Exception:
        pass
    return (None, None), None


def _desaturate_color(color, factor=0.6):
    avg = sum(color) / 3.0
    muted = []
    for c in color:
        muted.append(int(_clamp(avg + (c - avg) * factor, 0, 255)))
    return tuple(muted)


def _dim_color(color, intensity=0.8):
    return tuple(int(_clamp(c * intensity, 0, 255)) for c in color)


def _resolve_text_offsets(layout, layouts):
    shift_x, shift_y = 0, 0
    for other in layouts:
        if other is layout:
            continue
        if layout['center'][0] is None or other['center'][0] is None:
            continue
        dx = layout['center'][0] - other['center'][0]
        dy = layout['center'][1] - other['center'][1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        min_dist = layout['badge_r'] + other['badge_r'] + TEXT_OVERLAP_PADDING
        if dist < min_dist:
            if abs(dy) < abs(dx):
                shift_x += (layout['badge_r'] + 6) * (1 if dx >= 0 else -1)
            else:
                shift_y -= (layout['badge_r'] + 6)
    return shift_x, shift_y


def _foot_point(bb):
    if not bb:
        return (None, None)
    try:
        x1, y1, x2, y2 = bb
        if x2 <= x1 or y2 <= y1:
            x, y, w, h = x1, y1, x2, y2
            return (x + w / 2.0, y + h)
        return ((x1 + x2) / 2.0, y2)
    except Exception:
        return (None, None)


def draw_team_legend(frame: np.ndarray, team_colors: Dict[int, Tuple[int, int, int]] = None, x: int = 12, y: int = 8):
    if team_colors is None:
        team_colors = DEFAULT_TEAM_COLORS
    h, w = frame.shape[:2]
    box_w, box_h = 200, 56
    x0, y0 = w - box_w - 12, y
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    t1c = team_colors.get(1, (0, 120, 255))
    t2c = team_colors.get(2, (0, 200, 0))
    cv2.rectangle(frame, (x0 + 8, y0 + 12), (x0 + 28, y0 + 32), t1c, -1)
    cv2.putText(frame, "Team 1", (x0 + 36, y0 + 30), FONT, 0.57, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0 + 8, y0 + 34), (x0 + 28, y0 + 54), t2c, -1)
    cv2.putText(frame, "Team 2", (x0 + 36, y0 + 52), FONT, 0.57, (255, 255, 255), 1, cv2.LINE_AA)


def draw_camera_movement_overlay(frame: np.ndarray, cam_mv: Tuple[float, float] = (0.0, 0.0)):
    text1 = f"Camera Movement X: {cam_mv[0]:+.2f}"
    text2 = f"Camera Movement Y: {cam_mv[1]:+.2f}"
    h, w = frame.shape[:2]
    rect_h = 56
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (int(w * 0.72), 6 + rect_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    _draw_shadowed_text(frame, text1, (14, 28), font_scale=1.05, thickness=2)
    _draw_shadowed_text(frame, text2, (14, 52), font_scale=1.05, thickness=2)


def _draw_pass_arrow(frame: np.ndarray, from_pt: Tuple[int, int], to_pt: Tuple[int, int], color: Tuple[int, int, int], label: str = None):
    try:
        fx, fy = int(from_pt[0]), int(from_pt[1])
        tx, ty = int(to_pt[0]), int(to_pt[1])
    except Exception:
        return
    cv2.arrowedLine(frame, (fx, fy), (tx, ty), color, PASS_ARROW_THICKNESS, tipLength=PASS_ARROW_TIP)
    if label:
        lx = tx + 6
        ly = ty - 6
        _draw_shadowed_text(frame, label, (lx, ly), font_scale=0.6, thickness=2, text_color=(255, 255, 255))


def draw_visuals(frame: np.ndarray,
                 tracks: Dict[str, Any],
                 frame_idx: int,
                 events: Optional[List[Dict[str, Any]]] = None,
                 camera_movement: Tuple[float, float] = (0.0, 0.0),
                 team_colors: Dict[int, Tuple[int, int, int]] = None,
                 meters_per_pixel: Optional[float] = None,
                 draw_trails: bool = True,
                 trail_length: Optional[int] = None,
                 action_suggestions: Optional[List[Dict]] = None):
    """
    Main draw function — call for each annotated frame (after tracks & events computed).
    """

    if team_colors is None:
        team_colors = DEFAULT_TEAM_COLORS

    draw_camera_movement_overlay(frame, camera_movement)

    current_trail_len = trail_length if trail_length is not None else DEFAULT_TRAIL_LEN

    if draw_trails:
        player_ids = list({pid for frame_dict in tracks['players'] for pid in frame_dict.keys()})
        for pid in player_ids:
            pts = []
            for t in range(max(0, frame_idx - current_trail_len), frame_idx + 1):
                if t < 0 or t >= len(tracks['players']):
                    continue
                pf = tracks['players'][t].get(pid)
                if not pf:
                    continue
                c = pf.get('center_smoothed') or pf.get('center') or _center_from_bbox(pf.get('bbox'))
                if c and c[0] is not None:
                    pts.append((int(c[0]), int(c[1])))
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                alpha_ratio = (i + 1) / max(1, len(pts) - 1)
                alpha = TRAIL_ALPHA_MIN + (TRAIL_ALPHA_MAX - TRAIL_ALPHA_MIN) * alpha_ratio
                thickness = 1 if alpha < 0.5 else 2
                team = tracks['players'][frame_idx].get(pid, {}).get('team', 0)
                col = team_colors.get(team, (200, 200, 200))
                blended = tuple(int(col[k] * alpha) for k in range(3))
                cv2.line(frame, pts[i], pts[i + 1], blended, thickness, lineType=cv2.LINE_AA)

    players_here = tracks['players'][frame_idx]
    player_layouts: List[Dict[str, Any]] = []
    for pid, pinfo in players_here.items():
        c = pinfo.get('center_smoothed') or pinfo.get('center')
        bbox = pinfo.get('bbox')
        if c is None or c[0] is None:
            c, _ = _bbox_center_and_width(bbox)
            if c[0] is None:
                continue
        _, bbox_w = _bbox_center_and_width(bbox)
        badge_r = BADGE_FALLBACK if bbox_w is None else int(_clamp(bbox_w * BADGE_SCALE, MIN_BADGE_RADIUS, MAX_BADGE_RADIUS))
        team = int(pinfo.get('team', 0))
        color = tuple(int(x) for x in pinfo.get('team_color', team_colors.get(team, (200, 200, 200))))
        player_layouts.append({
            'pid': pid,
            'info': pinfo,
            'center': (int(c[0]), int(c[1])),
            'team': team,
            'color': color,
            'badge_r': badge_r,
        })

    for layout in player_layouts:
        pid = layout['pid']
        pinfo = layout['info']
        cx, cy = layout['center']
        badge_r = layout['badge_r']
        color = layout['color']

        ring_radius = max(18, int(badge_r * 1.2))
        cv2.circle(frame, (cx, cy), ring_radius + 2, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), ring_radius, color, 2, lineType=cv2.LINE_AA)

        speed_value = float(pinfo.get('speed', 0.0))
        distance_value = float(pinfo.get('distance_from_prev', 0.0))
        speed_units = pinfo.get('speed_units', 'pxps')
        distance_units = pinfo.get('distance_units', 'px')

        if speed_units == 'mps':
            speed_text = f"{speed_value * 3.6:.2f} km/h"
        else:
            speed_text = f"{speed_value:.2f} px/s"

        if distance_units == 'm':
            dist_text = f"{distance_value:.2f} m"
        else:
            dist_text = f"{distance_value:.1f} px"

        speed_font = 0.55 * (badge_r / 14)
        text_dx, text_dy = _resolve_text_offsets(layout, player_layouts)
        base_x = cx - ring_radius + text_dx
        base_y = cy + ring_radius + 8 + text_dy
        
        # Make Player ID more prominent - larger font, colored background
        player_id_text = f"ID: {pid}"
        id_font_scale = max(0.7, speed_font * 1.3)  # Larger than other text
        id_text_size = cv2.getTextSize(player_id_text, FONT, id_font_scale, 2)[0]
        id_bg_x1 = base_x - 5
        id_bg_y1 = base_y - id_text_size[1] - 5
        id_bg_x2 = base_x + id_text_size[0] + 5
        id_bg_y2 = base_y + 5
        # Draw colored background for player ID
        cv2.rectangle(frame, (id_bg_x1, id_bg_y1), (id_bg_x2, id_bg_y2), color, -1)
        cv2.rectangle(frame, (id_bg_x1, id_bg_y1), (id_bg_x2, id_bg_y2), (0, 0, 0), 2)
        _draw_shadowed_text(frame, player_id_text, (base_x, base_y), font_scale=id_font_scale, thickness=2,
                            text_color=(255, 255, 255), shadow_color=(0, 0, 0), shadow_thickness=3)
        
        # Speed and distance below player ID
        _draw_shadowed_text(frame, speed_text, (base_x, base_y + int(18 * (badge_r / 14))), font_scale=speed_font, thickness=1,
                            text_color=(0, 0, 0), shadow_color=(255, 255, 255), shadow_thickness=2)
        _draw_shadowed_text(frame, dist_text, (base_x, base_y + int(32 * (badge_r / 14))), font_scale=speed_font, thickness=1,
                            text_color=(0, 0, 0), shadow_color=(255, 255, 255), shadow_thickness=2)

    if events:
        for ev in events:
            if ev.get('frame') != frame_idx:
                continue
            if ev.get('type') == 'pass':
                from_id = ev.get('from_id')
                to_id = ev.get('to_id')
                from_pos = None
                to_pos = None
                for b in range(frame_idx, max(-1, frame_idx - 8), -1):
                    if b < 0 or b >= len(tracks['players']):
                        continue
                    p = tracks['players'][b].get(from_id)
                    if p:
                        from_pos = p.get('center_smoothed') or p.get('center') or _center_from_bbox(p.get('bbox'))
                        if from_pos and from_pos[0] is not None:
                            break
                for a in range(frame_idx, min(len(tracks['players']), frame_idx + 8)):
                    p = tracks['players'][a].get(to_id)
                    if p:
                        to_pos = p.get('center_smoothed') or p.get('center') or _center_from_bbox(p.get('bbox'))
                        if to_pos and to_pos[0] is not None:
                            break
                if from_pos and to_pos and from_pos[0] is not None and to_pos[0] is not None:
                    team = int(ev.get('team', 0))
                    color = _dim_color(team_colors.get(team, (255, 200, 0)), intensity=0.8)
                    label = "PASS"
                    if ev.get('interception'):
                        label = "INTERCEPT"
                    _draw_pass_arrow(frame, (int(from_pos[0]), int(from_pos[1])), (int(to_pos[0]), int(to_pos[1])), color, label=label)
            elif ev.get('type') == 'shot':
                pos = ev.get('pos')
                if pos and pos[0] is not None:
                    x, y = int(pos[0]), int(pos[1])
                    cv2.circle(frame, (x, y), 10, (0, 0, 200), 2)
                    _draw_shadowed_text(frame, "SHOT", (x + 12, y - 6), font_scale=0.55, thickness=1, text_color=(255, 255, 255))

    if action_suggestions:
        for idx, suggestion in enumerate(action_suggestions[:2]):
            visual = suggestion.get("visual", {})
            from_pt = visual.get("from")
            to_pt = visual.get("to")
            if not from_pt or not to_pt:
                continue
            color = (0, 180, 255) if suggestion.get("type") == "pass" else (255, 200, 0)
            _draw_pass_arrow(frame, (int(from_pt[0]), int(from_pt[1])), (int(to_pt[0]), int(to_pt[1])), color, label=suggestion.get("type", "").upper())
            text = f"{suggestion.get('type','').upper()} {suggestion.get('score',0):.2f}"
            _draw_shadowed_text(frame, text, (int(from_pt[0]) + 6, int(from_pt[1]) - 6),
                                font_scale=0.5, thickness=1, text_color=(255, 255, 255), shadow_color=(0, 0, 0))

    draw_team_legend(frame, team_colors)
    return frame

