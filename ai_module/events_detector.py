
"""
events_detector.py

Improved heuristic pass & shot event detection utilities.
Works from tracking data (no re-reading of video).

Main improvements:
- Prefer positional 'position' or 'position_adjusted' stored in tracks (if present) instead of bbox center.
- When detecting passes, require sender and receiver to belong to same team (label by team).
- Use robust nearest-player lookup across short windows to find from_id and to_id.
- Add extra sanity checks (min ball travel, min receiver proximity).
- draw_events_on_frames now accepts `tracks` to color events by team and draw arrows between players.
"""

import csv
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[Optional[float], Optional[float]]:
    if bbox is None:
        return (None, None)
    try:
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # handle both formats [x1,y1,x2,y2] and [x,y,w,h]
            if x2 <= x1 or y2 <= y1:
                x, y, w, h = x1, y1, x2, y2
                cx = x + w / 2.0
                cy = y + h / 2.0
            else:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
            return (cx, cy)
    except Exception:
        pass
    return (None, None)

def euclid(a, b):
    if a is None or b is None:
        return float('inf')
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _get_ball_center_from_tracks(tracks: Dict[str, Any], frame_idx: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Prefer 'position' or 'position_adjusted' stored in ball track, else fallback to bbox_center.
    """
    try:
        ball_entry = tracks['ball'][frame_idx].get(1, {})
    except Exception:
        ball_entry = {}
    # candidate keys in order of preference
    for k in ('position_adjusted', 'position', 'pos_transformed', 'pos'):
        v = ball_entry.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2 and v[0] is not None:
            return (float(v[0]), float(v[1]))
    # fallback to bbox center
    bb = ball_entry.get('bbox', None)
    return bbox_center(bb)

def _get_player_center_from_tracks(tracks: Dict[str, Any], frame_idx: int, pid) -> Tuple[Optional[float], Optional[float]]:
    try:
        p = tracks['players'][frame_idx].get(pid, {})
    except Exception:
        p = {}
    for k in ('position_adjusted', 'position', 'pos_transformed', 'pos'):
        v = p.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2 and v[0] is not None:
            return (float(v[0]), float(v[1]))
    return bbox_center(p.get('bbox', None))

def compute_ball_speed_from_tracks(tracks: Dict[str, Any], fps: float) -> List[float]:
    num_frames = len(tracks['players'])
    centers = [ _get_ball_center_from_tracks(tracks, f) for f in range(num_frames) ]
    speeds = [0.0] * num_frames
    for i in range(1, num_frames):
        a = centers[i-1]
        b = centers[i]
        if a[0] is None or b[0] is None:
            speeds[i] = 0.0
            continue
        dist = euclid(a,b)
        speeds[i] = dist * fps
    return speeds

def _nearest_player_to_point(frame_players: Dict[int, Dict], point: Tuple[float,float]) -> Tuple[Optional[int], float]:
    best_pid = None
    best_d = float('inf')
    if point[0] is None:
        return (None, best_d)
    for pid, pinfo in frame_players.items():
        # try to get center from bbox or precomputed positions
        if 'position_adjusted' in pinfo:
            pc = pinfo['position_adjusted']
        elif 'position' in pinfo:
            pc = pinfo['position']
        else:
            bb = pinfo.get('bbox')
            pc = bbox_center(bb)
        if pc[0] is None:
            continue
        d = euclid(pc, point)
        if d < best_d:
            best_d = d
            best_pid = pid
    return (best_pid, best_d)

def detect_passes_and_shots(tracks: Dict[str, Any],
                            team_ball_control: np.ndarray,
                            fps: float = 25.0,
                            pass_dist_thresh: float = 200.0,
                            pass_speed_thresh: float = 300.0,
                            shot_speed_thresh: float = 700.0,
                            possession_change_window: int = 10,
                            sender_lookup_window: int = 6,
                            receiver_lookup_window: int = 6,
                            min_receiver_proximity: float = 120.0,
                            min_ball_travel_for_pass: float = 8.0
                            ) -> List[Dict[str, Any]]:
    """
    Improved pass detection.

    Logic:
    - Prefer to detect a pass when (A) possession changes between teams OR (B) ball speed/travel exceed thresholds.
    - When a candidate pass frame f is found:
        - Find probable sender by searching backwards up to `sender_lookup_window` frames for player with has_ball or nearest to ball.
        - Find probable receiver by searching forwards up to `receiver_lookup_window` frames for nearest player on receiving team.
        - Require sender and receiver to not be None, and (optionally) be on the same team (to mark as 'pass'). If receiver belongs to opposite team, classify as 'interception' and still record (optional).
        - Use ball travel distance between sender frame and receiver frame to filter small spurious movements.
    """
    num_frames = len(tracks['players'])
    ball_centers = [ _get_ball_center_from_tracks(tracks, f) for f in range(num_frames) ]
    ball_speeds = compute_ball_speed_from_tracks(tracks, fps)
    events: List[Dict[str, Any]] = []

    # per-frame player with ball (from has_ball), fallback to nearest player to ball
    per_frame_player_with_ball = [None] * num_frames
    for f in range(num_frames):
        for pid, pinfo in tracks['players'][f].items():
            if pinfo.get('has_ball', False):
                per_frame_player_with_ball[f] = pid
                break
        # fallback: if no has_ball and ball present, pick nearest player within a reasonable distance
        if per_frame_player_with_ball[f] is None and ball_centers[f][0] is not None:
            pid, d = _nearest_player_to_point(tracks['players'][f], ball_centers[f])
            # only accept if reasonably close (heuristic)
            if d < min_receiver_proximity * 1.2:
                per_frame_player_with_ball[f] = pid

    for f in range(1, num_frames):
        cur_team = int(team_ball_control[f]) if f < len(team_ball_control) else 0
        prev_team = int(team_ball_control[f-1]) if (f-1) < len(team_ball_control) else 0

        # candidate when possession changes between teams - strong signal for pass/interception
        candidate_pass = False
        if cur_team != prev_team and cur_team in (1,2) and prev_team in (1,2):
            candidate_pass = True

        # also candidate if ball speed or travel suggests a kick
        if not candidate_pass:
            speed = ball_speeds[f] if f < len(ball_speeds) else 0.0
            # simple travel between this frame and up to 3 frames earlier
            look_back = max(1, min(3, f))
            dist_moved = 0.0
            if ball_centers[f][0] is not None and ball_centers[f - look_back][0] is not None:
                dist_moved = euclid(ball_centers[f], ball_centers[f - look_back])
            if speed >= pass_speed_thresh or dist_moved >= pass_dist_thresh or dist_moved >= min_ball_travel_for_pass:
                candidate_pass = True

        if not candidate_pass:
            # shot detection by speed spike handled below
            # continue loop
            pass

        if candidate_pass:
            # find sender: look backward up to sender_lookup_window frames; prefer frames with known has_ball
            sender_id = None
            sender_frame = None
            for b in range(f, max(-1, f - sender_lookup_window), -1):
                if b < 0:
                    break
                if per_frame_player_with_ball[b] is not None:
                    sender_id = per_frame_player_with_ball[b]
                    sender_frame = b
                    break
                # fallback: nearest player to ball at frame b
                if ball_centers[b][0] is not None:
                    pid_near, dnear = _nearest_player_to_point(tracks['players'][b], ball_centers[b])
                    if pid_near is not None and dnear < min_receiver_proximity * 1.5:
                        sender_id = pid_near
                        sender_frame = b
                        break

            # find receiver: look forward up to receiver_lookup_window frames and pick nearest player to ball
            receiver_id = None
            receiver_frame = None
            for a in range(f, min(num_frames, f + receiver_lookup_window)):
                if ball_centers[a][0] is None:
                    continue
                pid_near, dnear = _nearest_player_to_point(tracks['players'][a], ball_centers[a])
                if pid_near is not None and dnear < min_receiver_proximity:
                    receiver_id = pid_near
                    receiver_frame = a
                    break

            # as a fallback, pick nearest player on current frame f
            if receiver_id is None and ball_centers[f][0] is not None:
                pid_near, dnear = _nearest_player_to_point(tracks['players'][f], ball_centers[f])
                if pid_near is not None and dnear < min_receiver_proximity * 1.6:
                    receiver_id = pid_near
                    receiver_frame = f

            # do additional checks
            # require at least sender or receiver to exist
            if sender_id is None and receiver_id is None:
                continue

            # determine teams
            sender_team = None
            receiver_team = None
            if sender_id is not None and sender_frame is not None:
                sender_team = tracks['players'][sender_frame].get(sender_id, {}).get('team', 0)
            if receiver_id is not None and receiver_frame is not None:
                receiver_team = tracks['players'][receiver_frame].get(receiver_id, {}).get('team', 0)

            # compute travel distance between sender frame and receiver frame positions (use ball centers)
            travel = None
            if sender_frame is not None and receiver_frame is not None:
                p1 = ball_centers[sender_frame]
                p2 = ball_centers[receiver_frame]
                if p1[0] is not None and p2[0] is not None:
                    travel = euclid(p1, p2)

            # final filter: require reasonable travel OR sender/receiver id mismatch (indicating pass)
            accept = False
            # if both teams equal and >0 => standard pass
            if sender_team and receiver_team and sender_team == receiver_team and sender_team in (1,2):
                # ensure receiver exists within proximity or travel is significant
                if receiver_id is not None and (travel is None or travel >= min_ball_travel_for_pass or ball_speeds[f] >= pass_speed_thresh):
                    accept = True
            else:
                # If teams differ, it might be an interception — still useful to record but label differently.
                # Accept only if travel is significant (avoid spurious)
                if travel is not None and travel >= min_ball_travel_for_pass and (receiver_id is not None):
                    accept = True

            if not accept:
                continue

            ev_type = 'pass'
            ev_team = sender_team if sender_team is not None else (receiver_team if receiver_team is not None else int(team_ball_control[f] if f < len(team_ball_control) else 0))
            # if teams differ, mark event as 'pass' but set an 'interception' flag
            interception = False
            if sender_team and receiver_team and sender_team != receiver_team:
                interception = True

            events.append({
                'type': ev_type,
                'frame': f,
                'from_id': sender_id,
                'to_id': receiver_id,
                'team': int(ev_team) if ev_team is not None else 0,
                'ball_speed': ball_speeds[f] if f < len(ball_speeds) else 0.0,
                'pos': ball_centers[f],
                'travel': travel,
                'interception': interception
            })

        # shot detection via speed spike (independent of candidate_pass)
        if ball_speeds[f] >= shot_speed_thresh:
            # find shooter: nearest player in last 3 frames
            shooter = None
            bestd = float('inf')
            for b in range(max(0, f - 3), f + 1):
                for pid, pinfo in tracks['players'][b].items():
                    pc = _get_player_center_from_tracks(tracks, b, pid)
                    if pc[0] is None:
                        continue
                    d = euclid(pc, ball_centers[f])
                    if d < bestd:
                        bestd = d
                        shooter = pid
            events.append({
                'type': 'shot',
                'frame': f,
                'from_id': shooter,
                'to_id': None,
                'team': int(team_ball_control[f]) if f < len(team_ball_control) else 0,
                'ball_speed': ball_speeds[f],
                'pos': ball_centers[f]
            })

    return events

def detect_passes_by_trajectory(tracks, max_frame_window=6, max_receiver_dist=140.0, min_ball_travel=10.0):
    """
    Keep the fallback behavior but ensure teams are respected and distances are stricter.
    """
    num_frames = len(tracks['players'])
    ball_centers = []
    for f in range(num_frames):
        ball_centers.append(_get_ball_center_from_tracks(tracks, f))

    fallback_events = []
    for f in range(0, num_frames - max_frame_window):
        bc1 = ball_centers[f]
        bc2 = ball_centers[f + max_frame_window]
        if bc1[0] is None or bc2[0] is None:
            continue
        travel = euclid(bc1, bc2)
        if travel < min_ball_travel:
            continue
        # find nearest start and end players
        start_pid, start_d = None, float('inf')
        for pid, pinfo in tracks['players'][f].items():
            pc = _get_player_center_from_tracks(tracks, f, pid)
            d = euclid(pc, bc1)
            if d < start_d:
                start_d = d
                start_pid = pid
        end_pid, end_d = None, float('inf')
        for pid, pinfo in tracks['players'][f + max_frame_window].items():
            pc = _get_player_center_from_tracks(tracks, f + max_frame_window, pid)
            d = euclid(pc, bc2)
            if d < end_d:
                end_d = d
                end_pid = pid
        if start_pid is None or end_pid is None or start_pid == end_pid:
            continue
        start_team = tracks['players'][f].get(start_pid, {}).get('team', 0)
        end_team = tracks['players'][f + max_frame_window].get(end_pid, {}).get('team', 0)
        if start_d < max_receiver_dist and end_d < max_receiver_dist:
            fallback_events.append({
                'type': 'pass',
                'frame': f + max_frame_window,
                'from_id': start_pid,
                'to_id': end_pid,
                'team': start_team,
                'ball_speed': travel,
                'pos': bc2,
                'travel': travel,
                'interception': (start_team != end_team)
            })
    return fallback_events

def merge_event_lists(primary: List[dict], secondary: List[dict]):
    existing_keys = {(e.get('type'), e.get('frame'), e.get('from_id'), e.get('to_id')) for e in primary}
    for e in secondary:
        key = (e.get('type'), e.get('frame'), e.get('from_id'), e.get('to_id'))
        if key not in existing_keys:
            primary.append(e)
            existing_keys.add(key)
    return primary

def draw_events_on_frames(frames: List, events: List[dict], tracks: Optional[Dict[str, Any]] = None, team_colors: Optional[Dict[int, Tuple[int,int,int]]] = None):
    """
    Draw events. If `tracks` is provided, use player positions to draw arrows from sender to receiver and color by team.
    Otherwise draw simple markers (circle + label).
    """
    # team colors default
    if team_colors is None and tracks is not None:
        # try to fetch from track entries (fallback)
        team_colors = {}
        # look into player frames for 'team_color' if present
        for f in range(min(3, len(frames))):
            for pid, pinfo in tracks['players'][f].items():
                if 'team' in pinfo and 'team_color' in pinfo:
                    team_colors[pinfo['team']] = pinfo['team_color']
    if team_colors is None:
        team_colors = {1: (0,120,255), 2:(0,255,0), 0:(200,200,200)}

    for ev in events:
        f = ev['frame']
        if f < 0 or f >= len(frames):
            continue
        pos = ev.get('pos', (None, None))
        if pos[0] is None:
            # try to recover position from tracks/from/to ids
            if tracks is not None:
                if ev.get('from_id') is not None:
                    pos = _get_player_center_from_tracks(tracks, min(f, len(tracks['players'])-1), ev['from_id'])
                elif ev.get('to_id') is not None:
                    pos = _get_player_center_from_tracks(tracks, min(f, len(tracks['players'])-1), ev['to_id'])
        if pos[0] is None:
            continue
        x, y = int(pos[0]), int(pos[1])

        if ev['type'] == 'pass':
            color = team_colors.get(ev.get('team',0), (0,180,200))
            # prefer drawing an arrow between from_id and to_id if tracks provided
            if tracks is not None and ev.get('from_id') is not None and ev.get('to_id') is not None:
                # find positions (try receiver frame first, else current frame)
                from_pos = None
                to_pos = None
                # search backward for sender location
                max_search = 8
                for b in range(max(0, f - max_search), min(len(tracks['players']), f + 1)):
                    if ev.get('from_id') in tracks['players'][b]:
                        from_pos = _get_player_center_from_tracks(tracks, b, ev['from_id'])
                        break
                # search forward for receiver location
                for a in range(f, min(len(tracks['players']), f + max_search)):
                    if ev.get('to_id') in tracks['players'][a]:
                        to_pos = _get_player_center_from_tracks(tracks, a, ev['to_id'])
                        break
                # fallback to pos if not found
                if from_pos is None or from_pos[0] is None:
                    from_pos = pos
                if to_pos is None or to_pos[0] is None:
                    to_pos = pos
                fx, fy = int(from_pos[0]), int(from_pos[1])
                tx, ty = int(to_pos[0]), int(to_pos[1])
                # draw arrowed line
                cv2.arrowedLine(frames[f], (fx, fy), (tx, ty), color, 2, tipLength=0.15)
                # label
                lbl = "PASS"
                if ev.get('interception'):
                    lbl = "INTERCEPT"
                cv2.putText(frames[f], lbl, (tx + 6, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                # simply draw a circle + label
                cv2.circle(frames[f], (x, y), 10, color, -1)
                cv2.putText(frames[f], "PASS", (x + 14, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        elif ev['type'] == 'shot':
            color = (0,0,220)
            cv2.circle(frames[f], (x, y), 12, color, -1)
            cv2.putText(frames[f], "SHOT", (x + 14, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            # generic label for unknown event type
            color = (120,120,120)
            cv2.circle(frames[f], (x, y), 8, color, -1)
            cv2.putText(frames[f], ev.get('type','EV').upper(), (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def export_events_csv(events: List[dict], csv_path: str):
    import os
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    fieldnames = ['type', 'frame', 'from_id', 'to_id', 'team', 'interception', 'ball_speed', 'travel', 'pos_x', 'pos_y']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ev in events:
            px = ev.get('pos', (None,None))[0] if ev.get('pos') else ''
            py = ev.get('pos', (None,None))[1] if ev.get('pos') else ''
            w.writerow({
                'type': ev.get('type'),
                'frame': ev.get('frame'),
                'from_id': ev.get('from_id'),
                'to_id': ev.get('to_id'),
                'team': ev.get('team'),
                'interception': ev.get('interception', False),
                'ball_speed': ev.get('ball_speed'),
                'travel': ev.get('travel'),
                'pos_x': px if px is not None else '',
                'pos_y': py if py is not None else ''
            })
