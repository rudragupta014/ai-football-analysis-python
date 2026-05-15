
"""
Robust TeamAssigner (extended)

- Keeps previous API (assign_team_color, get_player_team) intact.
- Adds:
    - assign_reference_colors_from_video(frames, players_frames, sample_frames=5)
    - assign_teams_to_tracks(tracks, frames)
    - compute_player_motion_stats(tracks, fps=25.0, smooth_window=3)
- Stores per-frame fields into tracks['players'][frame][pid]:
    - 'team' (int 0/1/2)
    - 'team_color' (BGR tuple)
    - 'center_smoothed' (x,y)
    - 'speed' (pixels per second)
    - 'distance_from_prev' (pixels)
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any, Tuple, List, Optional
import math

class TeamAssigner:
    def __init__(self):
        # drawing colors (BGR)
        self.team_colors = {
            0: (200,200,200),  # unknown / neutral
            1: (0,120,255),    # team 1 (orange-ish)
            2: (0,255,0)       # team 2 (green)
        }
        self._reference_colors = None  # np.ndarray shape (2,3)

    # -------------------------
    # existing methods (kept)
    # -------------------------
    def _safe_kmeans(self, samples: np.ndarray, n_clusters=2):
        try:
            if samples is None or samples.size == 0 or samples.shape[0] < 6:
                return None
            N = samples.shape[0]
            if N > 2000:
                idx = np.random.choice(N, 2000, replace=False)
                samples = samples[idx]
            k = max(1, min(n_clusters, samples.shape[0]))
            if k == 1:
                return np.array([np.mean(samples, axis=0)])
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(samples)
            return kmeans.cluster_centers_
        except Exception:
            return None

    def _extract_region_pixels(self, frame: np.ndarray, bbox: Tuple[float,float,float,float]) -> np.ndarray:
        h, w = frame.shape[:2]
        bx = [int(v) for v in bbox]
        if bx[2] > bx[0] and bx[3] > bx[1]:
            x0, y0, x1, y1 = bx[0], bx[1], bx[2], bx[3]
        else:
            x0, y0, bw, bh = bx[0], bx[1], bx[2], bx[3]
            x1, y1 = x0 + bw, y0 + bh

        x0 = max(0, min(w-1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h-1, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0,3), dtype=np.float32)

        # upper torso region: top 45% of bbox
        top_h = y0 + max(1, int((y1 - y0) * 0.45))
        region = frame[y0:top_h, x0:x1].copy()
        if region.size == 0:
            return np.zeros((0,3), dtype=np.float32)

        # downsize to reduce noise, preserve BGR channels
        try:
            region_small = cv2.resize(region, (24, 24), interpolation=cv2.INTER_AREA)
        except Exception:
            region_small = region

        pixels = region_small.reshape(-1, 3).astype(np.float32)

        # filter out obvious grass/green using HSV thresholds
        try:
            hsv_small = cv2.cvtColor(region_small, cv2.COLOR_BGR2HSV)
            s = hsv_small[:,:,1].reshape(-1)
            v = hsv_small[:,:,2].reshape(-1)
            hch = hsv_small[:,:,0].reshape(-1)
            mask = (s > 25) & (v > 35) & ~((hch >= 35) & (hch <= 85))
            if mask.sum() >= 12:
                pixels = pixels[mask]
        except Exception:
            pass

        return pixels

    def assign_team_color(self, first_frame: np.ndarray, players_first_frame: Dict[int, Dict[str,Any]], 
                          sample_frames: int = 1):
        """
        Backwards-compatible initialization from a single frame (kept).
        """
        samples: List[np.ndarray] = []
        for pid, pinfo in players_first_frame.items():
            bb = pinfo.get('bbox', None)
            if bb is None:
                continue
            px = self._extract_region_pixels(first_frame, bb)
            if px.size > 0:
                samples.append(px)

        if len(samples) == 0:
            # fallback to heuristic colors
            self._reference_colors = np.array([[30.,90.,200.],[60.,200.,60.]])
            return

        all_pixels = np.vstack(samples)
        centers = self._safe_kmeans(all_pixels, n_clusters=2)
        if centers is None or centers.shape[0] < 2:
            order = np.argsort(all_pixels[:,2])
            half = max(1, len(order)//2)
            c0 = np.mean(all_pixels[order[:half]], axis=0)
            c1 = np.mean(all_pixels[order[half:]], axis=0)
            centers = np.vstack([c0, c1])
        self._reference_colors = centers.astype(np.float32)

    def get_player_team(self, frame: np.ndarray, bbox: Tuple[float,float,float,float], player_id: int) -> int:
        """
        Backwards-compatible team query for a single bbox/frame.
        """
        if self._reference_colors is None:
            return 0
        px = self._extract_region_pixels(frame, bbox)
        if px.size == 0:
            return 0
        med = np.median(px, axis=0)
        dists = np.linalg.norm(self._reference_colors - med, axis=1)
        idx = int(np.argmin(dists))
        return 1 if idx == 0 else 2

    # -------------------------
    # new helpers (additive)
    # -------------------------
    def assign_reference_colors_from_video(self,
                                           frames: List[np.ndarray],
                                           players_frames: List[Dict[int, Dict[str,Any]]],
                                           sample_frames: int = 5,
                                           max_samples_per_frame: int = 25):
        """
        Build more robust reference colors by sampling torso pixels across multiple frames.
        - frames: list of frames (np.ndarray)
        - players_frames: list of player dicts with bboxes (same indexing as frames)
        """
        samples_list = []
        n_frames = min(len(frames), len(players_frames))
        if n_frames == 0:
            return

        step = max(1, n_frames // sample_frames)
        for i in range(0, n_frames, step):
            frame = frames[i]
            players = players_frames[i]
            for pid, pinfo in players.items():
                bb = pinfo.get('bbox')
                if bb is None:
                    continue
                px = self._extract_region_pixels(frame, bb)
                if px.size > 0:
                    # subsample px to limit count
                    if px.shape[0] > max_samples_per_frame:
                        idx = np.random.choice(px.shape[0], max_samples_per_frame, replace=False)
                        px = px[idx]
                    samples_list.append(px)
            if len(samples_list) >= sample_frames * 3:
                # enough samples
                break

        if len(samples_list) == 0:
            return

        all_pixels = np.vstack(samples_list)
        centers = self._safe_kmeans(all_pixels, n_clusters=2)
        if centers is None or centers.shape[0] < 2:
            # fallback
            order = np.argsort(all_pixels[:,2])
            half = max(1, len(order)//2)
            c0 = np.mean(all_pixels[order[:half]], axis=0)
            c1 = np.mean(all_pixels[order[half:]], axis=0)
            centers = np.vstack([c0, c1])
        self._reference_colors = centers.astype(np.float32)

    def _bbox_center(self, bbox: Tuple[float,float,float,float]) -> Tuple[Optional[float], Optional[float]]:
        if bbox is None:
            return (None, None)
        try:
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    x, y, w, h = x1, y1, x2, y2
                    cx = x + w/2.0
                    cy = y + h/2.0
                else:
                    cx = (x1 + x2)/2.0
                    cy = (y1 + y2)/2.0
                return (float(cx), float(cy))
        except Exception:
            pass
        return (None, None)

    def assign_teams_to_tracks(self, tracks: Dict[str, Any], frames: List[np.ndarray], sample_frames: int = 5):
        """
        Walk tracks['players'] and assign 'team' and 'team_color' to each player's frame entry.
        - tracks: your project's tracks dict (expects tracks['players'] = list of dicts per frame)
        - frames: list of frames aligned to tracks['players'] indices
        """
        # build robust reference colors if not present
        if self._reference_colors is None:
            # try to sample across first few frames
            players_frames = tracks.get('players', [])[:len(frames)]
            try:
                self.assign_reference_colors_from_video(frames, players_frames, sample_frames=sample_frames)
            except Exception:
                pass

        # iterate frames and assign
        nframes = min(len(tracks.get('players', [])), len(frames))
        for f in range(nframes):
            frame = frames[f]
            for pid, pinfo in tracks['players'][f].items():
                bb = pinfo.get('bbox')
                team = 0
                if bb is not None:
                    try:
                        team = self.get_player_team(frame, bb, pid)
                    except Exception:
                        team = 0
                pinfo['team'] = int(team)
                pinfo['team_color'] = tuple(self.team_colors.get(team, (200,200,200)))

    def compute_player_motion_stats(self, tracks: Dict[str, Any], fps: float = 25.0, smooth_window: int = 3):
        """
        Compute per-frame centers, smoothed centers, speed (pixels/s) and per-frame distance.
        Results are written into each_tracks record as:
            - 'center' (raw center)
            - 'center_smoothed' (smoothed across small window)
            - 'distance_from_prev' (pixels)
            - 'speed' (pixels per second)
        This function is robust to players appearing/disappearing across frames.
        """
        nframes = len(tracks.get('players', []))
        # Build per-player timeline of centers
        player_ids = set()
        for f in range(nframes):
            for pid in tracks['players'][f].keys():
                player_ids.add(pid)
        # initialize per-player arrays
        centers_by_player = {pid: [None] * nframes for pid in player_ids}
        for f in range(nframes):
            for pid, pinfo in tracks['players'][f].items():
                # prefer 'position_adjusted' or 'position' if present
                c = None
                for k in ('position_adjusted', 'position', 'pos_transformed','pos'):
                    v = pinfo.get(k)
                    if isinstance(v, (list,tuple)) and len(v) >= 2 and v[0] is not None:
                        c = (float(v[0]), float(v[1]))
                        break
                if c is None:
                    c = self._bbox_center(pinfo.get('bbox'))
                centers_by_player[pid][f] = c

        # smoothing (simple moving average)
        half = smooth_window // 2
        for pid, centers in centers_by_player.items():
            # compute smoothed centers
            smoothed = [None] * nframes
            for f in range(nframes):
                xs, ys, cnt = 0.0, 0.0, 0
                for k in range(max(0, f - half), min(nframes, f + half + 1)):
                    c = centers[k]
                    if c is None or c[0] is None:
                        continue
                    xs += c[0]; ys += c[1]; cnt += 1
                if cnt > 0:
                    smoothed[f] = (xs / cnt, ys / cnt)
                else:
                    smoothed[f] = (None, None)
            # write values back into tracks
            prev = None
            for f in range(nframes):
                frame_players = tracks['players'][f]
                if pid in frame_players:
                    pinfo = frame_players[pid]
                    raw_c = centers[f]
                    smooth_c = smoothed[f]
                    pinfo['center'] = raw_c
                    pinfo['center_smoothed'] = smooth_c
                    # compute distance from prev (use smoothed positions if available)
                    if prev is not None and smooth_c[0] is not None and prev[0] is not None:
                        d = math.hypot(smooth_c[0] - prev[0], smooth_c[1] - prev[1])
                        pinfo['distance_from_prev'] = float(d)
                        pinfo['speed'] = float(d * fps)  # pixels per second
                    else:
                        pinfo['distance_from_prev'] = 0.0
                        pinfo['speed'] = 0.0
                    prev = smooth_c if smooth_c[0] is not None else prev
                else:
                    # player not present this frame
                    prev = prev

    # -------------------------
    # small utility
    # -------------------------
    def set_team_color_override(self, team_idx: int, bgr_tuple: Tuple[int,int,int]):
        """Allow overriding team colors used for drawing."""
        self.team_colors[int(team_idx)] = tuple(int(x) for x in bgr_tuple)
