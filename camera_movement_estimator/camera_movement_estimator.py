import pickle
import cv2
import numpy as np
import os
import sys
from typing import List, Tuple

# if your utils are one level up, keep sys.path usage outside - here it's shown for completeness
# sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mask features near left and right edges (example heuristic; adapt for your resolution)
        mask_features = np.zeros_like(first_frame_grayscale, dtype=np.uint8)
        mask_features[:, 0:20] = 1
        # ensure we do not exceed frame width
        h, w = first_frame_grayscale.shape[:2]
        right_start = max(0, min(w, 900))
        right_end = max(0, min(w, 1050))
        if right_start < right_end:
            mask_features[:, right_start:right_end] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust stored 'position' in tracks by subtracting camera movement per frame.
        """
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position', None)
                    if position is None:
                        continue
                    camera_movement = camera_movement_per_frame[frame_num]
                    # camera_movement expected as [dx, dy]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object_name][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames: List[np.ndarray], read_from_stub=False, stub_path=None) -> List[List[float]]:
        """
        Estimate camera movement per frame using sparse optical flow (LK).
        Returns a list of [dx, dy] for each frame (first entry is [0,0]).
        If read_from_stub True and stub_path exists, loads and returns that instead.
        """
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # initialize camera movement list (separate lists per frame)
        camera_movement = [[0.0, 0.0] for _ in range(len(frames))]

        # prepare first frame features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        # old_features may be None if no good corners
        if old_features is None:
            old_features = np.empty((0, 1, 2), dtype=np.float32)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # if no old features, try to re-initialize
            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                if old_features is None:
                    old_features = np.empty((0, 1, 2), dtype=np.float32)

            if old_features.size == 0:
                # no features to track; keep movement as [0,0] and continue
                old_gray = frame_gray.copy()
                continue

            # calc optical flow (may return None arrays)
            new_features, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            if new_features is None or status is None:
                # optical flow failed for this step
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if old_features is None:
                    old_features = np.empty((0, 1, 2), dtype=np.float32)
                old_gray = frame_gray.copy()
                continue

            # compute the point that moved the most (robust heuristic)
            max_distance = 0.0
            camera_movement_x, camera_movement_y = 0.0, 0.0

            # iterate only over successfully tracked points
            for nf, of, st in zip(new_features, old_features, status):
                if st[0] == 0:
                    continue
                new_pt = nf.ravel()
                old_pt = of.ravel()
                distance = measure_distance(new_pt, old_pt)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_pt, new_pt)

            # If sufficient movement measured, record it and re-detect features on current frame
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                new_good = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if new_good is not None:
                    old_features = new_good
                else:
                    old_features = np.empty((0, 1, 2), dtype=np.float32)
            else:
                # otherwise keep [0,0] (or you could use small movement estimate)
                camera_movement[frame_num] = [0.0, 0.0]

            old_gray = frame_gray.copy()

        # save stub if requested
        if stub_path is not None:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(camera_movement, f)
            except Exception:
                pass

        return camera_movement

    def draw_camera_movement(self, frames: List[np.ndarray], camera_movement_per_frame: List[Tuple[float, float]]):
        """
        Draw compact camera movement overlay (single box top-left) per frame.
        Modifies frames in-place and returns them.
        """
        if frames is None or len(frames) == 0:
            return frames

        n = min(len(frames), len(camera_movement_per_frame))
        for i in range(n):
            frame = frames[i]
            try:
                dx, dy = camera_movement_per_frame[i]
            except Exception:
                dx, dy = 0.0, 0.0

            h, w = frame.shape[:2]
            # draw semi-transparent black box
            box_w, box_h = int(w * 0.34), 44
            x0, y0 = 8, 8
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # text lines
            txt1 = f"Camera Movement X: {dx:+.2f}"
            txt2 = f"Camera Movement Y: {dy:+.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, txt1, (x0 + 8, y0 + 18), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, txt2, (x0 + 8, y0 + 36), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return frames
