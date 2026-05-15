import os

import cv2

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover - fallback import
    iio = None


def read_video(video_path: str, force_backend: str = "auto", max_frames: int = None):
    """
    Robust video reader with codec fallback.

    Tries OpenCV first; if it cannot open/decode frames it falls back to imageio/ffmpeg.
    Returns a list of BGR frames.
    """
    frames = []
    backends_tried = []

    def _read_with_cv2():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture could not open file")
        collected = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                collected.append(frame)
                if max_frames and len(collected) >= max_frames:
                    break
        finally:
            cap.release()
        if not collected:
            raise RuntimeError("cv2.VideoCapture returned zero frames")
        return collected

    def _read_with_imageio():
        if iio is None:
            raise RuntimeError("imageio is not available")
        collected = []
        try:
            for idx, frame in enumerate(iio.imiter(video_path)):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                collected.append(frame_bgr)
                if max_frames and len(collected) >= max_frames:
                    break
        except Exception as exc:
            raise RuntimeError(f"imageio failed: {exc}") from exc
        if not collected:
            raise RuntimeError("imageio returned zero frames")
        return collected

    backend_order = []
    if force_backend == "opencv":
        backend_order = ["opencv"]
    elif force_backend == "imageio":
        backend_order = ["imageio"]
    else:
        backend_order = ["opencv", "imageio"]

    last_error = None
    for backend in backend_order:
        try:
            if backend == "opencv":
                frames = _read_with_cv2()
            elif backend == "imageio":
                frames = _read_with_imageio()
            backends_tried.append(backend)
            if frames:
                if len(backends_tried) > 1:
                    print(f"[read_video] succeeded using fallback backend: {backend}")
                return frames
        except Exception as exc:
            last_error = exc
            backends_tried.append(f"{backend} (failed: {exc})")
            continue

    raise RuntimeError(f"Unable to read video '{video_path}'. Tried backends: {backends_tried}. Last error: {last_error}")


def save_video(ouput_video_frames, output_video_path, fps: float = 24.0):
    if not ouput_video_frames:
        raise ValueError("No frames provided to save_video")
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    height, width = ouput_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_video_path}")
    try:
        for frame in ouput_video_frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
