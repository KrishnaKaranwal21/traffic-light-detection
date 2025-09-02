import os
import cv2
import numpy as np
import time

# ---------------- CONFIG ----------------
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 120, 120]), 'upper': np.array([15, 255, 255])},
        {'lower': np.array([165, 120, 120]), 'upper': np.array([180, 255, 255])}
    ],
    'yellow': [
        {'lower': np.array([18, 90, 110]), 'upper': np.array([32, 255, 255])}
    ],
    'green': [
        {'lower': np.array([40, 40, 40]), 'upper': np.array([90, 255, 255])}
    ]
}
BOX_COLORS = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
MIN_CONTOUR_AREA = 150


# ---------------- DETECTION HELPERS ----------------
def detect_light_blobs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for color, ranges in COLOR_RANGES.items():
        mask_total = None
        for r in ranges:
            mask = cv2.inRange(hsv, r['lower'], r['upper'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        if mask_total is None:
            continue

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((color, (x, y, w, h)))
    return detections


def cluster_lights(detections, max_vertical_gap=120, max_horizontal_gap=80):
    clusters = []
    detections_sorted = sorted(detections, key=lambda d: d[1][0])  # sort by x

    for color, (x, y, w, h) in detections_sorted:
        cx, cy = x + w // 2, y + h // 2
        matched = False

        for cluster in clusters:
            cxs = [b[1][0] + b[1][2] // 2 for b in cluster]
            cys = [b[1][1] + b[1][3] // 2 for b in cluster]

            if (abs(cx - np.mean(cxs)) < max_horizontal_gap and
                    abs(cy - np.mean(cys)) < max_vertical_gap * 2):
                cluster.append((color, (x, y, w, h)))
                matched = True
                break

        if not matched:
            clusters.append([(color, (x, y, w, h))])

    return clusters


def get_active_color(cluster, last_color=None):
    """Pick the color of the largest blob in this cluster. Fallback to last_color."""
    if not cluster:
        return last_color
    largest = max(cluster, key=lambda c: c[1][2] * c[1][3])
    return largest[0]


# ---------------- MAIN PROCESSOR (no GUI) ----------------
def process_video(input_path: str, output_path: str) -> bool:
    """
    Reads a video, runs your traffic-light detector, draws boxes/labels/fps,
    and writes a processed video to 'output_path'.

    Returns True on success, False otherwise.
    """
    try:
        if not os.path.exists(input_path):
            print(f"[process_video] Input does not exist: {input_path}")
            return False

        # Ensure output folder exists
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[process_video] Could not open video: {input_path}")
            return False

        # Read one frame early to guarantee width/height != 0 for VideoWriter
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            print("[process_video] Could not read first frame.")
            cap.release()
            return False

        height, width = first_frame.shape[:2]

        # FPS can be 0 on some files; pick a safe fallback
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1:
            fps = 25.0

        # Try mp4 writer (works well on Windows; for Cloud, we only serve as download)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not writer.isOpened():
            print("[process_video] VideoWriter failed to open. Aborting.")
            cap.release()
            return False

        prev_frame_time = time.time()
        last_colors = {}  # remembers last color per TL id

        # ---- process first frame then the rest ----
        def draw_overlays(frame, fps_val):
            detections = detect_light_blobs(frame)
            clusters = cluster_lights(detections)

            for i, cluster in enumerate(clusters, start=1):
                active_color = get_active_color(cluster, last_colors.get(i, "red"))
                last_colors[i] = active_color

                all_x = [b[1][0] for b in cluster]
                all_y = [b[1][1] for b in cluster]
                all_x2 = [b[1][0] + b[1][2] for b in cluster]
                all_y2 = [b[1][1] + b[1][3] for b in cluster]
                cluster_box = (min(all_x), min(all_y),
                               max(all_x2) - min(all_x), max(all_y2) - min(all_y))

                cv2.rectangle(
                    frame,
                    (cluster_box[0], cluster_box[1]),
                    (cluster_box[0] + cluster_box[2], cluster_box[1] + cluster_box[3]),
                    BOX_COLORS.get(active_color, (255, 255, 255)),
                    3
                )

                cv2.putText(
                    frame, f"TL-{i}",
                    (cluster_box[0], max(0, cluster_box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    BOX_COLORS.get(active_color, (255, 255, 255)),
                    2
                )

                cv2.putText(
                    frame, active_color.upper(),
                    (cluster_box[0], min(frame.shape[0] - 5, cluster_box[1] + cluster_box[3] + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    BOX_COLORS.get(active_color, (255, 255, 255)),
                    2
                )

            # draw FPS
            cv2.putText(
                frame, f"FPS: {int(fps_val)}",
                (frame.shape[1] - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )
            return frame

        # first frame
        now = time.time()
        fps_est = 1.0 / max(1e-6, (now - prev_frame_time))
        prev_frame_time = now
        frame = draw_overlays(first_frame, fps_est)
        writer.write(frame)

        # remaining frames
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            now = time.time()
            fps_est = 1.0 / max(1e-6, (now - prev_frame_time))
            prev_frame_time = now

            frame = draw_overlays(frame, fps_est)
            writer.write(frame)

        cap.release()
        writer.release()

        ok = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
        if not ok:
            print("[process_video] Output file missing or empty.")
        return ok

    except Exception as e:
        print(f"[process_video] Exception: {e}")
        return False
