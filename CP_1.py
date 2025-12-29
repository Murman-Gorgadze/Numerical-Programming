# autobahn_motion.py
# Requirements: Python 3.10+, opencv-python, numpy, scipy, scikit-learn, pandas, matplotlib
# Install: pip install opencv-python numpy scipy scikit-learn pandas matplotlib

import argparse, os, math, json, itertools, warnings
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Utility
# ---------------------------


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def moving_average_1d(x, k=3):
    if k <= 1: return x
    k = max(1, int(k))
    k = k + 1 - (k % 2)  # force odd
    return savgol_filter(x, window_length=k, polyorder=min(2, k-1), deriv=0)

# ---------------------------
# Simple Multi-Object Tracker (centroid + Hungarian + max_age)
# ---------------------------


class Track:
    _next_id = 1

    def __init__(self, cx, cy, bbox, frame_idx):
        self.id = Track._next_id; Track._next_id += 1
        self.history = []            # (frame_idx, cx, cy, w, h)
        self.add(frame_idx, cx, cy, bbox)
        self.missed = 0

    def add(self, frame_idx, cx, cy, bbox):
        x, y, w, h = bbox
        self.history.append((frame_idx, float(cx), float(cy), float(w), float(h)))

    def last_centroid(self):
        _, cx, cy, _, _ = self.history[-1]
        return np.array([cx, cy])

class CentroidTracker:
    def __init__(self, max_age=15, dist_thresh=80.0):
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.tracks = []

    def update(self, detections, frame_idx):
        # detections: list of (x, y, w, h)
        det_centroids = np.array([[x + w/2, y + h/2] for (x,y,w,h) in detections], dtype=np.float32)
        track_centroids = np.array([t.last_centroid() for t in self.tracks], dtype=np.float32) if self.tracks else np.zeros((0,2), np.float32)

        if len(self.tracks) == 0:
            for bbox in detections:
                cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
                self.tracks.append(Track(cx, cy, bbox, frame_idx))
            return

        if len(detections) == 0:
            # no detections; age all tracks
            for t in self.tracks: t.missed += 1
        else:
            C = cdist(track_centroids, det_centroids) if len(track_centroids) and len(det_centroids) else np.empty((0,0))
            assigned_tracks = set()
            assigned_dets = set()

            if C.size:
                r_idx, c_idx = linear_sum_assignment(C)
                for r, c in zip(r_idx, c_idx):
                    if C[r, c] <= self.dist_thresh:
                        t = self.tracks[r]
                        x, y, w, h = detections[c]
                        cx, cy = x + w/2, y + h/2
                        t.add(frame_idx, cx, cy, (x,y,w,h))
                        t.missed = 0
                        assigned_tracks.add(r); assigned_dets.add(c)

            # unmatched tracks
            for i, t in enumerate(self.tracks):
                if i not in assigned_tracks:
                    t.missed += 1

            # new tracks for unmatched detections
            for j, bbox in enumerate(detections):
                if j not in assigned_dets:
                    cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
                    self.tracks.append(Track(cx, cy, bbox, frame_idx))

        # prune old
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

# ---------------------------
# Detection via background subtraction + morphology + contour filtering
# ---------------------------


def detect_vehicles(frame, fgbg, min_area=400, dilate_iters=2):
    fgmask = fgbg.apply(frame)
    # remove shadows (optional)
    _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=dilate_iters)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # reject extreme aspect ratios to avoid noise
        if w*h < min_area:
            continue
        detections.append((x, y, w, h))
    return detections, th

# ---------------------------
# Derivatives via Savitzky-Golay (robust, smooth)
# ---------------------------


def compute_derivatives(times, xs, ys, fps, sg_window_sec=0.5, polyorder=3):
    dt = 1.0 / fps
    n = len(times)
    if n < 7:  # too short for smooth derivatives; fallback to finite differences
        vx = np.gradient(xs, dt); vy = np.gradient(ys, dt)
        ax = np.gradient(vx, dt); ay = np.gradient(vy, dt)
        jx = np.gradient(ax, dt); jy = np.gradient(ay, dt)
        sx = np.gradient(jx, dt); sy = np.gradient(jy, dt)
        return vx, vy, ax, ay, jx, jy, sx, sy

    # window length in samples (odd)
    w = max(5, int(round(sg_window_sec * fps)))
    if w % 2 == 0: w += 1
    w = min(w, n - (1 - n % 2))  # ensure <= n and stays odd
    if w < 5: w = 5
    # derivatives
    vx = savgol_filter(xs, window_length=w, polyorder=polyorder, deriv=1, delta=dt)
    vy = savgol_filter(ys, window_length=w, polyorder=polyorder, deriv=1, delta=dt)
    ax = savgol_filter(xs, window_length=w, polyorder=polyorder, deriv=2, delta=dt)
    ay = savgol_filter(ys, window_length=w, polyorder=polyorder, deriv=2, delta=dt)
    jx = savgol_filter(xs, window_length=w, polyorder=polyorder, deriv=3, delta=dt)
    jy = savgol_filter(ys, window_length=w, polyorder=polyorder, deriv=3, delta=dt)
    sx = savgol_filter(xs, window_length=w, polyorder=polyorder, deriv=4, delta=dt)
    sy = savgol_filter(ys, window_length=w, polyorder=polyorder, deriv=4, delta=dt)
    return vx, vy, ax, ay, jx, jy, sx, sy

# ---------------------------
# Feature engineering for clustering
# ---------------------------


def summarize_features(df_one):
    # df_one: rows of a single track, columns: time, x,y, vx,vy, ax,ay, jx,jy, sx,sy
    # magnitudes
    v = np.hypot(df_one.vx, df_one.vy)
    a = np.hypot(df_one.ax, df_one.ay)
    j = np.hypot(df_one.jx, df_one.jy)
    s = np.hypot(df_one.sx, df_one.sy)
    # path straightness: displacement / path length (0..1)
    dx = df_one.x.iloc[-1] - df_one.x.iloc[0]
    dy = df_one.y.iloc[-1] - df_one.y.iloc[0]
    disp = math.hypot(dx, dy)
    path = np.sum(np.hypot(np.diff(df_one.x), np.diff(df_one.y))) + 1e-9
    straightness = disp / path
    feats = {
        "v_mean": float(np.mean(v)),
        "v_std": float(np.std(v)),
        "a_mean": float(np.mean(a)),
        "a_std": float(np.std(a)),
        "j_mean": float(np.mean(j)),
        "j_std": float(np.std(j)),
        "s_mean": float(np.mean(s)),
        "s_std": float(np.std(s)),
        "straightness": float(straightness),
        "duration": float(df_one.time.iloc[-1] - df_one.time.iloc[0]),
        "n_frames": int(len(df_one)),
    }
    return feats


def build_feature_matrix(tracks_df, use_weights=(1.0, 1.0, 1.0, 1.0)):
    # use_weights = (w_v, w_a, w_j, w_s) to "incorporate derivatives" into norm
    rows = []
    ids = []
    meta = []
    for tid, grp in tracks_df.groupby("track_id"):
        feats = summarize_features(grp)
        wv, wa, wj, ws = use_weights
        vec = [
            wv * feats["v_mean"], wv * feats["v_std"],
            wa * feats["a_mean"], wa * feats["a_std"],
            wj * feats["j_mean"], wj * feats["j_std"],
            ws * feats["s_mean"], ws * feats["s_std"],
            feats["straightness"], feats["duration"]
        ]
        rows.append(vec)
        ids.append(tid)
        meta.append(feats)
    X = np.array(rows, dtype=np.float32) if rows else np.zeros((0,10), np.float32)
    return ids, X, pd.DataFrame(meta, index=ids)

# ---------------------------
# Main processing
# ---------------------------

def process_video(
    video_path,
    outdir,
    meters_per_pixel=None,    # e.g., 0.05 if known. Otherwise stays in pixels
    fps_override=None,
    min_track_frames=12,      # ignore very short tracks (~0.4s @30fps)
    bg_history=400, var_thresh=16, detect_dilate=2,
    min_area=400, max_age=12, dist_thresh=90.0,
    show_preview=False
):
    ensure_dir(outdir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps>0 else 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # background subtraction (no pretrained model)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=var_thresh, detectShadows=True)
    tracker = CentroidTracker(max_age=max_age, dist_thresh=dist_thresh)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(outdir, "annotated.mp4"), fourcc, fps, (W, H))

    frame_idx = 0
    masks_dir = os.path.join(outdir, "masks"); ensure_dir(masks_dir)

    while True:
        ret, frame = cap.read()
        if not ret: break
        detections, mask = detect_vehicles(frame, fgbg, min_area=min_area, dilate_iters=detect_dilate)
        tracker.update(detections, frame_idx)

        # draw
        vis = frame.copy()
        for t in tracker.tracks:
            # last bbox for drawing
            _, cx, cy, w, h = t.history[-1]
            x, y = int(cx - w/2), int(cy - h/2)
            cv2.rectangle(vis, (x, y), (x+int(w), y+int(h)), (0,255,0), 2)
            cv2.putText(vis, f"ID {t.id}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        out_video.write(vis)
        if frame_idx % 5 == 0:
            cv2.imwrite(os.path.join(masks_dir, f"mask_{frame_idx:05d}.png"), mask)

        if show_preview:
            cv2.imshow("annotated", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        frame_idx += 1

    cap.release(); out_video.release()
    if show_preview: cv2.destroyAllWindows()

    # Build per-track data
    rows = []
    for t in tracker.tracks:
        if len(t.history) < min_track_frames:
            continue
        for (fidx, cx, cy, w, h) in t.history:
            time = fidx / fps
            rows.append((t.id, fidx, time, cx, cy, w, h))

    traj_df = pd.DataFrame(rows, columns=["track_id", "frame", "time", "x", "y", "w", "h"])
    if traj_df.empty:
        print("No sufficiently long tracks found.")
        return None

    # Sort & compute derivatives per track
    traj_df = traj_df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    all_chunks = []
    for tid, grp in traj_df.groupby("track_id"):
        times = grp["time"].values
        xs = grp["x"].values
        ys = grp["y"].values

        # smooth positions slightly (pre-derivative)
        xs_s = moving_average_1d(xs, k=max(3, int(round(fps*0.2))))
        ys_s = moving_average_1d(ys, k=max(3, int(round(fps*0.2))))

        vx, vy, ax, ay, jx, jy, sx, sy = compute_derivatives(times, xs_s, ys_s, fps)

        df = grp.copy()
        df["x"]  = xs_s; df["y"]  = ys_s
        df["vx"] = vx;   df["vy"] = vy
        df["ax"] = ax;   df["ay"] = ay
        df["jx"] = jx;   df["jy"] = jy
        df["sx"] = sx;   df["sy"] = sy

        # pixel->meter conversion (if provided)
        if meters_per_pixel is not None:
            scale = float(meters_per_pixel)
            df[["x","y","w","h"]] *= scale
            df[["vx","vy"]]       *= scale
            df[["ax","ay"]]       *= scale
            df[["jx","jy"]]       *= scale
            df[["sx","sy"]]       *= scale

        all_chunks.append(df)

    tracks_df = pd.concat(all_chunks, ignore_index=True)
    # Save raw time series
    tracks_df.to_csv(os.path.join(outdir, "tracks_time_series.csv"), index=False)

    # Build features (derivative-aware)
    # You can tune these weights to emphasize derivatives in clustering
    weights = (1.0, 1.0, 0.5, 0.25)  # (speed, accel, jerk, jounce)
    ids, X, meta_df = build_feature_matrix(tracks_df, use_weights=weights)
    meta_df.index.name = "track_id"
    meta_df.to_csv(os.path.join(outdir, "track_summary_features.csv"))

    # KMeans clustering (choose k automatically by trying 2..4 and picking best inertia elbow-ish)
    labels_km = None
    if len(ids) >= 2:
        best_k, best_inertia, best_labels = None, float("inf"), None
        for k in range(2, min(5, len(ids))+1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            L = km.fit_predict(X)
            if km.inertia_ < best_inertia:
                best_inertia, best_k, best_labels = km.inertia_, k, L
        labels_km = best_labels
    else:
        labels_km = np.zeros(len(ids), dtype=int)

    # DBSCAN on standardized features (robust to #clusters)
    if len(ids) >= 2:
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        db = DBSCAN(eps=1.2, min_samples=2).fit(Xs)
        labels_db = db.labels_
    else:
        labels_db = np.zeros(len(ids), dtype=int)

    clustering_df = pd.DataFrame({
        "track_id": ids,
        "km_label": labels_km,
        "db_label": labels_db
    })
    clustering_df.to_csv(os.path.join(outdir, "clustering_labels.csv"), index=False)

    # Merge labels back to time series and save
    label_map_km = {tid: int(lbl) for tid, lbl in zip(ids, labels_km)}
    label_map_db = {tid: int(lbl) for tid, lbl in zip(ids, labels_db)}
    tracks_df["km_label"] = tracks_df["track_id"].map(label_map_km)
    tracks_df["db_label"] = tracks_df["track_id"].map(label_map_db)
    tracks_df.to_csv(os.path.join(outdir, "tracks_time_series_with_labels.csv"), index=False)

    # ---------------------------
    # Plots (per-track speed/accel; cluster scatter)
    # ---------------------------
    def safe_plot_speed_accel():
        for tid, grp in tracks_df.groupby("track_id"):
            t = grp["time"].values
            v = np.hypot(grp["vx"].values, grp["vy"].values)
            a = np.hypot(grp["ax"].values, grp["ay"].values)

            plt.figure()
            plt.plot(t, v, label="speed")
            plt.xlabel("time [s]"); plt.ylabel("speed" + (" [m/s]" if meters_per_pixel else " [px/s]"))
            plt.title(f"Track {tid} speed")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"speed_track_{tid}.png"))
            plt.close()

            plt.figure()
            plt.plot(t, a, label="accel")
            plt.xlabel("time [s]"); plt.ylabel("accel" + (" [m/s^2]" if meters_per_pixel else " [px/s^2]"))
            plt.title(f"Track {tid} acceleration")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"accel_track_{tid}.png"))
            plt.close()

    def safe_plot_cluster_space():
        if len(ids) >= 2:
            # 2D projection using two informative axes (mean speed vs straightness)
            ix_v = 0  # v_mean * wv
            ix_straight = 8
            plt.figure()
            for tid, vec, lab in zip(ids, X, labels_km):
                plt.scatter(vec[ix_v], vec[ix_straight])
                plt.annotate(str(tid), (vec[ix_v], vec[ix_straight]))
            plt.xlabel("weighted v_mean")
            plt.ylabel("straightness")
            plt.title("KMeans cluster space (annotated by track id)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "cluster_space_kmeans.png"))
            plt.close()

    safe_plot_speed_accel()
    safe_plot_cluster_space()

    # Save a simple run manifest
    manifest = {
        "video_path": video_path,
        "fps_used": fps,
        "meters_per_pixel": meters_per_pixel,
        "min_track_frames": min_track_frames,
        "tracker": {"max_age": max_age, "dist_thresh": dist_thresh},
        "detector": {"min_area": min_area, "bg_history": int(bg_history), "var_thresh": var_thresh},
        "weights": {"speed": 1.0, "accel": 1.0, "jerk": 0.5, "jounce": 0.25}
    }
    with open(os.path.join(outdir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Outputs written to: {outdir}")
    return {
        "tracks_df": tracks_df,
        "summary_df": meta_df,
        "clustering_df": clustering_df
    }

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Autobahn car motion extraction (library-based)")
    ap.add_argument("--video", required=True, help="Path to input video (e.g., traffic_10s.mp4)")
    ap.add_argument("--outdir", default="out_autobahn", help="Output directory")
    ap.add_argument("--meters-per-pixel", type=float, default=None,
                    help="If known, convert pixels to meters with this scale (e.g., 0.05)")
    ap.add_argument("--fps-override", type=float, default=None, help="Override FPS if metadata is wrong")
    ap.add_argument("--preview", action="store_true", help="Show live preview window (Esc to quit)")
    args = ap.parse_args()

    process_video(
        video_path=args.video,
        outdir=args.outdir,
        meters_per_pixel=args.meters_per_pixel,
        fps_override=args.fps_override,
        show_preview=args.preview
    )

if __name__ == "__main__":
    main()
