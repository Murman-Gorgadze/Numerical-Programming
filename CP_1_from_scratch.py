# autobahn_motion_pureclustering.py
# Same as your original pipeline, BUT:
#   - KMeans rewritten manually (NumPy only)
#   - DBSCAN rewritten manually (NumPy only)
#   - No scikit-learn used

import argparse, os, math, json, warnings
from collections import deque

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Utility
# ============================================================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def moving_average_1d(x, k=3):
    if k <= 1: return x
    if k % 2 == 0: k += 1
    return savgol_filter(x, window_length=k, polyorder=2, deriv=0)

# ============================================================
# Tracking
# ============================================================

class Track:
    _next_id = 1
    def __init__(self, cx, cy, bbox, frame_idx):
        self.id = Track._next_id; Track._next_id += 1
        self.history = []
        self.add(frame_idx, cx, cy, bbox)
        self.missed = 0

    def add(self, frame, cx, cy, bbox):
        x,y,w,h = bbox
        self.history.append((frame, float(cx), float(cy), float(w), float(h)))

    def last_centroid(self):
        _, cx, cy, _, _ = self.history[-1]
        return np.array([cx, cy], dtype=float)

class CentroidTracker:
    def __init__(self, max_age=15, dist_thresh=80.0):
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.tracks = []

    def update(self, detections, frame_idx):
        det_centroids = np.array([[x+w/2, y+h/2] for (x,y,w,h) in detections], float)
        track_centroids = np.array([t.last_centroid() for t in self.tracks], float) \
                          if self.tracks else np.zeros((0,2), float)

        if len(self.tracks) == 0:
            for (x,y,w,h) in detections:
                self.tracks.append(Track(x+w/2, y+h/2, (x,y,w,h), frame_idx))
            return

        if len(detections) == 0:
            for t in self.tracks:
                t.missed += 1
        else:
            C = cdist(track_centroids, det_centroids) if len(track_centroids) else np.empty((0,0))
            assigned_t = set()
            assigned_d = set()

            if C.size:
                r_idx, c_idx = linear_sum_assignment(C)
                for r,c in zip(r_idx, c_idx):
                    if C[r,c] <= self.dist_thresh:
                        t = self.tracks[r]
                        x,y,w,h = detections[c]
                        t.add(frame_idx, x+w/2, y+h/2, (x,y,w,h))
                        t.missed = 0
                        assigned_t.add(r)
                        assigned_d.add(c)

            # unmatched tracks
            for i,t in enumerate(self.tracks):
                if i not in assigned_t:
                    t.missed += 1

            # new tracks for mismatched detections
            for j,(x,y,w,h) in enumerate(detections):
                if j not in assigned_d:
                    self.tracks.append(Track(x+w/2, y+h/2, (x,y,w,h), frame_idx))

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

# ============================================================
# Detection (same as before)
# ============================================================

def detect_vehicles(frame, fgbg, min_area=400, dilate_iters=2):
    fgmask = fgbg.apply(frame)
    _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.dilate(th, kernel, iterations=dilate_iters)

    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: continue
        dets.append((x,y,w,h))

    return dets, th

# ============================================================
# Derivatives (SG)
# ============================================================

def compute_derivatives(times, xs, ys, fps, sg_window_sec=0.5):
    dt = 1/fps
    n = len(times)
    if n < 7:
        vx = np.gradient(xs, dt); vy = np.gradient(ys, dt)
        ax = np.gradient(vx, dt); ay = np.gradient(vy, dt)
        jx = np.gradient(ax, dt); jy = np.gradient(ay, dt)
        sx = np.gradient(jx, dt); sy = np.gradient(jy, dt)
        return vx,vy,ax,ay,jx,jy,sx,sy

    w = int(round(sg_window_sec*fps))
    if w%2==0: w+=1
    w = max(5, min(w, n-(1-n%2)))

    vx = savgol_filter(xs, w, 3, deriv=1, delta=dt)
    vy = savgol_filter(ys, w, 3, deriv=1, delta=dt)
    ax = savgol_filter(xs, w, 3, deriv=2, delta=dt)
    ay = savgol_filter(ys, w, 3, deriv=2, delta=dt)
    jx = savgol_filter(xs, w, 3, deriv=3, delta=dt)
    jy = savgol_filter(ys, w, 3, deriv=3, delta=dt)
    sx = savgol_filter(xs, w, 3, deriv=4, delta=dt)
    sy = savgol_filter(ys, w, 3, deriv=4, delta=dt)
    return vx,vy,ax,ay,jx,jy,sx,sy

# ============================================================
# Feature Engineering
# ============================================================

def summarize_features(df):
    v = np.hypot(df.vx, df.vy)
    a = np.hypot(df.ax, df.ay)
    j = np.hypot(df.jx, df.jy)
    s = np.hypot(df.sx, df.sy)

    dx = df.x.iloc[-1] - df.x.iloc[0]
    dy = df.y.iloc[-1] - df.y.iloc[0]
    disp = math.hypot(dx, dy)
    path = np.sum(np.hypot(np.diff(df.x), np.diff(df.y))) + 1e-9
    straight = disp/path

    return {
        "v_mean": float(v.mean()), "v_std": float(v.std()),
        "a_mean": float(a.mean()), "a_std": float(a.std()),
        "j_mean": float(j.mean()), "j_std": float(j.std()),
        "s_mean": float(s.mean()), "s_std": float(s.std()),
        "straightness": float(straight),
        "duration": float(df.time.iloc[-1] - df.time.iloc[0])
    }

def build_feature_matrix(df, weights=(1,1,1,1)):
    rows=[]; ids=[]; meta=[]
    for tid,grp in df.groupby("track_id"):
        f = summarize_features(grp)
        wv,wa,wj,ws = weights
        vec = [
            wv*f["v_mean"], wv*f["v_std"],
            wa*f["a_mean"], wa*f["a_std"],
            wj*f["j_mean"], wj*f["j_std"],
            ws*f["s_mean"], ws*f["s_std"],
            f["straightness"], f["duration"]
        ]
        rows.append(vec); ids.append(tid); meta.append(f)
    return ids, np.array(rows,float), pd.DataFrame(meta,index=ids)

# ============================================================
# MANUAL KMEANS IMPLEMENTATION
# ============================================================

def kmeans_manual(X, k, iters=100):
    n,d = X.shape
    idx = np.random.choice(n, k, replace=False)
    centers = X[idx]

    for _ in range(iters):
        dist = cdist(X, centers)
        labels = dist.argmin(axis=1)

        new_centers = np.vstack([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i]
                                 for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers

# ============================================================
# MANUAL DBSCAN IMPLEMENTATION
# ============================================================

def dbscan_manual(X, eps=1.2, min_pts=2):
    n = len(X)
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    distances = cdist(X, X)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(distances[i] <= eps)[0]
        if len(neighbors) < min_pts:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        queue = deque(neighbors)

        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                j_neighbors = np.where(distances[j] <= eps)[0]
                if len(j_neighbors) >= min_pts:
                    queue.extend(j_neighbors)
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels

# ============================================================
# Main video processing (same logic as your original)
# ============================================================

def process_video(video_path, outdir, meters_per_pixel=None, fps_override=None,
                  min_track_frames=12, bg_history=400, var_thresh=16,
                  detect_dilate=2, min_area=400, max_age=12, dist_thresh=90.0,
                  show_preview=False):

    ensure_dir(outdir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fgbg = cv2.createBackgroundSubtractorMOG2(history=bg_history,
                                             varThreshold=var_thresh,
                                             detectShadows=True)
    tracker = CentroidTracker(max_age=max_age, dist_thresh=dist_thresh)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(os.path.join(outdir,"annotated.mp4"),
                                fourcc, fps, (W,H))

    frame_idx=0
    masks_dir=os.path.join(outdir,"masks")
    ensure_dir(masks_dir)

    while True:
        ok, frame = cap.read()
        if not ok: break

        detections, mask = detect_vehicles(frame, fgbg, min_area, detect_dilate)
        tracker.update(detections, frame_idx)

        vis = frame.copy()
        for t in tracker.tracks:
            _, cx, cy, w, h = t.history[-1]
            x = int(cx - w/2)
            y = int(cy - h/2)
            cv2.rectangle(vis, (x,y), (x+int(w),y+int(h)), (0,255,0), 2)
            cv2.putText(vis, f"ID {t.id}", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        out_video.write(vis)
        if frame_idx % 5 == 0:
            cv2.imwrite(os.path.join(masks_dir,f"mask_{frame_idx:05d}.png"), mask)

        if show_preview:
            cv2.imshow("annotated", vis)
            if cv2.waitKey(1)&0xFF==27:
                break

        frame_idx += 1

    cap.release(); out_video.release()
    if show_preview: cv2.destroyAllWindows()

    # Build track dataframe
    rows=[]
    for t in tracker.tracks:
        if len(t.history) < min_track_frames:
            continue
        for (f,cx,cy,w,h) in t.history:
            rows.append((t.id, f, f/fps, cx, cy, w, h))

    df = pd.DataFrame(rows, columns=[
        "track_id","frame","time","x","y","w","h"
    ])
    if df.empty:
        print("No tracks long enough")
        return None

    df = df.sort_values(["track_id","frame"]).reset_index(drop=True)

    # Derivatives
    chunks=[]
    for tid,grp in df.groupby("track_id"):
        t=grp.time.values
        xs=grp.x.values; ys=grp.y.values

        xs_s = moving_average_1d(xs, k=5)
        ys_s = moving_average_1d(ys, k=5)

        vx,vy,ax,ay,jx,jy,sx,sy = compute_derivatives(t, xs_s, ys_s, fps)

        out = grp.copy()
        out["x"]=xs_s; out["y"]=ys_s
        out["vx"]=vx; out["vy"]=vy
        out["ax"]=ax; out["ay"]=ay
        out["jx"]=jx; out["jy"]=jy
        out["sx"]=sx; out["sy"]=sy

        if meters_per_pixel:
            scale=float(meters_per_pixel)
            for col in ["x","y","w","h","vx","vy","ax","ay","jx","jy","sx","sy"]:
                out[col] *= scale

        chunks.append(out)

    tracks_df = pd.concat(chunks).reset_index(drop=True)
    tracks_df.to_csv(os.path.join(outdir,"tracks_time_series.csv"), index=False)

    # Build features
    weights = (1,1,0.5,0.25)
    ids, X, meta = build_feature_matrix(tracks_df, weights)
    meta.index.name = "track_id"
    meta.to_csv(os.path.join(outdir,"track_summary_features.csv"))

    # Manual KMeans (try k=2..4)
    if len(ids) >= 2:
        best_labels=None
        best_inertia=float("inf")

        for k in range(2, min(5,len(ids))+1):
            labels,_ = kmeans_manual(X, k)
            inertia = np.sum((X - X[labels])**2)
            if inertia < best_inertia:
                best_inertia=inertia
                best_labels=labels
        labels_km = best_labels
    else:
        labels_km = np.zeros(len(ids), int)

    # Manual DBSCAN
    if len(ids) >= 2:
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-9)
        labels_db = dbscan_manual(Xs, eps=1.2, min_pts=2)
    else:
        labels_db = np.zeros(len(ids), int)

    # Save final label table
    lab_df = pd.DataFrame({
        "track_id": ids,
        "km_label": labels_km,
        "db_label": labels_db
    })
    lab_df.to_csv(os.path.join(outdir,"clustering_labels.csv"), index=False)

    # Merge into time series
    km_map = dict(zip(ids, labels_km))
    db_map = dict(zip(ids, labels_db))

    tracks_df["km_label"] = tracks_df.track_id.map(km_map)
    tracks_df["db_label"] = tracks_df.track_id.map(db_map)
    tracks_df.to_csv(os.path.join(outdir,"tracks_time_series_with_labels.csv"), index=False)

    print("[DONE] All outputs stored in:", outdir)
    return {
        "tracks_df": tracks_df,
        "summary_df": meta,
        "labels": lab_df
    }

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="out_pureclust")
    ap.add_argument("--meters-per-pixel", type=float, default=None)
    ap.add_argument("--fps-override", type=float, default=None)
    ap.add_argument("--preview", action="store_true")
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
