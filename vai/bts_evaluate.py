"""
BTS Gaussian Visibility / Activation Score Protocol
====================================================

Muc tieu
--------
Voi moi Gaussian trong point_cloud.ply, tinh:
  - orbit_visible_count[i, o] : so anh crop trong orbit o ma Gaussian i visible
  - orbit_ratio[i, o]         : ty le visible trong orbit o
  - n_orbits_stable[i]        : so orbit ma Gaussian i on dinh (ratio >= tau)
  - activation_score[i]       : trung binh orbit_ratio tren cac orbit co xuat hien

Sau do phan loai BTS / boundary / background bang Otsu threshold,
va xuat: CSV score, PNG scatter 3D, heatmap overlay tren tung anh crop.

Dependencies
------------
pip install numpy pillow plyfile torch scikit-image pandas matplotlib

Usage
-----
Doc truc tiep tu COLMAP bin (KHONG can CSV trung gian):
    python bts_gaussian_visibility_protocol.py  --ply point_cloud/HCM0204/point_cloud.ply --sparse-dir output/HCM0204/train/sparse/0 --csv output/HCM0204/test/test_poses.csv --crop-images-dir output/HCM0204/train/images  --outdir save_visibility/HCM0204 --n-orbits 10

Hoac dung CSV neu ban da co san (tuong thich nguoc):
    python bts_gaussian_visibility_protocol.py \
        --ply point_cloud.ply --csv test_poses.csv --outdir save_visibility

Co the ket hop CA HAI cung luc (vd train tu bin + test tu csv):
    python bts_gaussian_visibility_protocol.py \
        --ply point_cloud/scene/point_cloud.ply \
        --sparse-dir output/scene/train/sparse/0 \
        --csv output/scene/test/test_poses.csv \
        --outdir save_visibility

Ghi chu
-------
- Dung --sparse-dir tro toi thu muc chua cameras.bin + images.bin CUA ANH
  DA CROP (vd output/<scene>/train/sparse/0), khong phai thu muc colmap/ goc,
  vi width/height/cx/cy phai khop voi anh crop dang dung de rasterize.
- qw,qx,qy,qz,tx,ty,tz trong images.bin cua COLMAP la world->camera (w2c),
  khop dung --pose-convention w2c (mac dinh).
- Cach doc bin (struct format, ten field, thu tu byte) duoc dong bo voi
  script fix_cropped_colmap.py de dam bao khong lech offset.
- up_axis mac dinh la 'z'. Doi neu scene ban dung 'y' la up.
- Neu vua truyen --sparse-dir vua truyen --csv, script se doc CA HAI va
  gop (concat) danh sach view lai voi nhau truoc khi xu ly tiep. Neu co
  image_name trung nhau giua 2 nguon, view tu --sparse-dir se duoc giu,
  view trung ten tu --csv se bi bo qua (va in canh bao).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = e

try:
    from plyfile import PlyData
except Exception as e:  # pragma: no cover
    PlyData = None
    _PLY_IMPORT_ERROR = e

from skimage.filters import threshold_otsu

SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Data loading (giong file goc)
# ----------------------------------------------------------------------

@dataclass
class GaussianCloud:
    xyz: np.ndarray
    scales: np.ndarray | None
    quats: np.ndarray | None


def quat_to_rotmat_torch(q: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    q = q.to(torch.float32)
    if order == "xyzw":
        x, y, z, w = q.unbind(dim=-1)
    elif order == "wxyz":
        w, x, y, z = q.unbind(dim=-1)
    else:
        raise ValueError(f"Unsupported quat order: {order}")

    n = torch.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.empty(q.shape[:-1] + (3, 3), device=q.device, dtype=torch.float32)
    R[..., 0, 0] = ww + xx - yy - zz
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = ww - xx + yy - zz
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = ww - xx - yy + zz
    return R


def load_gaussian_cloud(ply_path: Path) -> GaussianCloud:
    if PlyData is None:
        raise ImportError(f"plyfile required: {_PLY_IMPORT_ERROR}")
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    names = v.dtype.names or ()

    def get_cols(prefix: str, n: int):
        cols = [f"{prefix}_{i}" for i in range(n)]
        if all(c in names for c in cols):
            return np.stack([np.asarray(v[c], dtype=np.float64) for c in cols], axis=1)
        return None

    xyz = np.stack([np.asarray(v[c], dtype=np.float64) for c in ["x", "y", "z"]], axis=1)
    scales = get_cols("scale", 3)
    quats = get_cols("rot", 4)
    return GaussianCloud(xyz=xyz, scales=scales, quats=quats)


def read_pose_csv(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                    "fx", "fy", "cx", "cy", "width", "height"]
        for k in required:
            if k not in reader.fieldnames:
                raise ValueError(f"CSV missing column '{k}'. Found: {reader.fieldnames}")
        for row in reader:
            rows.append({
                "image_name": row["image_name"],
                "qw": float(row["qw"]), "qx": float(row["qx"]),
                "qy": float(row["qy"]), "qz": float(row["qz"]),
                "tx": float(row["tx"]), "ty": float(row["ty"]), "tz": float(row["tz"]),
                "fx": float(row["fx"]), "fy": float(row["fy"]),
                "cx": float(row["cx"]), "cy": float(row["cy"]),
                "width": int(float(row["width"])), "height": int(float(row["height"])),
            })
    return rows


# ----------------------------------------------------------------------
# Doc truc tiep COLMAP cameras.bin / images.bin
# (dong bo cach doc struct voi fix_cropped_colmap.py de tranh lech offset)
# ----------------------------------------------------------------------

import struct as _struct

# model_id -> (ten, so param, chi so cx trong params, chi so cy trong params)
_COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, 1, 2),
    1: ("PINHOLE", 4, 2, 3),
    2: ("SIMPLE_RADIAL", 4, 1, 2),
    3: ("RADIAL", 5, 1, 2),
    4: ("OPENCV", 8, 2, 3),
    5: ("OPENCV_FISHEYE", 8, 2, 3),
    6: ("FULL_OPENCV", 12, 2, 3),
    7: ("FOV", 5, 2, 3),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, 1, 2),
    9: ("RADIAL_FISHEYE", 5, 1, 2),
    10: ("THIN_PRISM_FISHEYE", 12, 2, 3),
}


def _read_cameras_bin(path: Path) -> dict:
    """Tra ve dict camera_id -> dict(model_id, model_name, width, height, params)."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = _struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id = _struct.unpack("<ii", f.read(8))
            width, height = _struct.unpack("<QQ", f.read(16))
            if model_id not in _COLMAP_CAMERA_MODELS:
                raise ValueError(f"Camera model_id={model_id} khong duoc ho tro.")
            model_name, num_params, cx_idx, cy_idx = _COLMAP_CAMERA_MODELS[model_id]
            params = list(_struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[camera_id] = {
                "model_id": model_id, "model_name": model_name,
                "width": width, "height": height, "params": params,
                "cx_idx": cx_idx, "cy_idx": cy_idx,
            }
    return cameras


def _read_images_bin(path: Path) -> dict:
    """Tra ve dict image_id -> dict(qvec, tvec, camera_id, name)."""
    images = {}
    with open(path, "rb") as f:
        num_reg_images = _struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg_images):
            image_id = _struct.unpack("<i", f.read(4))[0]
            qvec = _struct.unpack("<dddd", f.read(32))  # qw, qx, qy, qz
            tvec = _struct.unpack("<ddd", f.read(24))   # tx, ty, tz
            camera_id = _struct.unpack("<i", f.read(4))[0]

            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                if ch == b"":
                    raise EOFError("images.bin ket thuc bat ngo khi doc ten anh.")
                name_chars.append(ch)
            name = b"".join(name_chars).decode("utf-8")

            num_points2d = _struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2d)  # bo qua x,y,point3D_id (khong can cho protocol nay)

            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name}
    return images


def _camera_params_to_fx_fy_cx_cy(cam: dict):
    p = cam["params"]
    model_name = cam["model_name"]
    if model_name == "SIMPLE_PINHOLE":
        f, cx, cy = p[0], p[1], p[2]
        return f, f, cx, cy
    if model_name == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if model_name in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        return f, f, cx, cy
    if model_name in ("RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        return f, f, cx, cy
    if model_name in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"):
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"Camera model chua ho tro: {model_name}")


def read_pose_rows_from_colmap_bin(sparse_dir: Path) -> List[dict]:
    """
    Doc cameras.bin + images.bin trong sparse_dir (phai la thu muc cua ANH
    DA CROP, vd output/<scene>/train/sparse/0), tra ve list dict cung format
    voi read_pose_csv() de dung chung cho phan con lai cua script.
    """
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    if not cameras_path.exists():
        raise FileNotFoundError(f"Khong tim thay: {cameras_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Khong tim thay: {images_path}")

    cameras = _read_cameras_bin(cameras_path)
    images = _read_images_bin(images_path)
    print(f"[INFO] doc bin: {len(cameras)} cameras, {len(images)} images tu {sparse_dir}")

    model_names = {c["model_name"] for c in cameras.values()}
    distortion_models = {"OPENCV", "RADIAL", "SIMPLE_RADIAL", "OPENCV_FISHEYE",
                          "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "FULL_OPENCV",
                          "THIN_PRISM_FISHEYE"}
    if model_names & distortion_models:
        print(f"[WARN] camera model co distortion ({model_names & distortion_models}) "
              f"nhung script chi lay fx,fy,cx,cy, bo qua he so distortion. "
              f"Neu anh crop chua undistort thi mu_x,mu_y se lech.")

    rows = []
    for img in images.values():
        cam = cameras.get(img["camera_id"])
        if cam is None:
            print(f"[WARN] image '{img['name']}' tham chieu camera_id={img['camera_id']} "
                  f"khong ton tai, bo qua.")
            continue
        fx, fy, cx, cy = _camera_params_to_fx_fy_cx_cy(cam)
        qw, qx, qy, qz = img["qvec"]
        tx, ty, tz = img["tvec"]
        rows.append({
            "image_name": img["name"],
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": cam["width"], "height": cam["height"],
        })

    rows.sort(key=lambda r: r["image_name"])
    return rows


def world_to_camera_matrix(qw, qx, qy, qz, tx, ty, tz, convention: str = "w2c"):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    R = quat_to_rotmat_torch(torch.tensor(q[None, :]), order="wxyz")[0].cpu().numpy()
    t = np.array([tx, ty, tz], dtype=np.float64)
    if convention == "w2c":
        return R, t
    if convention == "c2w":
        R_inv = R.T
        t_inv = -R_inv @ t
        return R_inv, t_inv
    raise ValueError(f"Unsupported pose convention: {convention}")


def camera_center_world(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    # Rcw, tcw la world->camera. Camera center trong world: C = -R^T t
    return -Rcw.T @ tcw


# ----------------------------------------------------------------------
# Fit truc quy dao (KHONG gia dinh truc = world axis co dinh)
# ----------------------------------------------------------------------

def fit_orbit_axis_from_camera_centers(cam_centers: np.ndarray,
                                        world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
                                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Uoc luong TRUC quy dao bay (khong gia dinh trung voi 1 truc world co
    dinh nhu x/y/z), bang cach minimize variance cua ban kinh (khoang cach
    tu camera toi truc) qua Nelder-Mead — cung cach tiep can voi
    estimate_bts_cylinder.py (fit_cylinder_axis_from_camera_centers).

    Scene COLMAP thuong KHONG duoc gravity-align, nen cot/toa nha (va do
    do quy dao bay quanh no) co the nghieng so voi world Z/X/Y. Dung truc
    that su cho ket qua orbit-splitting on dinh hon nhieu so voi ep truc
    theo 1 chieu world co dinh.

    Tra ve: (axis unit vector, diem tren truc p0 gan trong tam camera nhat,
    goc lech so voi world_up tinh bang do).
    Neu khong co scipy, fallback ve world_up (in canh bao).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[WARN] scipy khong san co -> khong the fit truc quy dao tu dong, "
              "fallback ve up_axis co dinh do nguoi dung khai bao. "
              "Cai scipy (pip install scipy) de fit truc chinh xac hon.")
        centroid = cam_centers.mean(axis=0)
        return world_up / np.linalg.norm(world_up), centroid, 0.0

    world_up = world_up / np.linalg.norm(world_up)
    centroid = cam_centers.mean(axis=0)

    # --- loc so bo outlier (camera lech qua xa centroid) truoc khi fit truc ---
    # Fit truc bang variance/MAD cua ban kinh la ky thuat cuc nhay voi outlier:
    # chi vai camera lech xa (vd test-view rat xa scene) cung du "keo" truc
    # lech han di de giam sai so cho rieng chung no, lam sai toan bo huong
    # truc that. Loc so bo bang khoang cach toi centroid (MAD-based) truoc.
    dist_to_centroid = np.linalg.norm(cam_centers - centroid.reshape(1, 3), axis=1)
    med_d = np.median(dist_to_centroid)
    mad_d = np.median(np.abs(dist_to_centroid - med_d)) + 1e-9
    robust_std_d = 1.4826 * mad_d
    inlier_mask = dist_to_centroid <= med_d + 5.0 * max(robust_std_d, 1e-9)
    n_outliers = int((~inlier_mask).sum())
    fit_points = cam_centers[inlier_mask] if inlier_mask.sum() >= 3 else cam_centers
    if n_outliers > 0 and inlier_mask.sum() >= 3:
        print(f"[INFO] fit_orbit_axis_from_camera_centers: loai {n_outliers} camera "
              f"lech qua xa centroid truoc khi fit truc (tranh truc bi keo lech).")
    centroid = fit_points.mean(axis=0)

    def axis_from_angles(theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])

    def radii_for_axis(n, pts):
        n = n / np.linalg.norm(n)
        p0 = centroid - np.dot(centroid, n) * n
        rel = pts - p0.reshape(1, 3)
        proj = rel - np.outer(rel @ n, n)
        r = np.linalg.norm(proj, axis=1)
        return r, p0

    def cost(params):
        theta, phi = params
        n = axis_from_angles(theta, phi)
        r, _ = radii_for_axis(n, fit_points)
        # Dung robust spread (MAD) thay vi variance (L2): variance rat nhay
        # voi vai outlier con sot, MAD gan nhu khong bi anh huong.
        med_r = np.median(r)
        return float(np.median(np.abs(r - med_r)))

    theta0 = np.arccos(np.clip(world_up[2], -1.0, 1.0))
    phi0 = np.arctan2(world_up[1], world_up[0])
    res = minimize(cost, x0=[theta0, phi0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})

    theta, phi = res.x
    axis = axis_from_angles(theta, phi)
    axis /= np.linalg.norm(axis)
    if np.dot(axis, world_up) < 0:
        axis = -axis

    _, p0 = radii_for_axis(axis, fit_points)
    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(axis, world_up), -1.0, 1.0))))
    return axis, p0, angle_deg


def make_orthonormal_basis(axis: np.ndarray):
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


# ----------------------------------------------------------------------
# Buoc 1: Chia camera thanh orbits theo do cao doc theo truc quy dao
# ----------------------------------------------------------------------

def _kmeans_1d(values: np.ndarray, k: int, n_iter: int = 100, n_init: int = 10,
                seed: int = 0) -> np.ndarray:
    """
    KMeans don gian tren du lieu 1 chieu (khong can sklearn). Chay n_init
    lan voi khoi tao khac nhau (theo quantile + jitter nho), giu lan chay
    co inertia thap nhat, de tranh local minimum xau. Robust voi outlier
    hon nhieu so voi gap-threshold vi outlier chi bi gan vao centroid gan
    nhat, khong tao cum rieng.
    """
    rng = np.random.default_rng(seed)
    v = values.reshape(-1, 1)
    best_labels, best_inertia = None, np.inf

    for init_i in range(n_init):
        if init_i == 0:
            # khoi tao theo quantile (deterministic, on dinh)
            qs = np.linspace(0, 1, k + 2)[1:-1]
            centers = np.quantile(values, qs)
        else:
            centers = rng.choice(values, size=k, replace=False).astype(np.float64)
        centers = np.sort(centers)

        for _ in range(n_iter):
            dist = np.abs(v - centers[None, :])
            labels = np.argmin(dist, axis=1)
            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centers[c] = values[mask].mean()
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        dist = np.abs(v - centers[None, :])
        labels = np.argmin(dist, axis=1)
        inertia = float(((values - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    # remap label id theo thu tu height tang dan de orbit_id co y nghia
    # (orbit 0 = thap nhat, orbit k-1 = cao nhat)
    order = np.argsort([values[best_labels == c].mean() if (best_labels == c).any() else np.inf
                         for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in best_labels], dtype=int)


def assign_orbits(cam_centers: np.ndarray,
                   axis: np.ndarray, axis_point: np.ndarray,
                   n_orbits: int | None = None,
                   height_gap_std_mult: float = 1.0,
                   min_orbit_size: int = 3) -> np.ndarray:
    """
    cam_centers: (N,3) camera centers trong world.
    axis: (3,) vector don vi cua truc quy dao (tu fit_orbit_axis_from_camera_centers,
          KHONG con gia dinh la mot truc world co dinh nua).
    axis_point: (3,) 1 diem tren truc, dung lam goc de chieu.
    Tra ve orbit_id (N,) int.

    Neu n_orbits duoc chi dinh (khuyen nghi: nhin so vong xoan trong scatter
    3D roi truyen --n-orbits): dung 1D KMeans tren toa do doc truc, k=n_orbits.
    Cach nay ON DINH va ROBUST voi outlier (vd test view lech xa scene) vi
    outlier chi bi gan vao centroid gan nhat, khong tao thanh orbit rac.

    Neu khong chi dinh n_orbits: fallback ve phuong phap gap-based tu dong
    (dung nguong median + MAD, robust hon mean+std voi outlier), sau do
    gop cac cum qua nho (< min_orbit_size anh, thuong la outlier) vao cum
    lang gieng gan nhat. Phuong phap nay it on dinh hon KMeans khi so
    luong anh/orbit khong deu hoac nhieu noise, nen chi dung khi khong
    biet truoc so orbit.
    """
    axis = axis / np.linalg.norm(axis)
    rel = cam_centers - axis_point.reshape(1, 3)
    heights = rel @ axis                      # toa do doc truc quy dao
    perp = rel - np.outer(heights, axis)      # phan vuong goc voi truc
    radius = np.linalg.norm(perp, axis=1)     # ban kinh quanh truc

    if n_orbits is not None:
        if n_orbits < 1:
            raise ValueError("n_orbits phai >= 1")
        if n_orbits == 1 or len(heights) <= n_orbits:
            return np.zeros(len(heights), dtype=int)

        # --- loc outlier bang MAD tren height va radius truoc khi fit KMeans ---
        def _mad_outlier_mask(x: np.ndarray, mult: float = 5.0) -> np.ndarray:
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-9
            robust_std = 1.4826 * mad
            return np.abs(x - med) <= mult * max(robust_std, 1e-9)

        inlier_mask = _mad_outlier_mask(heights) & _mad_outlier_mask(radius)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < n_orbits or n_inliers == len(heights):
            # khong du inlier de fit rieng, hoac khong co outlier -> fit tren tat ca
            return _kmeans_1d(heights, k=n_orbits)

        n_outliers = len(heights) - n_inliers
        print(f"[INFO] assign_orbits: phat hien {n_outliers} camera outlier "
              f"(lech manh ca height va radius quanh truc quy dao), "
              f"loai khoi buoc fit KMeans, se gan vao orbit gan nhat sau.")

        # fit KMeans (theo height doc truc) chi tren inlier -> tam cum khong bi outlier keo lech
        inlier_orbit_id = _kmeans_1d(heights[inlier_mask], k=n_orbits)
        centers = np.array([
            heights[inlier_mask][inlier_orbit_id == c].mean() for c in range(n_orbits)
        ])

        orbit_id = np.zeros(len(heights), dtype=int)
        orbit_id[inlier_mask] = inlier_orbit_id
        outlier_idx = np.nonzero(~inlier_mask)[0]
        outlier_heights = heights[outlier_idx]
        nearest = np.argmin(np.abs(outlier_heights[:, None] - centers[None, :]), axis=1)
        orbit_id[outlier_idx] = nearest
        return orbit_id

    order = np.argsort(heights)
    h_sorted = heights[order]
    gaps = np.diff(h_sorted)

    if len(gaps) == 0:
        return np.zeros(len(heights), dtype=int)

    median_gap = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - median_gap))) + 1e-12
    # he so 1.4826 de MAD xap xi std duoi gia dinh phan phoi gan chuan
    robust_std = 1.4826 * mad
    thresh = median_gap + height_gap_std_mult * robust_std
    split_after = np.where(gaps > thresh)[0]  # index trong h_sorted truoc cho split

    orbit_id_sorted = np.zeros(len(h_sorted), dtype=int)
    cur = 0
    split_set = set(split_after.tolist())
    for i in range(1, len(h_sorted)):
        if (i - 1) in split_set:
            cur += 1
        orbit_id_sorted[i] = cur

    # --- gop cac cum qua nho (outlier) vao cum lang gieng gan nhat ---
    n_clusters = int(orbit_id_sorted.max()) + 1
    if n_clusters > 1 and min_orbit_size > 1:
        sizes = np.bincount(orbit_id_sorted, minlength=n_clusters)
        cluster_mean_h = np.array([
            h_sorted[orbit_id_sorted == c].mean() for c in range(n_clusters)
        ])
        # sap xep lai remap de cac cum lon duoc giu id rieng, cum nho duoc
        # gop dan vao cum lang gieng (lap lai cho toi khi khong con cum nho,
        # tru khi chi con dung 1 cum)
        remap = np.arange(n_clusters)  # cluster_old -> cluster_target (truoc remap cuoi)
        active = list(range(n_clusters))
        changed = True
        while changed and len(active) > 1:
            changed = False
            sizes_active = {c: sizes[c] for c in active}
            small = [c for c in active if sizes_active[c] < min_orbit_size]
            if not small:
                break
            # gop cum nho nhat truoc
            c = min(small, key=lambda cc: sizes_active[cc])
            others = [o for o in active if o != c]
            nearest = min(others, key=lambda o: abs(cluster_mean_h[o] - cluster_mean_h[c]))
            # gop c vao nearest: cap nhat sizes, mean, remap
            total = sizes[c] + sizes[nearest]
            cluster_mean_h[nearest] = (
                cluster_mean_h[nearest] * sizes[nearest] + cluster_mean_h[c] * sizes[c]
            ) / total
            sizes[nearest] = total
            sizes[c] = 0
            remap[remap == c] = nearest
            active.remove(c)
            changed = True

        # nen lai id ve 0..K-1 lien tuc
        final_ids = sorted(set(remap[c] for c in range(n_clusters)))
        compress = {old: new for new, old in enumerate(final_ids)}
        orbit_id_sorted = np.array([compress[remap[c]] for c in orbit_id_sorted], dtype=int)

    orbit_id = np.zeros(len(heights), dtype=int)
    orbit_id[order] = orbit_id_sorted
    return orbit_id


# ----------------------------------------------------------------------
# Buoc 2+3: Rasterize + occlusion (z-buffer) -> visible mask per Gaussian
# ----------------------------------------------------------------------

def make_camera_covariance_torch(xyz_world, scales_raw, quats_raw, Rcw, tcw,
                                   fx, fy, scale_mode="auto", quat_order="wxyz",
                                   epsilon=1e-8):
    device = xyz_world.device
    dtype = xyz_world.dtype
    N = xyz_world.shape[0]
    means_cam = xyz_world @ Rcw.T + tcw[None, :]

    if scales_raw is None:
        sigma = torch.full((N, 3), 0.01, device=device, dtype=dtype)
    else:
        if scale_mode == "auto":
            med = float(torch.median(scales_raw).item())
            use_log = med < 0.0
        elif scale_mode == "log":
            use_log = True
        else:
            use_log = False
        sigma = torch.exp(scales_raw) if use_log else scales_raw.clone()
        sigma = torch.clamp(sigma, min=1e-9)

    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    inv_z = 1.0 / torch.clamp(z, min=epsilon)
    return means_cam, sigma, x, y, z, inv_z


@torch.no_grad()
def compute_visible_mask_for_view(
    xyz_world: torch.Tensor,       # (N,3)
    scales_raw: torch.Tensor | None,
    Rcw: torch.Tensor, tcw: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    depth_tolerance_scale: float = 3.0,
    default_tolerance: float = 0.05,
    near: float = 1e-4,
) -> np.ndarray:
    """
    Tra ve boolean mask (N,) : Gaussian nao visible (trong frustum + khong bi occlude)
    trong view nay.

    Occlusion test: group theo pixel nguyen (round(mu_x), round(mu_y)),
    tim z_min tai moi pixel, giu lai Gaussian co z <= z_min + tolerance.
    Tolerance ~ vai lan mean scale cua Gaussian (cho phep nhieu Gaussian
    cung mot lop ket cau mong nhu giao ang-ten).
    """
    device = xyz_world.device
    N = xyz_world.shape[0]

    means_cam = xyz_world @ Rcw.T + tcw[None, :]
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]

    valid = z > near
    inv_z = 1.0 / torch.clamp(z, min=near)
    mu_x = fx * x * inv_z + cx
    mu_y = fy * y * inv_z + cy

    in_frustum = (mu_x >= 0) & (mu_x < width) & (mu_y >= 0) & (mu_y < height)
    candidate = valid & in_frustum & torch.isfinite(mu_x) & torch.isfinite(mu_y)

    visible = torch.zeros(N, dtype=torch.bool, device=device)
    idxs = torch.nonzero(candidate, as_tuple=False).squeeze(1)
    if idxs.numel() == 0:
        return visible.cpu().numpy()

    px = torch.clamp(mu_x[idxs].round().long(), 0, width - 1)
    py = torch.clamp(mu_y[idxs].round().long(), 0, height - 1)
    pz = z[idxs]
    pixel_flat = py * width + px

    # z-buffer: tim z_min tai moi pixel bang scatter_reduce
    n_pixels = width * height
    zbuf = torch.full((n_pixels,), float("inf"), device=device, dtype=pz.dtype)
    zbuf.scatter_reduce_(0, pixel_flat, pz, reduce="amin", include_self=True)

    if scales_raw is not None:
        mean_scale = float(scales_raw[idxs].abs().mean().item())
        # scale co the o dang log; neu qua nho/am thi fallback default
        tol = mean_scale * depth_tolerance_scale
        if not np.isfinite(tol) or tol <= 0:
            tol = default_tolerance
    else:
        tol = default_tolerance

    z_min_at_pixel = zbuf[pixel_flat]
    keep = pz <= (z_min_at_pixel + tol)

    visible[idxs[keep]] = True
    return visible.cpu().numpy()


# ----------------------------------------------------------------------
# Gop nhieu nguon view (bin + csv, ...) lai voi nhau
# ----------------------------------------------------------------------

def merge_pose_rows(*row_lists: List[dict]) -> List[dict]:
    """
    Gop nhieu list rows (tu --sparse-dir, --csv, ...) lai thanh 1 list duy nhat.
    Neu trung image_name giua cac nguon, giu row xuat hien truoc (theo thu tu
    truyen vao *row_lists) va bo qua + canh bao cac row trung sau do.
    """
    merged: List[dict] = []
    seen: dict[str, int] = {}  # image_name -> index nguon da lay
    for src_idx, rows in enumerate(row_lists):
        for row in rows:
            name = row["image_name"]
            if name in seen:
                print(f"[WARN] image_name trung lap '{name}' (nguon #{src_idx}), "
                      f"da lay tu nguon #{seen[name]} truoc do, bo qua row nay.")
                continue
            seen[name] = src_idx
            merged.append(row)
    merged.sort(key=lambda r: r["image_name"])
    return merged


# ----------------------------------------------------------------------
# Main protocol
# ----------------------------------------------------------------------

def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (SCRIPT_DIR / pp)


def main():
    if torch is None:
        raise ImportError(f"torch required: {_TORCH_IMPORT_ERROR}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=str, required=True)
    ap.add_argument("--csv", type=str, default=None,
                     help="CSV cua camera ANH CROP. Co the dung DOC LAP hoac KET HOP "
                          "cung luc voi --sparse-dir (2 nguon se duoc gop lai).")
    ap.add_argument("--sparse-dir", type=str, default=None,
                     help="Thu muc chua cameras.bin + images.bin CUA ANH CROP "
                          "(vd output/scene/train/sparse/0). Co the dung KET HOP "
                          "cung luc voi --csv (2 nguon se duoc gop lai).")
    ap.add_argument("--crop-images-dir", type=str, default=None,
                     help="Thu muc anh crop de ve heatmap overlay (tuy chon)")
    ap.add_argument("--outdir", type=str, default="save_visibility")
    ap.add_argument("--pose-convention", type=str, default="w2c", choices=["w2c", "c2w"])
    ap.add_argument("--scale-mode", type=str, default="auto", choices=["auto", "log", "linear"])
    ap.add_argument("--up-axis", type=str, default="z", choices=["x", "y", "z"],
                     help="World-up dung lam DIEM KHOI TAO cho thuat toan fit truc quy dao "
                          "(scipy Nelder-Mead, xem fit_orbit_axis_from_camera_centers) — KHONG "
                          "con bi ep cung. Neu scene nghieng nhieu, doi gia tri nay giup thuat "
                          "toan hoi tu dung huong hon. Dung --force-up-axis de ep cung nhu cu.")
    ap.add_argument("--force-up-axis", action="store_true",
                     help="Bo qua buoc fit truc tu dong, ep truc quy dao = world axis "
                          "chi dinh boi --up-axis (hanh vi cu, chi nen dung khi biet chac "
                          "scene da gravity-align).")
    ap.add_argument("--tau", type=float, default=0.5, help="Nguong orbit_ratio de tinh n_orbits_stable")
    ap.add_argument("--n-orbits", type=int, default=None,
                     help="So orbit mong muon (dem tu scatter 3D cua camera, vd bao nhieu "
                          "vong xoan). NEU TRUYEN: dung 1D KMeans robust theo height, "
                          "KHUYEN NGHI dung tham so nay de tranh loi phan orbit sai do outlier. "
                          "Neu khong truyen: fallback ve gap-based tu dong (kem on dinh hon).")
    ap.add_argument("--depth-tolerance-scale", type=float, default=3.0)
    ap.add_argument("--height-gap-std-mult", type=float, default=1.0)
    ap.add_argument("--min-orbit-size", type=int, default=3,
                     help="Cum co it hon so anh nay se bi coi la outlier va gop vao "
                          "orbit lang gieng gan nhat theo height (thay vi thanh orbit rieng)")
    ap.add_argument("--no-heatmap", action="store_true", help="Bo qua ve heatmap overlay tren tung anh")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    ply_path = resolve_path(args.ply).resolve()
    outdir = resolve_path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.sparse_dir and not args.csv:
        raise ValueError("Can truyen --sparse-dir (doc bin truc tiep) va/hoac --csv.")

    cloud = load_gaussian_cloud(ply_path)

    # Doc tung nguon view neu duoc truyen, roi gop lai. Uu tien --sparse-dir
    # khi trung image_name (xem merge_pose_rows()).
    row_sources: List[List[dict]] = []
    if args.sparse_dir:
        sparse_dir = resolve_path(args.sparse_dir).resolve()
        row_sources.append(read_pose_rows_from_colmap_bin(sparse_dir))
    if args.csv:
        csv_path = resolve_path(args.csv).resolve()
        csv_rows = read_pose_csv(csv_path)
        print(f"[INFO] doc csv: {len(csv_rows)} images tu {csv_path}")
        row_sources.append(csv_rows)

    rows = merge_pose_rows(*row_sources)
    print(f"[INFO] tong so view sau khi gop: {len(rows)}")

    N = cloud.xyz.shape[0]
    print(f"[INFO] gaussians: {N}, images: {len(rows)}")

    xyz_t = torch.tensor(cloud.xyz, device=device, dtype=torch.float32)
    scales_t = None if cloud.scales is None else torch.tensor(cloud.scales, device=device, dtype=torch.float32)
    if cloud.scales is not None and args.scale_mode == "auto":
        med = float(np.median(cloud.scales))
        if med < 0.0:
            scales_lin_t = torch.exp(scales_t)
        else:
            scales_lin_t = scales_t
    elif cloud.scales is not None and args.scale_mode == "log":
        scales_lin_t = torch.exp(scales_t)
    else:
        scales_lin_t = scales_t

    # --- Buoc 1: camera centers + orbit assignment ---
    cam_centers = np.zeros((len(rows), 3), dtype=np.float64)
    Rcw_list, tcw_list = [], []
    for i, row in enumerate(rows):
        Rcw, tcw = world_to_camera_matrix(row["qw"], row["qx"], row["qy"], row["qz"],
                                           row["tx"], row["ty"], row["tz"],
                                           convention=args.pose_convention)
        Rcw_list.append(Rcw)
        tcw_list.append(tcw)
        cam_centers[i] = camera_center_world(Rcw, tcw)

    world_up = np.array({"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[args.up_axis])

    if args.force_up_axis:
        axis, axis_point, angle_deg = world_up, cam_centers.mean(axis=0), 0.0
        print(f"[INFO] --force-up-axis: ep truc quy dao = world {args.up_axis} (0 deg)")
    else:
        axis, axis_point, angle_deg = fit_orbit_axis_from_camera_centers(cam_centers, world_up=world_up)
        print(f"[INFO] truc quy dao fit tu vi tri camera: "
              f"[{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}] (lech {angle_deg:.2f} deg so voi world {args.up_axis})")
        if angle_deg > 15.0:
            print(f"[WARN] truc quy dao lech {angle_deg:.2f} deg so voi world {args.up_axis} "
                  f"-> scene co the khong gravity-align. Dung --force-up-axis neu ban chac chan "
                  f"muon ep truc theo world {args.up_axis}.")

    orbit_id = assign_orbits(cam_centers, axis=axis, axis_point=axis_point,
                              n_orbits=args.n_orbits,
                              height_gap_std_mult=args.height_gap_std_mult,
                              min_orbit_size=args.min_orbit_size)
    n_orbits = int(orbit_id.max()) + 1
    print(f"[INFO] detected {n_orbits} orbits")
    for o in range(n_orbits):
        print(f"  orbit {o}: {int((orbit_id == o).sum())} images")

    # --- Buoc 2+3: visible mask per view, accumulate per orbit ---
    orbit_visible_count = np.zeros((N, n_orbits), dtype=np.int32)
    orbit_total = np.zeros(n_orbits, dtype=np.int32)

    per_image_visible = {}  # luu lai de dung cho heatmap sau

    for i, row in enumerate(rows):
        Rcw_t = torch.tensor(Rcw_list[i], device=device, dtype=torch.float32)
        tcw_t = torch.tensor(tcw_list[i], device=device, dtype=torch.float32)

        vis_mask = compute_visible_mask_for_view(
            xyz_world=xyz_t,
            scales_raw=scales_lin_t,
            Rcw=Rcw_t, tcw=tcw_t,
            fx=row["fx"], fy=row["fy"], cx=row["cx"], cy=row["cy"],
            width=row["width"], height=row["height"],
            depth_tolerance_scale=args.depth_tolerance_scale,
        )
        o = orbit_id[i]
        orbit_visible_count[:, o] += vis_mask.astype(np.int32)
        orbit_total[o] += 1
        per_image_visible[row["image_name"]] = (vis_mask, o, row)

        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            print(f"[{i+1}/{len(rows)}] processed")

    # --- Buoc 4+5: tinh score ---
    orbit_ratio = orbit_visible_count / np.maximum(orbit_total[None, :], 1)
    has_seen = orbit_visible_count > 0  # (N, n_orbits)

    n_orbits_stable = (orbit_ratio >= args.tau).sum(axis=1)

    activation_score = np.zeros(N, dtype=np.float64)
    seen_any = has_seen.any(axis=1)
    # trung binh orbit_ratio chi tren cac orbit da tung xuat hien
    sums = np.where(has_seen, orbit_ratio, 0.0).sum(axis=1)
    counts_seen = has_seen.sum(axis=1)
    activation_score[seen_any] = sums[seen_any] / np.maximum(counts_seen[seen_any], 1)

    # --- Buoc 6: phan loai bang Otsu ---
    scores_nonzero = activation_score[activation_score > 0]
    if len(scores_nonzero) > 10 and scores_nonzero.max() > scores_nonzero.min():
        tau_act = threshold_otsu(scores_nonzero)
    else:
        tau_act = 0.5
    print(f"[INFO] Otsu threshold on activation_score: {tau_act:.4f}")

    labels = np.full(N, "background", dtype=object)
    frac_stable = n_orbits_stable / max(n_orbits, 1)

    is_bts = (frac_stable >= 0.7) & (activation_score >= max(tau_act, 0.6))
    is_boundary = (~is_bts) & ((frac_stable >= 0.3) | (activation_score >= tau_act * 0.5))
    labels[is_boundary] = "boundary"
    labels[is_bts] = "BTS"

    print(f"[INFO] BTS: {int(is_bts.sum())}, boundary: {int(is_boundary.sum())}, "
          f"background: {int((labels=='background').sum())}")

    # --- Xuat CSV ---
    df = pd.DataFrame({
        "gaussian_idx": np.arange(N),
        "x": cloud.xyz[:, 0], "y": cloud.xyz[:, 1], "z": cloud.xyz[:, 2],
        "n_orbits_stable": n_orbits_stable,
        "activation_score": activation_score,
        "label": labels,
    })
    csv_out = outdir / "gaussian_scores.csv"
    df.to_csv(csv_out, index=False)
    print(f"[INFO] saved scores: {csv_out}")

    # --- Buoc 7a: scatter 3D ---
    # plot_3d_scatter(cloud.xyz, activation_score, labels, outdir / "score_scatter_3d.png")

    # --- Buoc 7b: heatmap overlay tren tung anh crop ---
    # if not args.no_heatmap:
    #     heatmap_dir = outdir / "heatmaps"
    #     heatmap_dir.mkdir(exist_ok=True)
    #     for i, row in enumerate(rows):
    #         Rcw_t = torch.tensor(Rcw_list[i], device=device, dtype=torch.float32)
    #         tcw_t = torch.tensor(tcw_list[i], device=device, dtype=torch.float32)
    #         vis_mask, o, _ = per_image_visible[row["image_name"]]
    #         out_path = heatmap_dir / f"{Path(row['image_name']).stem}_score_heatmap.png"
    #         crop_img_path = None
    #         if args.crop_images_dir:
    #             cand = resolve_path(args.crop_images_dir) / row["image_name"]
    #             if cand.exists():
    #                 crop_img_path = cand
    #         render_score_heatmap(
    #             xyz_t=xyz_t, activation_score_t=torch.tensor(activation_score, device=device, dtype=torch.float32),
    #             vis_mask=vis_mask, Rcw_t=Rcw_t, tcw_t=tcw_t,
    #             fx=row["fx"], fy=row["fy"], cx=row["cx"], cy=row["cy"],
    #             width=row["width"], height=row["height"],
    #             out_path=out_path, base_image_path=crop_img_path,
    #         )
    #     print(f"[INFO] saved heatmaps to: {heatmap_dir}")

    print("[DONE]")


def plot_3d_scatter(xyz: np.ndarray, activation_score: np.ndarray, labels: np.ndarray, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=activation_score,
                     cmap="viridis", s=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="activation_score", shrink=0.6)
    ax.set_title("Gaussian activation score (color) - kiem tra cluster co trung BTS khong")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def render_score_heatmap(xyz_t, activation_score_t, vis_mask, Rcw_t, tcw_t,
                          fx, fy, cx, cy, width, height, out_path: Path,
                          base_image_path: Path | None = None,
                          splat_radius_px: int = 3):
    """
    Chieu cac Gaussian visible trong view nay len anh, ve gia tri
    activation_score tai vi tri projected (dilate nhe bang splat_radius_px
    de de nhin), tao grayscale heatmap + overlay len anh goc neu co.
    """
    device = xyz_t.device
    idxs = np.nonzero(vis_mask)[0]
    if len(idxs) == 0:
        heat = np.zeros((height, width), dtype=np.float32)
    else:
        idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
        xyz_sel = xyz_t[idxs_t]
        means_cam = xyz_sel @ Rcw_t.T + tcw_t[None, :]
        x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
        inv_z = 1.0 / torch.clamp(z, min=1e-4)
        mu_x = (fx * x * inv_z + cx).round().long().clamp(0, width - 1)
        mu_y = (fy * y * inv_z + cy).round().long().clamp(0, height - 1)
        scores = activation_score_t[idxs_t]

        heat_sum = torch.zeros((height, width), device=device, dtype=torch.float32)
        heat_cnt = torch.zeros((height, width), device=device, dtype=torch.float32)

        r = splat_radius_px
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy = (mu_y + dy).clamp(0, height - 1)
                xx = (mu_x + dx).clamp(0, width - 1)
                flat = yy * width + xx
                heat_sum.view(-1).scatter_add_(0, flat, scores)
                heat_cnt.view(-1).scatter_add_(0, flat, torch.ones_like(scores))

        heat = (heat_sum / torch.clamp(heat_cnt, min=1)).cpu().numpy()
        heat[heat_cnt.cpu().numpy() == 0] = 0.0

    mx = float(heat.max())
    heat_norm = heat / mx if mx > 0 else heat
    heat_u8 = np.clip(np.round(heat_norm * 255.0), 0, 255).astype(np.uint8)

    gray_path = out_path
    Image.fromarray(heat_u8, mode="L").save(gray_path)

    if base_image_path is not None and base_image_path.exists():
        base = Image.open(base_image_path).convert("RGB").resize((width, height))
        heat_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        heat_rgb[..., 0] = heat_u8  # do len vung score cao
        heat_img = Image.fromarray(heat_rgb, mode="RGB")
        blended = Image.blend(base, heat_img, alpha=0.5)
        blended.save(out_path.with_name(out_path.stem + "_overlay.png"))


if __name__ == "__main__":
    main()