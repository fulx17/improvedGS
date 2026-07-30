"""
Load per-Gaussian BTS visibility labels tu CSV da tinh san
(gaussian_scores.csv, xuat boi bts_gaussian_visibility_protocol.py).

KHONG tinh lai protocol o day - chi doc va convert label string -> int8,
dam bao dung thu tu gaussian_idx khop voi thu tu Gaussian trong point_cloud.ply
ma GaussianModel.load_ply() da doc vao self._xyz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Quy uoc encoding, phai khop voi is_background trong scene/gaussian_model.py
# (is_background tra ve True khi label == 0)
_LABEL_TO_INT = {
    "background": 0,
    "boundary": 1,
    "BTS": 2,
}


def compute_gaussian_labels(csv_path: str) -> np.ndarray:
    """
        Doc gaussian_scores.csv, tra ve mang int8 (N,) theo dung thu tu gaussian_idx
        (0..N-1), de gan truc tiep vao GaussianModel.set_bts_labels().

        CSV phai co cot 'gaussian_idx' va 'label' (gia tri: background/boundary/BTS),
        dung dinh dang do bts_gaussian_visibility_protocol.py xuat ra.
    """
    if not csv_path:
        raise ValueError("--bts_scores_csv rong. Truyen duong dan toi gaussian_scores.csv.")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay CSV: {path}")

    df = pd.read_csv(path)
    required_cols = {"gaussian_idx", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV thieu cot: {missing}. Cot hien co: {list(df.columns)}")

    unknown_labels = set(df["label"].unique()) - set(_LABEL_TO_INT.keys())
    if unknown_labels:
        raise ValueError(
            f"CSV chua label khong nhan dien duoc: {unknown_labels}. "
            f"Chi chap nhan: {list(_LABEL_TO_INT.keys())}"
        )

    # sap xep theo gaussian_idx tang dan de dam bao dung thu tu 0..N-1
    df_sorted = df.sort_values("gaussian_idx").reset_index(drop=True)

    expected_idx = np.arange(len(df_sorted))
    actual_idx = df_sorted["gaussian_idx"].to_numpy()
    if not np.array_equal(expected_idx, actual_idx):
        raise ValueError(
            "gaussian_idx trong CSV khong lien tuc 0..N-1 (co the thieu dong hoac trung idx). "
            "Kiem tra lai file gaussian_scores.csv, hoac Gaussian count trong point_cloud.ply "
            "da thay doi so voi luc chay protocol."
        )

    labels_int = df_sorted["label"].map(_LABEL_TO_INT).to_numpy().astype(np.int8)

    n_bts = int((labels_int == 2).sum())
    n_boundary = int((labels_int == 1).sum())
    n_bg = int((labels_int == 0).sum())
    print(f"[bts_visibility] loaded {len(labels_int)} labels tu {path}: "
          f"BTS={n_bts}, boundary={n_boundary}, background={n_bg}")

    return labels_int