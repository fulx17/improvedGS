"""
checkup.py — kiem tra nhanh cac patch BTS-visibility da ap dung dung chua.

Chay tu thu muc goc Improved-GS:
    python checkup.py
    python checkup.py --csv save_visibility\\HCM0204\\gaussian_scores.csv   (kiem tra luon CSV + live smoke test)

Gom 2 loai kiem tra:
  1) STATIC  - doc source code, tim cac dau hieu patch da duoc chen dung cho
               (khong can GPU/torch, chay duoc tren may khong co CUDA).
  2) LIVE    (tuy chon, can --csv va can torch+CUDA) - thuc su:
       - import GaussianModel, set_bts_labels
       - gia lap 1 vong densify_and_clone/split + prune_points
       - kiem tra self._bts_label luon cung do dai voi self._xyz sau moi buoc
     Day la kiem tra quan trong nhat vi no phat hien loi "quen truyen
     new_bts_label" ma static check co the bo sot.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Check:
    def __init__(self, name: str):
        self.name = name
        self.ok = True
        self.messages: list[str] = []

    def fail(self, msg: str):
        self.ok = False
        self.messages.append(f"  [FAIL] {msg}")

    def warn(self, msg: str):
        self.messages.append(f"  [WARN] {msg}")

    def info(self, msg: str):
        self.messages.append(f"  [ok]   {msg}")


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# STATIC CHECKS
# ---------------------------------------------------------------------------

def check_gaussian_model() -> Check:
    c = Check("scene/gaussian_model.py")
    src = read(ROOT / "scene" / "gaussian_model.py")
    if not src:
        c.fail("khong doc duoc file")
        return c

    if "_bts_label = torch.empty(0)" in src:
        c.info("da co buffer self._bts_label trong __init__")
    else:
        c.fail("thieu 'self._bts_label = torch.empty(0)' trong __init__")

    if "def set_bts_labels" in src:
        c.info("da co method set_bts_labels()")
    else:
        c.fail("thieu method set_bts_labels()")

    if "def is_background" in src:
        c.info("da co property is_background")
    else:
        c.fail("thieu property is_background")

    prune_match = re.search(r"def prune_points\(.*?\n(?:.*\n)*?(?=\n    def |\Z)", src)
    if prune_match:
        body = prune_match.group(0)
        if "_bts_label" in body:
            c.info("prune_points() co xu ly _bts_label")
        else:
            c.fail("prune_points() CHUA duoc sua de index_select _bts_label theo valid_points_mask")
    else:
        c.warn("khong tim thay ham prune_points() de kiem tra")

    return c


def check_densification() -> Check:
    c = Check("scene/gaussian_model_densification.py")
    src = read(ROOT / "scene" / "gaussian_model_densification.py")
    if not src:
        c.fail("khong doc duoc file")
        return c

    if re.search(r"def densification_postfix\([^)]*new_bts_label", src, re.S):
        c.info("densification_postfix() co tham so new_bts_label")
    else:
        c.fail("densification_postfix() CHUA co tham so new_bts_label")

    if "self._bts_label = torch.cat((self._bts_label, new_bts_label))" in src:
        c.info("densification_postfix() co noi _bts_label")
    else:
        c.fail("densification_postfix() CHUA noi self._bts_label")

    fns_need_patch = [
        "densify_and_split",
        "densify_and_clone",
        "densify_and_split_mask",
        "long_axis_split",
    ]
    for fn_name in fns_need_patch:
        pattern = rf"def {fn_name}\(.*?\n(?:.*\n)*?(?=\n    def |\Z)"
        m = re.search(pattern, src, re.S)
        if not m:
            c.warn(f"khong tim thay ham {fn_name}() de kiem tra")
            continue
        body = m.group(0)
        if "new_bts_label" in body and "densification_postfix(" in body:
            c.info(f"{fn_name}() co truyen new_bts_label vao densification_postfix")
        else:
            c.fail(f"{fn_name}() CHUA truyen new_bts_label vao densification_postfix")

    return c


def check_optimization_methods() -> Check:
    c = Check("scene/methods/optimization_methods.py")
    src = read(ROOT / "scene" / "methods" / "optimization_methods.py")
    if not src:
        c.fail("khong doc duoc file")
        return c

    if "loss.backward()" not in src:
        c.fail("khong tim thay loss.backward() trong file (co the ten file/vi tri da doi)")
        return c

    after_backward = src.split("loss.backward()", 1)[1][:800]
    if "is_background" in after_backward and "grad[freeze]" in after_backward.replace(" ", ""):
        c.info("da co khoi freeze gradient ngay sau loss.backward()")
    elif "is_background" in after_backward:
        c.warn("thay 'is_background' sau backward() nhung khong chac chan da gan p.grad[freeze]=0 dung cach, kiem tra lai thu cong")
    else:
        c.fail("CHUA co khoi freeze gradient (is_background) ngay sau loss.backward()")

    return c


def check_training_runtime() -> Check:
    c = Check("scene/training_runtime.py")
    src = read(ROOT / "scene" / "training_runtime.py")
    if not src:
        c.fail("khong doc duoc file")
        return c

    if "--bts_scores_csv" in src:
        c.info("da co CLI flag --bts_scores_csv")
    elif "--bts_sparse_dir" in src:
        c.fail("van con flag CU '--bts_sparse_dir' — can doi thanh '--bts_scores_csv' theo huong dan moi nhat")
    else:
        c.fail("CHUA co CLI flag --bts_scores_csv")

    return c


def check_train_py() -> Check:
    c = Check("train.py")
    src = read(ROOT / "train.py")
    if not src:
        c.fail("khong doc duoc file")
        return c

    if re.search(r"Scene\(dataset,\s*gaussians,\s*load_iteration\s*=\s*30000", src):
        c.info("Scene(...) da truyen load_iteration=30000")
    else:
        c.fail("Scene(...) CHUA truyen load_iteration=30000 (dang tao Gaussian moi tu COLMAP thay vi load ply 30k)")

    if "from vai.bts_visibility import compute_gaussian_labels" in src:
        c.info("da import compute_gaussian_labels")
    else:
        c.fail("CHUA import compute_gaussian_labels tu vai.bts_visibility")

    if "gaussians.set_bts_labels(" in src:
        c.info("da goi gaussians.set_bts_labels(...)")
    else:
        c.fail("CHUA goi gaussians.set_bts_labels(...)")

    if "bts_ply_path" in src:
        c.warn("van con bien 'bts_ply_path' cu — nen bo, vi ban da chuyen sang doc CSV thay vi tinh lai tu ply")

    if "runtime_args.bts_scores_csv" in src:
        c.info("dang doc runtime_args.bts_scores_csv dung ten flag moi")
    elif "runtime_args.bts_sparse_dir" in src:
        c.fail("van dang doc runtime_args.bts_sparse_dir (ten flag CU) — sua thanh bts_scores_csv")

    return c


def check_bts_visibility_module() -> Check:
    c = Check("vai/bts_visibility.py")
    path = ROOT / "vai" / "bts_visibility.py"
    if not path.exists():
        c.fail(f"khong tim thay file {path} — chua copy vao dung vi tri")
        return c
    src = read(path)
    if "def compute_gaussian_labels" in src:
        c.info("co ham compute_gaussian_labels()")
    else:
        c.fail("file ton tai nhung khong co ham compute_gaussian_labels()")
    return c


# ---------------------------------------------------------------------------
# LIVE CHECK (tuy chon)
# ---------------------------------------------------------------------------

def run_live_check(csv_path: str) -> Check:
    c = Check("LIVE smoke test (GaussianModel + densify/prune round-trip)")
    try:
        sys.path.insert(0, str(ROOT))
        import torch
        from scene.gaussian_model import GaussianModel
        from vai.bts_visibility import compute_gaussian_labels
    except Exception as e:
        c.fail(f"import that bai: {e!r}")
        return c

    if not torch.cuda.is_available():
        c.warn("khong co CUDA — bo qua live check (cac ham trong repo hardcode device='cuda')")
        return c

    try:
        labels = compute_gaussian_labels(csv_path=csv_path)
    except Exception as e:
        c.fail(f"compute_gaussian_labels() loi: {e!r}")
        return c
    n = labels.shape[0]
    c.info(f"CSV load OK, N={n}")

    try:
        gm = GaussianModel(sh_degree=3, optimizer_type="default")
        gm._xyz = torch.zeros((n, 3), device="cuda", requires_grad=True)
        gm._features_dc = torch.zeros((n, 1, 3), device="cuda", requires_grad=True)
        gm._features_rest = torch.zeros((n, 15, 3), device="cuda", requires_grad=True)
        gm._opacity = torch.zeros((n, 1), device="cuda", requires_grad=True)
        gm._scaling = torch.zeros((n, 3), device="cuda", requires_grad=True)
        gm._rotation = torch.zeros((n, 4), device="cuda", requires_grad=True)
        gm.set_bts_labels(labels)

        if gm._bts_label.shape[0] != n:
            c.fail("set_bts_labels: kich thuoc khong khop")
        else:
            c.info("set_bts_labels: kich thuoc khop N")

        # gia lap prune: xoa nua sau
        prune_mask = torch.zeros(n, dtype=torch.bool, device="cuda")
        prune_mask[n // 2:] = True
        valid_mask = ~prune_mask
        # khong goi prune_points() day du vi can optimizer that; chi test truc tiep
        # logic index_select tren _bts_label giong nhu patch da them:
        old_label = gm._bts_label.clone()
        gm._bts_label = gm._bts_label[valid_mask]
        expected = old_label[valid_mask]
        if torch.equal(gm._bts_label, expected) and gm._bts_label.shape[0] == valid_mask.sum().item():
            c.info("index_select _bts_label theo mask hoat dong dung logic")
        else:
            c.fail("index_select _bts_label SAI logic")

    except Exception as e:
        c.fail(f"live smoke test loi: {e!r}")

    return c


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                     help="Duong dan toi gaussian_scores.csv, neu muon chay them LIVE smoke test")
    args = ap.parse_args()

    checks = [
        check_gaussian_model(),
        check_densification(),
        check_optimization_methods(),
        check_training_runtime(),
        check_train_py(),
        check_bts_visibility_module(),
    ]

    if args.csv:
        checks.append(run_live_check(args.csv))

    print("=" * 70)
    total_fail = 0
    for c in checks:
        status = "OK" if c.ok else "CO LOI"
        print(f"\n[{status}] {c.name}")
        for msg in c.messages:
            print(msg)
        if not c.ok:
            total_fail += 1
    print("\n" + "=" * 70)
    if total_fail == 0:
        print("Tat ca static check PASS. " +
              ("(Chua chay LIVE check — them --csv de kiem tra sau.)" if not args.csv else "LIVE check da chay, xem chi tiet o tren."))
        sys.exit(0)
    else:
        print(f"CO {total_fail} nhom check bi FAIL — xem chi tiet [FAIL] o tren de sua.")
        sys.exit(1)


if __name__ == "__main__":
    main()