# ───────────────── gradcam_perlayer_slowfast.py ─────────────────
"""
Run Grad-CAM++ on *each* chosen layer separately.

Creates:
    slowfast/work_dir/gradcam_per_layer/<layer-name>/frameXXX.jpg

Prints:
    • Layer processed
    • Top-1 predicted gloss id + string
"""

import os, sys, glob, cv2
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

# ───── 1.  make pytorch-grad-cam image-scaling robust to weird sizes ──────
import pytorch_grad_cam.utils.image as _cam_utils
import pytorch_grad_cam.base_cam    as _base_cam
_orig_scale = _cam_utils.scale_cam_image
def _safe_scale(cam, target_size=None):
    if cam.ndim == 3: cam = cam.mean(0)
    if cam.ndim == 1: cam = cam[None, :]
    if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
        target_size = (int(target_size[-1]), int(target_size[-2]))  # (W,H)
    return _orig_scale(cam, target_size)
_cam_utils.scale_cam_image = _safe_scale
_base_cam.scale_cam_image  = _safe_scale
# ───────────────────────────────────────────────────────────────────────────

# ───── 2.  project imports ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))   # repo root
from slowfast_setup import parse_args, load_dataset, load_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# ───────────────────────────────────────────────────────────────────────────

# normalise / de-normalise utils -------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])[None, None, :]
denorm = lambda x: np.clip(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

def reshape_transform(t):
    """Make any tensor coming from SlowFast look like [N,C,H,W] for Grad-CAM."""
    if isinstance(t, tuple):           t = t[0]
    if isinstance(t, PackedSequence):  t = t.data
    if t.dim() == 5:                   # [B,C,T,H,W] → [B*T,C,H,W]
        B,C,T,H,W = t.shape
        return t.permute(0,2,1,3,4).reshape(B*T, C, H, W)
    if t.dim() == 3:                   # [B,C,T] → [B*T,C,1,1]
        B,C,T = t.shape
        return t.permute(0,2,1).reshape(B*T, C, 1, 1)
    if t.dim() == 2:                   # [B,C] → [B*C,1,1]
        B,C = t.shape
        return t.view(B*C, 1, 1)
    return t.flatten().view(-1, 1, 1)  # fallback

# --------------------------------------------------------------------------
def load_frames(info_str, args):
    """Return raw RGB frames as float32 in [0,1], shape [N,H,W,3]."""
    rel = info_str.split("|")[1]
    root = Path(args.dataset_info["dataset_root"])
    patt = root / "features" / f"fullFrame-256x256px/{args.mode}" / rel
    paths = sorted(glob.glob(str(patt)))
    return np.stack([cv2.imread(p)[:,:,::-1].astype(np.float32)/255.0
                     for p in paths])

# --------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / model --------------------------------------------------------
    gloss_dict = np.load(args.dataset_info["dict_path"], allow_pickle=True).item()
    feeder     = load_dataset(args=args, gloss_dict=gloss_dict)
    model      = load_model(args=args, gloss_dict=gloss_dict,
                            checkpoint_path=args.slowfast_ckpt).to(device)

    # choose one sample ------------------------------------------------------
    data, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])
    frames = load_frames(info[0], args)             # raw RGB frames
    inp    = data.to(device)                        # SlowFast input tensor

    # ───── 3.  make first BiLSTM CAM-friendly (tensor out, not tuple) ──────
    class RNNOutputOnly(nn.Module):
        def __init__(self, rnn): super().__init__(); self.rnn = rnn
        def forward(self, x, hx=None):
            packed,_ = self.rnn(x, hx)
            padded,_ = rnn_utils.pad_packed_sequence(packed, batch_first=False)
            return padded
    model.temporal_model[0].rnn = RNNOutputOnly(model.temporal_model[0].rnn)

    # patch its forward so non-tuple output is OK ---------------------------
    import types
    def fwd(self, src, lens, hid=None):
        packed = rnn_utils.pack_padded_sequence(src, lens)
        out = self.rnn(packed, hid)
        if isinstance(out, torch.Tensor): rnn_out, hid = out, None
        elif isinstance(out, rnn_utils.PackedSequence):
            rnn_out,_ = rnn_utils.pad_packed_sequence(out); hid=None
        else:
            packed_out, hid = out
            rnn_out,_ = rnn_utils.pad_packed_sequence(packed_out)
        if self.bidirectional and hid is not None:
            hid = self._cat_directions(hid)
        if isinstance(hid, tuple): hid = torch.cat(hid, 0)
        return {"predictions": rnn_out, "hidden": hid}
    model.temporal_model[0].forward = types.MethodType(fwd, model.temporal_model[0])

    # wrapper so Grad-CAM sees per-class **scores** -------------------------
    class CAMWrapper(nn.Module):
        def __init__(self, core, sample_len_x): super().__init__(); self.core = core; self.L = sample_len_x
        def forward(self, x):
            B = x.size(0)
            l = self.L.repeat(B)
            train_state = self.core.training
            self.core.train()
            with torch.set_grad_enabled(True):
                out = self.core(x, l)["sequence_logits"][0]   # [T,B,C] or [B,C]
                if out.dim() == 3: out = out.permute(1,0,2).mean(1)  # [B,C]
            self.core.train(train_state)
            return out
    cam_model = CAMWrapper(model, len_x.to(device)).to(device)

    # layers to visualise ----------------------------------------------------
    layers = [
        ("stem_fast", model.conv2d.s1.pathway1_stem),
        ("res3_slow", model.conv2d.s3.pathway0_res1.branch2.c),
        ("res5_fast", model.conv2d.s5.pathway1_res2.branch2.c),
        ("bilstm0",   model.temporal_model[0].rnn),
    ]

    # top-1 predicted gloss --------------------------------------------------
    with torch.no_grad():
        top_id  = int(cam_model(inp)[0].argmax())
        top_txt = gloss_dict.get(top_id, "<unk>")
    print(f"\nTop-1 predicted gloss:  {top_id}  ({top_txt})\n")

    out_root = Path("./slowfast/work_dir/gradcam_per_layer")

    # ---- 4. loop over layers ----------------------------------------------
    for lname, layer in layers:
        print(f"→ Running Grad-CAM on layer [{lname}]")

        cam = GradCAMPlusPlus(model=cam_model,
                              target_layers=[layer],
                              reshape_transform=reshape_transform)

        gray = cam(inp, targets=[ClassifierOutputTarget(top_id)],
                   eigen_smooth=True, aug_smooth=False)        # (B*T,Hl,Wl)

        BT, Hl, Wl = gray.shape
        B          = inp.shape[0]
        T_heat     = BT // B                       # heat-maps per video

        # align heat-maps to raw frames -------------------------------------
        vid_len       = frames.shape[0]
        frame_idxs    = np.linspace(0, vid_len-1, T_heat).round().astype(int)

        save_dir = out_root / lname
        save_dir.mkdir(parents=True, exist_ok=True)

        for i_hm, i_fr in enumerate(frame_idxs):
            heat = gray[i_hm]                     # (Hl,Wl)
            img  = denorm(frames[i_fr])           # (H,W,3) in [0,1]
            vis  = show_cam_on_image(img, heat, use_rgb=True)
            cv2.imwrite(str(save_dir / f"frame{i_hm:03d}.jpg"), vis)

        print(f"   saved {T_heat} heat-maps → {save_dir}")

    print("\nDone!  Check results in  slowfast/work_dir/gradcam_per_layer/")
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────────────────────
