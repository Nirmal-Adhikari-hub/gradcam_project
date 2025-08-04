# ───────────────── gradcam_perlayer_slowfast.py ─────────────────
"""
Generate Grad-CAM++ overlays for several key layers of the SlowFast-based
SLR model, one folder per layer:

    slowfast/work_dir/gradcam_per_layer/<layer_name>/frameXXX.jpg
"""

import sys, glob, cv2
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

# ───── 1  –  make pytorch-grad-cam’s internal resize more robust ─────────
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
# ─────────────────────────────────────────────────────────────────────────

# ───── 2  –  repo-local imports ──────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))          # repo root
from slowfast_setup import parse_args, load_dataset, load_model
from pytorch_grad_cam                      import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets  import ClassifierOutputTarget
from pytorch_grad_cam.utils.image          import show_cam_on_image
# ─────────────────────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
STD  = np.array([0.229, 0.224, 0.225])[None, None, :]
denorm = lambda x: np.clip(x * STD + MEAN, 0, 1)

def reshape_transform(t):
    if isinstance(t, tuple):            t = t[0]
    if isinstance(t, PackedSequence):   t = t.data
    if t.dim() == 5:                    # [B,C,T,H,W] → [B*T,C,H,W]
        B,C,T,H,W = t.shape
        return t.permute(0,2,1,3,4).reshape(B*T, C, H, W)
    if t.dim() == 3:                    # [B,C,T]  → [B*T,C,1,1]
        B,C,T = t.shape
        return t.permute(0,2,1).reshape(B*T, C, 1, 1)
    if t.dim() == 2:                    # [B,C]    → [B*C,1,1]
        B,C = t.shape
        return t.view(B*C, 1, 1)
    return t.flatten().view(-1, 1, 1)

def load_frames(info_str, args):
    rel   = info_str.split("|")[1]
    root  = Path(args.dataset_info["dataset_root"])
    patt  = root / "features" / f"fullFrame-256x256px/{args.mode}" / rel
    paths = sorted(glob.glob(str(patt)))
    return np.stack([cv2.imread(p)[:,:,::-1].astype(np.float32) / 255.0
                     for p in paths])

# ─────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset / model ───────────────────────────────────────────────────
    gloss_dict = np.load(args.dataset_info["dict_path"], allow_pickle=True).item()
    feeder     = load_dataset(args=args, gloss_dict=gloss_dict)
    model      = load_model(args=args, gloss_dict=gloss_dict,
                            checkpoint_path=args.slowfast_ckpt).to(device)

    # one sample -----------------------------------------------------------
    data, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])
    frames = load_frames(info[0], args)
    inp    = data.to(device)

    # ── make first BiLSTM CAM-friendly (tensor out) ───────────────────────
    class RNNOutputOnly(nn.Module):
        def __init__(self, rnn): super().__init__(); self.rnn = rnn
        def forward(self, x, hx=None):
            packed,_ = self.rnn(x, hx)
            padded,_ = rnn_utils.pad_packed_sequence(packed, batch_first=False)
            return padded
    model.temporal_model[0].rnn = RNNOutputOnly(model.temporal_model[0].rnn)

    # patch its forward so tensor return is accepted ----------------------
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

    # wrapper that outputs class scores (needed by Grad-CAM) --------------
    class CAMWrapper(nn.Module):
        def __init__(self, core, sample_len): super().__init__(); self.core=core; self.L=sample_len
        def forward(self, x):
            B = x.size(0); l = self.L.repeat(B)
            training = self.core.training
            self.core.train()
            with torch.set_grad_enabled(True):
                logits = self.core(x, l)["sequence_logits"][0]  # [T,B,C] or [B,C]
                if logits.dim() == 3:
                    logits = logits.permute(1, 0, 2).mean(1)   # [B,C]
            self.core.train(training)
            return logits
    cam_model = CAMWrapper(model, len_x.to(device)).to(device)

    # layers to visualise ---------------------------------------------------
    layers = [
        ("stem_fast", model.conv2d.s1.pathway1_stem),
        ("res3_slow", model.conv2d.s3.pathway0_res1.branch2.c),
        ("res5_fast", model.conv2d.s5.pathway1_res2.branch2.c),
        ("bilstm0",   model.temporal_model[0].rnn),
    ]

    # predicted gloss -------------------------------------------------------
    with torch.no_grad():
        pred_scores = cam_model(inp)
        top_id      = int(pred_scores[0].argmax())
        top_txt     = gloss_dict.get(top_id, "<unk>")
    print(f"\nTop-1 predicted gloss:  {top_id}  ({top_txt})\n")

    out_root = Path("slowfast/work_dir/gradcam_per_layer")

    # ── loop over layers ───────────────────────────────────────────────────
    for lname, layer in layers:
        print(f"→ Running Grad-CAM on layer [{lname}]")

        cam = GradCAMPlusPlus(model=cam_model,
                              target_layers=[layer],
                              reshape_transform=reshape_transform)

        # Grad-CAM heat-maps (N = B×T); shape can be (N,H,W) **or** (N, P)
        gray = cam(inp, targets=[ClassifierOutputTarget(top_id)],
                   eigen_smooth=True, aug_smooth=False)

        if gray.ndim == 1:                             # (N,) → (N,1)
            gray = gray[:, None]
        if gray.ndim == 2:                             # flat maps → resize later
            Hm, Wm = 1, gray.shape[1]                  # placeholder
        else:                                          # (N,H,W)
            Hm, Wm = gray.shape[-2:]

        N_maps = gray.shape[0]
        vid_len = frames.shape[0]
        # linearly match heat-maps to frames
        frame_ids = np.linspace(0, vid_len-1, N_maps).round().astype(int)

        save_dir = out_root / lname
        save_dir.mkdir(parents=True, exist_ok=True)

        for i_hm, fr_id in enumerate(frame_ids):
            hm = gray[i_hm]                            # array(...)
            if hm.ndim == 1: hm = hm[:, None]          # (P,) → (P,1)
            if hm.ndim == 2 and (hm.shape != frames.shape[1:3]):
                hm = cv2.resize(hm.astype(np.float32),
                                (frames.shape[2], frames.shape[1]))
            img = denorm(frames[fr_id])
            vis = show_cam_on_image(img, hm, use_rgb=True)
            cv2.imwrite(str(save_dir / f"frame{i_hm:03d}.jpg"), vis)

        print(f"   saved {N_maps} heat-maps → {save_dir}")

    print("\nDone!  Check results in  slowfast/work_dir/gradcam_per_layer/")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# ─────────────────────────────────────────────────────────────────────────
