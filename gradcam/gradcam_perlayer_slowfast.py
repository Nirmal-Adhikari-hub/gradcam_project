# ───────────────── gradcam_per_layer.py ─────────────────
"""
Run Grad-CAM++ on *each* target layer separately.

• Writes one sub-folder per layer under  ./slowfast/work_dir/gradcam_per_layer/
  e.g.  layer3/   layer4/  …
• Prints:
    – the layer name
    – the top-1 predicted gloss id and its string (from gloss_dict)
"""

import os, sys, glob, cv2, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

# ────────────── 1. tiny fix for cv2.resize inside pytorch-grad-cam ─────────────
import pytorch_grad_cam.utils.image     as _cam_utils
import pytorch_grad_cam.base_cam        as _base_cam
_original_scale = _cam_utils.scale_cam_image
def _safe_scale(cam: np.ndarray, target_size):
    if cam.ndim == 3: cam = cam.mean(0)
    if cam.ndim == 1: cam = cam[None, :]
    if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
        target_size = (int(target_size[-1]), int(target_size[-2]))  # (W,H)
    return _original_scale(cam, target_size)
_cam_utils.scale_cam_image = _safe_scale
_base_cam.scale_cam_image  = _safe_scale
# ───────────────────────────────────────────────────────────────────────────────

# ---------------------- 2. project imports & helpers --------------------------
sys.path.append(str(Path(__file__).resolve().parent))   # project root
from slowfast_setup import parse_args, load_dataset, load_model, evaluate_single_sample
from pytorch_grad_cam               import GradCAMPlusPlus
from pytorch_grad_cam.utils.image   import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

imagenet_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
imagenet_std  = np.array([0.229, 0.224, 0.225])[None, None, :]
denorm = lambda x: np.clip(x * imagenet_std + imagenet_mean, 0, 1)

def reshape_transform(t):
    if isinstance(t, tuple):         t = t[0]
    if isinstance(t, PackedSequence): t = t.data
    if t.dim() == 5:                 # [B,C,T,H,W] -> [B*T,C,H,W]
        b,c,tim,h,w = t.shape; return t.permute(0,2,1,3,4).reshape(b*tim, c, h, w)
    if t.dim() == 3:                 # [B,C,T] -> [B*T,C,1,1]
        b,c,tim = t.shape; return t.permute(0,2,1).reshape(b*tim, c,1,1)
    if t.dim() == 2:                 # [B,C]
        b,c = t.shape; return t.view(b*c,1,1)
    return t.flatten().view(-1,1,1)  # fallback

# ---------------------- 3. main ------------------------------------------------
def main():
    args = parse_args(); verbose = getattr(args, "verbose", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / model ----------------------------------------------------------
    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()
    feeder      = load_dataset(args=args, gloss_dict=gloss_dict)
    model       = load_model(args=args, gloss_dict=gloss_dict,
                             checkpoint_path=args.slowfast_ckpt).to(device)

    # sample ------------------------------------------------------------------
    data, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])
    frames = load_frames(info[0], args)
    inp    = data.to(device)

    # LSTM wrappers -----------------------------------------------------------
    class RNNOutputOnly(nn.Module):
        def __init__(self, rnn): super().__init__(); self.rnn = rnn
        def forward(self,x,hx=None):
            packed,_ = self.rnn(x,hx)
            padded,_ = rnn_utils.pad_packed_sequence(packed, batch_first=False)
            return padded
    model.temporal_model[0].rnn = RNNOutputOnly(model.temporal_model[0].rnn)

    # patch BiLSTM.forward for tensor input -----------------------------------
    import types
    def fwd(self,src,lens,hid=None):
        packed = rnn_utils.pack_padded_sequence(src,lens)
        result = self.rnn(packed,hid)
        if isinstance(result, torch.Tensor): rnn_out, hid = result, None
        elif isinstance(result, rnn_utils.PackedSequence):
            rnn_out,_ = rnn_utils.pad_packed_sequence(result); hid=None
        else:
            packed_out,hid = result; rnn_out,_=rnn_utils.pad_packed_sequence(packed_out)
        if self.bidirectional and hid is not None: hid = self._cat_directions(hid)
        if isinstance(hid, tuple): hid = torch.cat(hid,0)
        return {"predictions": rnn_out, "hidden": hid}
    model.temporal_model[0].forward = types.MethodType(fwd, model.temporal_model[0])

    # wrapper so Grad-CAM sees logits ----------------------------------------
    class CAMWrapper(nn.Module):
        def __init__(self, slr_model, sample_len_x):
            super().__init__()
            self.model = slr_model
            self.sample_len_x = sample_len_x

        def forward(self, x):
            B = x.size(0)
            len_x_batch = self.sample_len_x.repeat(B)
            was_training = self.model.training
            self.model.train()
            with torch.set_grad_enabled(True):
                out = self.model(x, len_x_batch)
                seq_logits = out['sequence_logits'][0]
                if seq_logits.dim() == 3:
                    seq_logits = seq_logits.permute(1, 0, 2)
                    score = seq_logits.mean(dim=1)
                else:
                    score = seq_logits.mean(dim=0).unsqueeze(0)
            self.model.train(was_training)
            return score
    cam_model = CAMWrapper(model, len_x.to(device)).to(device)

    # pick the layers you care about -----------------------------------------
    layers = [
        ("stem_fast",   model.conv2d.s1.pathway1_stem),                     # low-level fast
        ("res3_slow",   model.conv2d.s3.pathway0_res1.branch2.c),          # mid-level slow
        ("res5_fast",   model.conv2d.s5.pathway1_res2.branch2.c),          # high-level fast
        ("bilstm0",     model.temporal_model[0].rnn)                       # first BiLSTM
    ]

    # top-1 label -------------------------------------------------------------
    with torch.no_grad():
        top_id  = int(cam_model(inp)[0].argmax())
        top_txt = gloss_dict.get(top_id, "<unk>")
    print(f"\nTop-1 predicted gloss:  {top_id}  ({top_txt})\n")

    out_root = Path("./slowfast/work_dir/gradcam_per_layer")
    b,T,C,H,W = inp.shape

    # loop over layers --------------------------------------------------------
    for name, layer in layers:
        print(f"→ Running Grad-CAM on layer [{name}]")

        cam = GradCAMPlusPlus(model=cam_model,
                              target_layers=[layer],
                              reshape_transform=reshape_transform)

        gray = cam(input_tensor=inp,
                   targets=[ClassifierOutputTarget(top_id)],
                   eigen_smooth=True, aug_smooth=False)          # [B*T,H,W]

        cams = gray.reshape(b, T, H, W)                          # back to [B,T]
        save_dir = out_root / name
        save_dir.mkdir(parents=True, exist_ok=True)

        for t in range(T):
            heat = cams[0,t]
            img  = denorm(frames[t])
            vis  = show_cam_on_image(img, heat, use_rgb=True)
            cv2.imwrite(str(save_dir / f"frame{t:03d}.jpg"), vis)

        print(f"   saved {T} frames to  {save_dir}")

    print("\nDone!  Check the layer folders inside  slowfast/work_dir/gradcam_per_layer/")
# ------------------------------------------------------------------------------
def load_frames(info_str, args):
    rel = info_str.split("|")[1]
    root = Path(args.dataset_info["dataset_root"])
    patt = root / "features" / f"fullFrame-256x256px/{args.mode}" / rel
    paths = sorted(glob.glob(str(patt)))
    return np.stack([cv2.imread(p)[:,:,::-1].astype(np.float32)/255. for p in paths])

if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────────────────────────
