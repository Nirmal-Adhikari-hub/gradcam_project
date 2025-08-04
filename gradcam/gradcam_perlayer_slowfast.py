# ───────── gradcam_perlayer_slowfast.py  (copy-paste whole file) ──────────
"""
Run Grad-CAM++ on several key layers of SlowFast and save one overlay
per sampled frame.

Outputs go to:  slowfast/work_dir/gradcam_per_layer/<layer>/frameXXX.jpg
"""

import sys, glob, cv2, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

# ─── 1. make pytorch-grad-cam’s resize tolerant of odd inputs ────────────
import pytorch_grad_cam.utils.image as _im
import pytorch_grad_cam.base_cam    as _bc
_orig = _im.scale_cam_image
def _safe(img, tgt=None):
    if img.ndim == 3:                       # squeeze colour-like dim
        img = img.mean(0)
    if img.ndim == 1:                       # promote scalar map
        img = img[None]
    if isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
        tgt = (int(tgt[-1]), int(tgt[-2]))  # (W, H) for cv2
    return _orig(img, tgt)
_im.scale_cam_image = _safe
_bc.scale_cam_image = _safe
# ──────────────────────────────────────────────────────────────────────────

# project-level helpers ----------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent))
from slowfast_setup import parse_args, load_dataset, load_model
from pytorch_grad_cam               import GradCAMPlusPlus
from pytorch_grad_cam.utils.image   import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MEAN = np.array([0.485, 0.456, 0.406])[None, None]
STD  = np.array([0.229, 0.224, 0.225])[None, None]
denorm = lambda x: np.clip(x * STD + MEAN, 0, 1)

def reshape(t):
    """Convert any SlowFast tensor into [N,C,H,W] for Grad-CAM."""
    if isinstance(t, tuple):              t = t[0]
    if isinstance(t, PackedSequence):     t = t.data
    if t.dim() == 5:                      # [B,C,T,H,W] → [B*T,C,H,W]
        B, C, T, H, W = t.shape
        return t.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    if t.dim() == 3:                      # [B,C,T]   → [B*T,C,1,1]
        B, C, T = t.shape
        return t.permute(0, 2, 1).reshape(B * T, C, 1, 1)
    if t.dim() == 2:                      # [B,C]     → [B*C,1,1]
        B, C = t.shape
        return t.view(B * C, 1, 1)
    return t.flatten().view(-1, 1, 1)     # fallback for 1-D

def load_frames(info, args):
    rel   = info.split('|')[1]
    root  = Path(args.dataset_info['dataset_root'])
    patt  = root / 'features' / f'fullFrame-256x256px/{args.mode}' / rel
    paths = sorted(glob.glob(str(patt)))
    return np.stack([cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255.0
                     for p in paths])

# ──────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    dev    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gloss_dict = np.load(args.dataset_info['dict_path'],
                         allow_pickle=True).item()
    feeder = load_dataset(args=args, gloss_dict=gloss_dict)
    model  = load_model(args=args, gloss_dict=gloss_dict,
                        checkpoint_path=args.slowfast_ckpt).to(dev)

    data, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])
    frames = load_frames(info[0], args)          # (N,H,W,3) float32 [0,1]
    inp    = data.to(dev)                        # (B,C,T,H,W)

    # ─── 2. patch first BiLSTM so it returns a Tensor, not a tuple ────────
    class ROut(nn.Module):
        def __init__(s, r): super().__init__(); s.r = r
        def forward(s, x, h=None):
            packed, _ = s.r(x, h)
            pad, _    = rnn_utils.pad_packed_sequence(packed, batch_first=False)
            return pad                              # [T,B,C]
    model.temporal_model[0].rnn = ROut(model.temporal_model[0].rnn)

    import types
    def new_forward(self, src, lens, h=None):
        packed = rnn_utils.pack_padded_sequence(src, lens)
        out = self.rnn(packed, h)
        if isinstance(out, torch.Tensor):
            rnn_out, hid = out, None
        elif isinstance(out, rnn_utils.PackedSequence):
            rnn_out, _ = rnn_utils.pad_packed_sequence(out); hid = None
        else:
            p_out, hid = out
            rnn_out, _ = rnn_utils.pad_packed_sequence(p_out)
        if self.bidirectional and hid is not None:
            hid = self._cat_directions(hid)
        if isinstance(hid, tuple):
            hid = torch.cat(hid, 0)
        return {'predictions': rnn_out, 'hidden': hid}
    model.temporal_model[0].forward = types.MethodType(new_forward,
                                                       model.temporal_model[0])

    # ─── 3. wrapper: use **final time-step** logits for gradients ──────────
    class CAMWrap(nn.Module):
        def __init__(s, core, L): super().__init__(); s.core, s.L = core, L
        def forward(s, x):
            lens = s.L.repeat(x.size(0))
            was_training = s.core.training
            s.core.train()                           # allow grads
            with torch.set_grad_enabled(True):
                logit_seq = s.core(x, lens)['sequence_logits'][0]  # [T,B,C]
                if logit_seq.dim() == 3:
                    scores = logit_seq[-1]           # take last time-step
                else:
                    scores = logit_seq               # already [B,C]
            s.core.train(was_training)
            return scores
    cam_model = CAMWrap(model, len_x.to(dev)).to(dev)

    # ─── 4. choose layers to visualise ────────────────────────────────────
    layers = [
        ('stem_fast', model.conv2d.s1.pathway1_stem),
        ('res3_slow', model.conv2d.s3.pathway0_res1.branch2.c),
        ('res5_fast', model.conv2d.s5.pathway1_res2.branch2.c),
        ('bilstm0',   model.temporal_model[0].rnn),
    ]

    with torch.no_grad():
        top_id = int(cam_model(inp)[0].argmax())
    print(f'\nTop-1 predicted gloss: {top_id} '
          f'({gloss_dict.get(top_id, "<unk>")})\n')

    out_root = Path('slowfast/work_dir/gradcam_per_layer')
    alpha    = 0.40                               # overlay opacity

    for lname, layer in layers:
        print(f'→ Running Grad-CAM on [{lname}] …')
        cam = GradCAMPlusPlus(cam_model, [layer], reshape_transform=reshape)
        g = cam(inp, targets=[ClassifierOutputTarget(top_id)],
                eigen_smooth=True, aug_smooth=False)      # (N,Hₘ,Wₘ)
        
        # ── NORMALISE CAM SHAPES ───────────────────────────────────────────
        if g.ndim == 2:                 # (N,P)  →  (N,1,1)
            g = g[:, None, None]

        elif g.ndim == 4:               # (N,C,H,W)  →  (N,H,W)
            g = g.mean(1)               # average over channels
        # now g is always (N, Hm, Wm)
        # ───────────────────────────────────────────────────────────────────

        # g.shape = (B*T, Hl, Wl); recover video length T
        BT = g.shape[0]
        Hl, Wl = g.shape[-2:]
        B          = inp.shape[0]
        T_heat     = BT // B

        # pick T_heat frames uniformly across original video
        vid_len   = frames.shape[0]
        idxs      = np.linspace(0, vid_len - 1, T_heat).round().astype(int)

        save_dir  = out_root / lname
        save_dir.mkdir(parents=True, exist_ok=True)

        for i_hm, (hm, fi) in enumerate(zip(g, idxs)):
            hm = cv2.resize(hm.astype(np.float32), (frames.shape[2], frames.shape[1]))
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-7)      # per-map normalise
            vis = show_cam_on_image(denorm(frames[fi]), hm,
                                    use_rgb=True, image_weight=1 - alpha)
            cv2.putText(vis, gloss_dict.get(top_id, "<unk>"), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        1, cv2.LINE_AA)
            cv2.imwrite(str(save_dir / f'frame{i_hm:03d}.jpg'), vis)

        print(f'   saved {T_heat} maps → {save_dir}')

    print('\nFinished.  See results in  slowfast/work_dir/gradcam_per_layer/')
# ──────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
