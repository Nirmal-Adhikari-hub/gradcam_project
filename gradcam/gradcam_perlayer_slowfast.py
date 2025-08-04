# ───────── gradcam_perlayer_slowfast.py ──────────────────────────────────
"""
Grad-CAM++ visualisation for four representative SlowFast layers.

• We pick the *last* frame-step in the SLR sequence whose arg-max gloss
  is **not** the <unk> id (0).  That gloss drives the CAM.
• Overlays are written to
      slowfast/work_dir/gradcam_per_layer/<layer>/frameXXX.jpg
"""

import sys, glob, cv2, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence

# ── make pytorch-grad-cam resize tolerant of odd inputs ─────────────────
import pytorch_grad_cam.utils.image as _im
import pytorch_grad_cam.base_cam    as _bc
_orig = _im.scale_cam_image
def _safe(img, tgt=None):
    if img.ndim == 3: img = img.mean(0)
    if img.ndim == 1: img = img[None]
    if isinstance(tgt,(list,tuple)) and len(tgt) >= 2:
        tgt = (int(tgt[-1]), int(tgt[-2]))          # (W,H) for cv2
    return _orig(img, tgt)
_im.scale_cam_image = _safe
_bc.scale_cam_image = _safe
# ─────────────────────────────────────────────────────────────────────────

sys.path.append(str(Path(__file__).resolve().parent))
from slowfast_setup import parse_args, load_dataset, load_model
from pytorch_grad_cam               import GradCAMPlusPlus
from pytorch_grad_cam.utils.image   import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MEAN = np.array([0.485,0.456,0.406])[None,None]
STD  = np.array([0.229,0.224,0.225])[None,None]
denorm = lambda x:(x*STD+MEAN).clip(0,1)

def reshape(t):
    if isinstance(t,tuple):              t = t[0]
    if isinstance(t,PackedSequence):     t = t.data
    if t.ndim == 5:                      # [B,C,T,H,W] → [B*T,C,H,W]
        B,C,T,H,W = t.shape
        return t.permute(0,2,1,3,4).reshape(B*T, C, H, W)
    if t.ndim == 3:                      # [B,C,T] → [B*T,C,1,1]
        B,C,T = t.shape
        return t.permute(0,2,1).reshape(B*T, C, 1, 1)
    if t.ndim == 2:                      # [B,C]   → [B*C,1,1]
        B,C = t.shape
        return t.view(B*C,1,1)
    return t.flatten().view(-1,1,1)

def load_frames(info, args):
    rel   = info.split('|')[1]
    root  = Path(args.dataset_info['dataset_root'])
    patt  = root/'features'/f"fullFrame-256x256px/{args.mode}"/rel
    paths = sorted(glob.glob(str(patt)))
    return np.stack([cv2.imread(p)[:,:,::-1].astype(np.float32)/255
                     for p in paths])

# ─────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gloss_dict = np.load(args.dataset_info['dict_path'],
                         allow_pickle=True).item()
    feeder = load_dataset(args=args, gloss_dict=gloss_dict)
    model  = load_model(args=args, gloss_dict=gloss_dict,
                        checkpoint_path=args.slowfast_ckpt).to(dev)

    data, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])
    frames = load_frames(info[0], args)          # raw RGB [N,H,W,3]
    inp    = data.to(dev)                        # (B,C,T,H,W)

    # ── 1. patch first BiLSTM so its .rnn may return tensor only ──────────
    class ROut(nn.Module):
        def __init__(s, r): super().__init__(); s.r = r
        def forward(s, x, h=None):
            p,_ = s.r(x, h)
            pad,_ = rnn_utils.pad_packed_sequence(p, batch_first=False)
            return pad                           # [T,B,C]
    model.temporal_model[0].rnn = ROut(model.temporal_model[0].rnn)

    # robust forward for that layer ---------------------------------------
    import types
    def patch_bilstm_forward(self, src, lens, hidden=None):
        packed = rnn_utils.pack_padded_sequence(src, lens)
        out = self.rnn(packed, hidden)
        # accept 3 possible return types
        if isinstance(out, torch.Tensor):
            packed_out, hidden = out, None
        elif isinstance(out, rnn_utils.PackedSequence):
            packed_out, hidden = out, None
        else:
            packed_out, hidden = out
        if isinstance(packed_out, rnn_utils.PackedSequence):
            preds,_ = rnn_utils.pad_packed_sequence(packed_out)
        else:
            preds = packed_out
        return {"predictions": preds, "hidden": hidden}
    model.temporal_model[0].forward = types.MethodType(
        patch_bilstm_forward, model.temporal_model[0])

    # ── 2. quick forward pass WITH grads enabled (model must be train()) ──
    was_training = model.training
    model.train()                                  # CuDNN RNN backward req.
    out = model(inp, len_x)['sequence_logits'][0].detach()  # [T,B,C]
    model.train(was_training)                      # restore original mode

    per_frame_ids = out.argmax(-1)[:,0]
    valid_ts = (per_frame_ids != 0).nonzero(as_tuple=False)
    t_sel = int(valid_ts[-1]) if len(valid_ts) else out.size(0)-1
    scores = out[t_sel].clone().requires_grad_(True)   # [B=1,C]
    top_id = int(scores.argmax())

    print(f"\nVideo file  : {info[0].split('|')[0]}")
    print(f"Raw frames  : {frames.shape[0]}")
    print(f"Picked t    : {t_sel}   (raw-frame ≈ "
          f"{round(t_sel*frames.shape[0]/out.size(0))})")
    probs = scores.softmax(-1)[0]
    k=5
    tk_p, tk_i = torch.topk(probs, k)
    print("Top-5       :", [(int(i), float(p)) for p,i in zip(tk_p, tk_i)])

    # ── 3. wrapper that recomputes logits (so grads flow) ────────────────
    class CAMWrap(nn.Module):
        def __init__(s, core, t): super().__init__(); s.core, s.t = core, t
        def forward(s, x):
            state = s.core.training
            s.core.train()                               # enable CuDNN bwd
            logits = s.core(x, len_x)['sequence_logits'][0]  # [T,B,C]
            s.core.train(state)
            return logits[s.t]                           # [B,C]
    cam_model = CAMWrap(model, t_sel).to(dev)

    layers = [
        ('stem_fast', model.conv2d.s1.pathway1_stem),
        ('res3_slow', model.conv2d.s3.pathway0_res1.branch2.c),
        ('res5_fast', model.conv2d.s5.pathway1_res2.branch2.c),
        ('bilstm0',   model.temporal_model[0].rnn),
    ]

    out_root = Path('slowfast/work_dir/gradcam_per_layer')
    alpha    = 0.40

    for lname, layer in layers:
        print(f"→ Running Grad-CAM on [{lname}] …")
        cam = GradCAMPlusPlus(cam_model, [layer], reshape_transform=reshape)
        g = cam(inp, targets=[ClassifierOutputTarget(top_id)],
                eigen_smooth=True, aug_smooth=False)          # (N,Hm,Wm)

        if g.ndim == 4: g = g.mean(1)                         # squeeze C
        if g.ndim == 2: g = g[:,None,None]

        BT = g.shape[0]
        idxs = np.linspace(0, frames.shape[0]-1, BT).round().astype(int)
        save_dir = out_root/lname
        save_dir.mkdir(parents=True, exist_ok=True)

        for i,(hm,fi) in enumerate(zip(g, idxs)):
            hm = cv2.resize(hm.astype(np.float32),
                            (frames.shape[2], frames.shape[1]))
            hm = (hm-hm.min())/(hm.max()-hm.min()+1e-7)
            vis = show_cam_on_image(denorm(frames[fi]), hm,
                                    use_rgb=True, image_weight=1-alpha)
            cv2.putText(vis, f"{gloss_dict.get(top_id,'<unk>')}  t={fi}",
                        (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(save_dir/f'frame{i:03d}.jpg'), vis)
        print(f"   saved {BT} maps → {save_dir}")

    print("\nFinished.  See results in  slowfast/work_dir/gradcam_per_layer/")

# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
