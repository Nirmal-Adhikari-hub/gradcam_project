import numpy as np
import pytorch_grad_cam.utils.image as _cam_image_utils
import pytorch_grad_cam.base_cam    as _base_cam

# keep a reference to the original
_original_scale = _cam_image_utils.scale_cam_image

def _safe_scale_cam_image(cam: np.ndarray, target_size):
    # --- collapse any extra cam dims to 2D as before ---
    if cam.ndim == 3:     cam = cam.mean(axis=0)
    if cam.ndim == 1:     cam = cam[np.newaxis, :]

    # --- now sanitize the target_size for cv2.resize ---
    # Accept tuples/lists of length >=2; drop all but last two
    if isinstance(target_size, (tuple, list)) and len(target_size) >= 2:
        tsize = (int(target_size[-1]), int(target_size[-2]))  # (W, H) order for cv2
    else:
        # fallback, assume it's already (W,H)
        tsize = target_size

    # delegate to the original
    return _original_scale(cam, tsize)

# override in both modules
_cam_image_utils.scale_cam_image = _safe_scale_cam_image
_base_cam.scale_cam_image    = _safe_scale_cam_image


import os
import glob
import cv2
import torch
import yaml
from pathlib import Path
import torch.nn as nn


# Ensure project root on PYTHONPATH so we can import the setup files
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from slowfast import utils
from slowfast_setup import parse_args, load_dataset, load_model, evaluate_single_sample

# pytorch-grad-cam imports
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.nn.utils.rnn import PackedSequence

imagenet_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
imagenet_std  = np.array([0.229, 0.224, 0.225])[None, None, :]

def denormalize_image(img_norm):
    # img_norm has shape [H,W,3] in normalized space
    img = img_norm * imagenet_std + imagenet_mean
    return np.clip(img, 0, 1)

def load_frames_from_info(info_str, args, verbose=False):
    """
    Read the raw RGB frames for a sample.
    Expects info_str like '...|/path/to/frames/*.png|...
    Returns a numpy array of shape [T, H, W, 3], values in [0,1].
    """
    rel_pattern = info_str.split('|')[1]
    print(f"[Frames] Loading frames from pattern: {rel_pattern}") if verbose else None

    # Build the full path under features directory
    root = args.dataset_info['dataset_root']
    feat_dir = os.path.join(root, 'features',
                            f"fullFrame-256x256px/{args.mode}")
    frame_pattern = os.path.join(feat_dir, rel_pattern)
    if verbose: print(f"[Frames] Glob pattern: {frame_pattern}")
    paths = sorted(glob.glob(frame_pattern))
    if verbose: print(f"[Frames] Found {len(paths)} frames")
    frames = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        frames.append(img)
        if verbose and (i + 1) % 20 == 0:
            print(f"  → Loaded {i+1}/{len(paths)} frames")
    return np.stack(frames, axis=0)

def reshape_transform(tensor):
    # 1) If it's a tuple (e.g. (output, hidden)), unwrap
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    # 2) If it's a PackedSequence, grab its .data tensor
    if isinstance(tensor, PackedSequence):
        tensor = tensor.data
    # 3) Now it's guaranteed a Tensor
    print(f"[Reshape] Transforming tensor shape {tensor.shape}") if tensor is not None else None
    dims = tensor.dim()
    if dims == 5:
        b, c, t, h, w = tensor.size()
        return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    elif dims == 3:
        b, c, t = tensor.size()
        return tensor.permute(0, 2, 1).reshape(b * t, c, 1, 1)
    elif dims == 2:
        # collapse [B, C] → [B*C, 1, 1] so we end up with a true 2D heatmap per channel
        b, c = tensor.size()
        return tensor.view(b * c, 1, 1)
    else:
        # 1D or unknown dims – make it 2D
        # e.g. [C] → [C, 1, 1]
        flat = tensor.flatten()
        return flat.view(flat.size(0), 1, 1)

def main():
    # 1) Parse all arguments (mode, index, config, ckpt, etc.)
    args = parse_args()
    args.verbose = getattr(args, 'verbose', False)
    verbose = args.verbose
    if verbose: print("[Main] Arguments:", args)

    # 2) load dataset and model using setup helpers
    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()
    if verbose: print("[Main] Loading dataset feeder...")
    feeder = load_dataset(args=args, gloss_dict=gloss_dict)

    if verbose: print(f"[Main] Loading model from checkpoint {args.slowfast_ckpt}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args=args, gloss_dict=gloss_dict, checkpoint_path=args.slowfast_ckpt)
    model = model.to(device)

    # 3) grab one sample and its raw frames
    if verbose: print(f"[Main] Collating sample index {args.index} (split={args.mode})")
    data, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])

    # --- add after you load the model (just before you create the RNN wrapper) ----
    import types, torch.nn.utils.rnn as rnn_utils

    def _bilstm_forward_patched(self, src_feats, src_lens, hidden=None):
        """
        Drop-in replacement for BiLSTMLayer.forward that copes with an RNN
        returning *either* (packed_outputs, hidden)  **or**  packed_outputs.
        """
        packed = rnn_utils.pack_padded_sequence(src_feats, src_lens)

        result = self.rnn(packed, hidden)        # may be tuple or tensor

        # ------------------------------------------------------------------
        # Normal run (tuple)          ‖  Wrapped run (PackedSequence only)
        # ------------------------------------------------------------------
        if isinstance(result, tuple):
            packed_outputs, hidden = result
        else:
            packed_outputs, hidden = result, None          # nothing to unpack

        # back to padded - torch.Tensor  → grads can flow & Grad-CAM can hook
        rnn_out, _ = rnn_utils.pad_packed_sequence(packed_outputs)   # [T,B,C]

        # keep the rest of the original bookkeeping -----------------------
        if self.bidirectional and hidden is not None:
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)

        return {"predictions": rnn_out, "hidden": hidden}

    # monkey-patch the very first BiLSTM layer only ------------------------
    model.temporal_model[0].forward = types.MethodType(
        _bilstm_forward_patched, model.temporal_model[0]
    )


    # Wrap the model so CAM can call it with a single input
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

    # ────────────────────────────────────────────────────────────────────────
    #  Add this wrapper so your LSTM returns only its activation Tensor,
    #  not the (output, hidden_state) tuple that Grad-CAM can’t handle.
    class RNNOutputOnly(nn.Module):
        def __init__(self, rnn):
            super().__init__()
            self.rnn = rnn
        def forward(self, x, hx=None):
            packed_out, _ = self.rnn(x, hx)           # original LSTM output
            # convert to *padded* tensor so Grad-CAM sees a real Tensor
            padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=False
            )                                         # [T,B,C]
            return padded

    # Replace the raw LSTM in model.temporal_model[0] with our wrapper
    wrapped_rnn = RNNOutputOnly(model.temporal_model[0].rnn)
    model.temporal_model[0].rnn = wrapped_rnn
    # ────────────────────────────────────────────────────────────────────────


    # 4) define the layers for viz
    target_layers = [
        model.conv2d.s1.pathway0_stem,
        model.conv2d.s1.pathway1_stem,
        model.conv2d.s3.pathway0_res1.branch2.c,
        model.conv2d.s3.pathway1_res1.branch2.c,
        model.conv2d.s5.pathway0_res2.branch2.c,
        model.conv2d.s5.pathway1_res2.branch2.c,
        model.conv1d.main_temporal_conv[4],
        # model.temporal_model[0].rnn,
        wrapped_rnn, 
    ]
    if verbose: print(f"[Main] {len(target_layers)} target layers set up for Grad-CAM")

    # 5) instantiate a Grad-CAM variant
    if verbose: print("[Main] Instantiating GradCAMPlusPlus...")
    cam = GradCAMPlusPlus(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # 6) load raw frames
    frames = load_frames_from_info(info[0], args, verbose=verbose)

    # 7) prepare the input tensor
    if verbose: print("[Main] Preparing input tensor and moving to device")
    inp = data.to(device)

    # 8) compute grayscale CAMs
    if verbose: print("[Main] Generating Grad-CAM heatmaps...")
    for layer in target_layers:
        layer.register_forward_hook(lambda m, i, o: print(f"[HOOK] {m.__class__.__name__} → {getattr(o, 'shape', type(o))}"))
    
    # --- 8a) run a forward to get the predicted class index ---
    #    We call cam_model (the wrapper) directly, since it returns [B, C] scores
    scores = cam_model(inp)              # shape [B, num_classes]
    pred_class = int(scores[0].argmax()) # pick the top class for sample 0

    # wrap that into a target for Grad-CAM
    targets = [ClassifierOutputTarget(pred_class)]

    # now call cam with that target
    grayscale_cams = cam(
        input_tensor=inp,
        targets=targets,
        eigen_smooth=True,
        aug_smooth=False
    )

    # 9) reshape back to [B, T, H, W]
    b, t_h, _, h, w = inp.size()
    cams = grayscale_cams.reshape(b, t_h, h, w)

    # 10) overlay and save per-layer per-frame
    out_base = Path('./slowfast/work_dir/gradcam_eval')
    for idx, _ in enumerate(target_layers):
        layer_dir = out_base / f'layer{idx}'
        layer_dir.mkdir(parents=True, exist_ok=True)
        if verbose: print(f"[Main] Saving layer {idx} heatmaps to {layer_dir}")
        for frame_i in range(t_h):
            heatmap = cams[0, frame_i]
            rgb = frames[frame_i]
            rgb_den = denormalize_image(rgb)
            cam_img = show_cam_on_image(rgb_den, heatmap, use_rgb=True)
            cv2.imwrite(str(layer_dir / f'frame{frame_i:03d}.jpg'), cam_img)
            if verbose and frame_i % 20 == 0:
                print(f"    → Saved frame {frame_i}/{t_h} for layer {idx}")

    # 11) print single-sample WER
    if verbose: print("[Main] Computing single-sample WER for correlation")
    wer, decoded, info = evaluate_single_sample(model, feeder, args, str(out_base))
    print(f"Sample WER: {wer:.2f}% | Video ID: {info[0].split('|')[0]}")

if __name__ == "__main__":
    main()
