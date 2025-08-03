import os
import glob
import cv2
import torch
import numpy as np
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
    print(f"[Reshape] Transforming tensor shape \
    {tensor.shape}") if tensor is not None else None 
    dims = tensor.dim()
    if dims == 5:
        b, c, t, h, w = tensor.size()
        return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    elif dims == 3:
        b, c, t = tensor.size()
        return tensor.permute(0, 2, 1).reshape(b * t, c, 1, 1)
    elif dims == 2:
        b, c = tensor.size()
        return tensor.reshape(b, c, 1, 1)
    else:
        flat = tensor.reshape(tensor.size(0), -1)
        return flat.unsqueeze(-1).unsqueeze(-1)



def main():
    # 1) Parse all arguments (mode, index, config, ckpt, etc.)
    args = parse_args()
    # ensure verbose exists
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

    # Wrap the model so CAM can call it with a single input
    class CAMWrapper(nn.Module):
        def __init__(self, slr_model, sample_len_x):
            super().__init__()
            self.model = slr_model
            # We will expand len_x along the batch dimension inside the forward
            self.sample_len_x = sample_len_x

        def forward(self, x):
            # x: [B, T, C, H, W] - as originally collated
            B = x.size(0)
            len_x_batch = self.sample_len_x.repeat(B)
            out = self.model(x, len_x_batch)
            # CAM needs raw logits shape [B, n_classes]
            return out['logits']  # adjust key if required
        
    cam_model = CAMWrapper(model, len_x.to(device)).to(device)

    # 4) define the layers for viz
    target_layers = [
        model.conv2d.s1.pathway0_stem,             # 🐢 Slow stream: early layer (S1)
        model.conv2d.s1.pathway1_stem,             # 🐇 Fast stream: early layer (S1)
        model.conv2d.s3.pathway0_res1.branch2.c,   # 🐢 Slow stream: mid-depth layer (S3)
        model.conv2d.s3.pathway1_res1.branch2.c,   # 🐇 Fast stream: mid-depth layer (S3)
        model.conv2d.s5.pathway0_res2.branch2.c,   # 🐢 Slow stream: deep layer (S5)
        model.conv2d.s5.pathway1_res2.branch2.c,   # 🐇 Fast stream: deep layer (S5)
        model.conv1d.main_temporal_conv[4],        # 🧠 SlowFast-fused temporal convolution
        model.temporal_model[0].rnn,               # 🧠 BiLSTM layer after temporal fusion
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
    frames = load_frames_from_info(info[0], args, verbose=verbose)  # [T, H, W, 3]

    # 7) prepare the input tensor: use original `data` shape [B, T, C, H, W]
    if verbose: print("[Main] Preparing input tensor and moving to device")
    inp = data.to(device)

    # 8) compute grayscale CAMs: returns [B*T, H, W]
    if verbose: print("[Main] Generating Grad-CAM heatmaps...")
    grayscale_cams = cam(
        input_tensor=inp,            # [B, T, C, H, W] on correct device
        eigen_smooth=True,           # apply PCA-based smoothing
        aug_smooth=False             # disable augmentation-based smoothing
    )

    # 9) reshape back to [B, T, H, W]
    b, t_h, _, h, w = inp.size()      # inp: [B, T, C, H, W]
    # note: after reshape_transform, grayscale_cams shape [B*T, H, W]
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

    # 11) Optionally, print single-sample WER to correlate
    if verbose: print("[Main] Computing single-sample WER for correlation")
    wer, decoded, info = evaluate_single_sample(model, feeder, args, str(out_base))
    print(f"Sample WER: {wer:.2f}% | Video ID: {info[0].split('|')[0]}")


if __name__ == "__main__":
    main()
