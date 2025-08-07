import torch
import os, sys
import numpy as np
from pathlib import Path
import torch.nn as nn

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM

from PIL import Image
from PIL import ImageDraw
from pathlib import Path


BaseCAM.get_target_width_height = lambda self, x: (x.size(-1), x.size(-2))  # Fix for Grad-CAM to work with odd-sized inputs    

sys.path.append(str(Path(__file__).resolve().parent))

from slowfast_setup import parse_args, load_dataset, load_model, evaluate_single_sample, load_frames_from_info


def reshape_transform(tensor):
    """
    Reshape tensor to [B*T, C, H, W] if it is in [B, C, T, H, W] format.
    """
    # Grad-CAM sees [B,C,T,H,W] activations
    B, C, T, H, W = tensor.shape
    tensor = tensor.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    return tensor.reshape(B * T, C, H, W)  # [B*T, C, H, W]


# --- CORRECTED SAFE UPSAMPLE ---
def upsample_cam(cam, size):
    denom = cam.max() - cam.min()
    if denom == 0:
        cam = np.zeros_like(cam, dtype=np.float32)
    else:
        cam = (cam - cam.min()) / denom #Normalize to [0,1]
    cam = (cam * 255).astype(np.uint8)
    img = Image.fromarray(cam)
    img = img.resize(size, resample=Image.BILINEAR)
    upsampled = np.array(img).astype(np.float32) / 255
    return upsampled


class ZeroTarget:
    """A dummy target that returns 0 (no gradient flows through that sample)."""
    def __call__(self, model_output):
        return model_output.new_zeros(1, requires_grad=True)


class TimeStepTarget:
    def __init__(self, time_index, class_index):
        self.time_index = time_index
        self.class_index = class_index

    def __call__(self, model_output):
        # model ouput shape: [T, C]
        return model_output[self.time_index, self.class_index]
    

class CAMWrapper(nn.Module):
    """
    Wrapper for Grad-CAM to handle the model's forward pass
    """
    def __init__(self, model, len_x):
        super().__init__()
        self.model = model
        self.len_x = len_x

    def forward(self, x):
        B = x.size(0)
        len_x_batch = self.len_x.repeat(B)
        was_training = self.model.training
        self.model.train()
        with torch.backends.cudnn.flags(enabled=False):
            # Disable cudnn for reproducibility
            # This is important for Grad-CAM to work correctly
            # This is important for Grad-CAM to work correctly
            # as it relies on the gradients being computed in a specific way.
            with torch.set_grad_enabled(True):
                out = self.model(x, len_x_batch)
                seq_logits = out['sequence_logits'][0]
                # --- add these two lines ---
                if seq_logits.ndim == 3 and seq_logits.shape[1] == 1:
                    seq_logits = seq_logits.squeeze(1)  # now [T_pred, num_classes]
                # -----------------------------
        self.model.train(was_training)
        print(f"[CAMWrapper] Output shape: {seq_logits.shape}")  # [B, num_classes]
        return seq_logits

def main():
    args = parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()
    inv_gloss = {v:k for k,v in gloss_dict.items() if isinstance(v, (int, np.integer))}

    feeder = load_dataset(args=args, gloss_dict=gloss_dict)
    ckpt = args.slowfast_ckpt

    model = load_model(args=args, gloss_dict=gloss_dict, 
                       checkpoint_path=ckpt)

    out_dir = './slowfast/work_dir/gradcam_eval'
    wer, decoded, info = evaluate_single_sample(model, feeder, args, out_dir)
    print(f"Single-sample WER: {wer:.2f}% | Info: {info}") 
    video_id = info[0].split('|', 1)[0]
    
    target_layers = [
        model.conv2d.s1.pathway0_stem,
        model.conv2d.s1.pathway1_stem,
        model.conv2d.s4.pathway0_res22,
        model.conv2d.s4.pathway1_res22,
        model.conv2d.s5.pathway0_res2,
        model.conv2d.s5.pathway1_res2,
    ]

    #### HOOKS TO CHECK THE TARGET LAYERS ACTIVATIONS AND GRADIENTS SHAPE ####
    def forward_hook(name):
        def hook(module,input, output):
            print(f"[FORWARD HOOK] {name} activation shape: {output.shape}")
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            print(f"[BACKWARD HOOK] {name} gradient shape: {grad_output[0].shape}")
        return hook
    
    hooks_registered = False  # Flag to prevent re-registering
    forward_handles = []
    backward_handles = []
    ##########################################################################

    input_tensor, len_x, _, _, info = feeder.collate_fn([feeder[args.index]])
    # print(f"Input tensor shape: {input_tensor.shape}") # [B, T, C, H, W]

    # input_tensor = input_tensor.permute(0,2,1,3,4).to(device)  # [B, C, T, H, W]
    input_tensor = input_tensor.to(device)

    wrapper_model = CAMWrapper(model, len_x.to(device)).to(device)
    # wrapper_model = nn.DataParallel(wrapper_model)  # Use DataParallel if needed

    # Get the model outputs to extract per-time-step predictions
    with torch.no_grad():
        logits = wrapper_model(input_tensor)         # [T_pred,  num_classes]
        pred_classes = torch.argmax(logits, dim=-1).tolist()

    # cam = GradCAM(
    #     model=wrapper_model,
    #     target_layers=target_layers,
    #     )

    frames = load_frames_from_info(info[0], args, verbose=args.verbose)
    print(f"Frames shape: {frames.shape}")  # [N, H, W, 3]

    # grayscale_cam = cam(
    #     input_tensor=input_tensor,
    #     targets=None,
    #     )
    
    # print(f"Grayscale CAM shape: {grayscale_cam.shape}")  # [B, T, H, W]
    # print(f"Grayscale CAM : {grayscale_cam}")  # Should be float32
    # grayscale_cam = grayscale_cam[0, :]
    # print(f"Grayscale CAM shape: {grayscale_cam.shape}")  # [T, H, W]

    N = frames.shape[0]

    out_root = Path('slowfast/work_dir/gradcam_per_layer') / args.mode / video_id
    out_root.mkdir(parents=True, exist_ok=True)

    target_layer_names = [
            'pathway0_stem',
            'pathway1_stem',
            'pathway0_res22',
            'pathway1_res22',
            'pathway0_res2',
            'pathway1_res2'
    ]
    
    for layer_idx, (layer, layer_name) in enumerate(zip(target_layers, target_layer_names)):

        if not hooks_registered:
            lname = f"target_layer_{layer_idx}"
            fh = layer.register_forward_hook(forward_hook(lname))
            bh = layer.register_full_backward_hook(backward_hook(lname))
            forward_handles.append(fh)
            backward_handles.append(bh)
            hooks_registered = True  # Register hooks only once

        cam = GradCAM(model=wrapper_model, 
                      target_layers=[layer],
                      reshape_transform=reshape_transform,
                      )
        # H, W = input_tensor.shape[-2], input_tensor.shape[-1]
        # cam.target_size = (W, H)  # (width, height) — OpenCV uses width first!
        # cam.target_size = None

        print(f"Input tensor shape: {input_tensor.shape}")  # [B, C, T, H, W]

        # grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # for t_pred, class_id in enumerate(pred_classes):
            # if t_pred % 5 != 0:
            #     continue
            # gloss = inv_gloss.get(class_id, str(class_id))
            # print(f"[GradCAM] layer={layer_name} t_pred={t_pred} gloss={gloss}")

            # # build one target per time–step (length = T_pred)
            # targets = []
            # for i in range(len(pred_classes)):
            #     if i == t_pred:
            #         targets.append(ClassifierOutputTarget(class_id))  # active sample
            #     else:
            #         targets.append(ZeroTarget())                      # silent sample

            # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # print(f"Grayscale CAM shape for {layer_name}: {grayscale_cam.shape}")  # [B*T, H, W]
            # B = input_tensor.shape[0]
            # cam_array = grayscale_cam

            # if cam_array.ndim == 3:
            #     # single batch case: (T_cam, H, W) -> make it (B, T_cam, H, W)
            #     T_cam, H, W = cam_array.shape
            #     cam_array = cam_array[None] # shape -> (1, T_cam, H, W)
            # else:
            #     # general case: (B*T_cam, H, W)
            #     B_T, H, W = cam_array.shape
            #     T_cam = B_T // B
            #     cam_array = cam_array.reshape(B, T_cam, H, W)
            # print(f"[Reshaped] cam_array shape: {cam_array.shape}")  # [B, T_cam, H, W]
            # grayscale_cam = cam_array
            # print(f"[Reshaped] Grayscale CAM: {grayscale_cam.shape}")  # [B, T, H, W]
            # cam_maps_layer = grayscale_cam[0]  # Select batch 0, shape [T_cam, H, W]
            # T_cam = cam_maps_layer.shape[0]

            # # folder structure
            # target_folder = out_root / layer_name / f"time_{t_pred:02d}"
            # target_folder.mkdir(parents=True, exist_ok=True)


            # # layer_folder = out_root / layer_name
            # # layer_folder.mkdir(exist_ok=True)

            # # compute the per-layer temporal stride S once, just the T_cam
            # # len_x.item() is the original input length (e.g. 200), T_cam is the number of
            # # CAM maps (e.g.50)
            # S = len_x.item() / T_cam
            # offset = S // 2 # center of each receptive field
            # for t in range(T_cam):
            #     # map each CAM time-step t back to its "center" input frame
            #     frame_idx = int(min(N - 1, t * S + offset))
            #     rgb_img = frames[frame_idx]
            #     if rgb_img.max() > 1:
            #         rgb_img = rgb_img.astype(np.float32) / 255
            #     cam_img = upsample_cam(cam_maps_layer[t], size=(rgb_img.shape[0], rgb_img.shape[1]))
            #     visualization = show_cam_on_image(rgb_img, cam_img, use_rgb=True)

            #     # Overlay the gloss text
            #     pil = Image.fromarray(visualization)
            #     draw = ImageDraw.Draw(pil)
            #     text = f"{gloss}"
            #     draw.text((5,5), text, fill=(255,255,255))
            #     visualization = np.array(pil)

            #     save_path = target_folder / f"frame_{frame_idx:04d}_cam_{t:02d}.png"
            #     Image.fromarray(visualization).save(save_path)

            # # kill the old graph, free activations
            # del grayscale_cam
            # torch.cuda.empty_cache()

        ##########################################################################

        pred_classes = -1
        for t_pred, class_id in enumerate([pred_classes]):
            if t_pred % 20 != 0:
                continue
            gloss = inv_gloss.get(class_id, str(class_id))
            print(f"[GradCAM] layer={layer_name} t_pred={t_pred} gloss={gloss}")

            # build one target per time–step (length = T_pred)
            targets = []
            for i in range(len([pred_classes])):
                if i == t_pred:
                    targets.append(ClassifierOutputTarget(class_id))  # active sample
                else:
                    targets.append(ZeroTarget())                      # silent sample

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            print(f"Grayscale CAM shape for {layer_name}: {grayscale_cam.shape}")  # [B*T, H, W]
            B = input_tensor.shape[0]
            cam_array = grayscale_cam

            if cam_array.ndim == 3:
                # single batch case: (T_cam, H, W) -> make it (B, T_cam, H, W)
                T_cam, H, W = cam_array.shape
                cam_array = cam_array[None] # shape -> (1, T_cam, H, W)
            else:
                # general case: (B*T_cam, H, W)
                B_T, H, W = cam_array.shape
                T_cam = B_T // B
                cam_array = cam_array.reshape(B, T_cam, H, W)
            print(f"[Reshaped] cam_array shape: {cam_array.shape}")  # [B, T_cam, H, W]
            grayscale_cam = cam_array
            print(f"[Reshaped] Grayscale CAM: {grayscale_cam.shape}")  # [B, T, H, W]
            cam_maps_layer = grayscale_cam[0]  # Select batch 0, shape [T_cam, H, W]
            T_cam = cam_maps_layer.shape[0]

            # folder structure
            target_folder = out_root / layer_name / f"class_{pred_classes}" / f"time_{t_pred:02d}"
            target_folder.mkdir(parents=True, exist_ok=True)


            # layer_folder = out_root / layer_name
            # layer_folder.mkdir(exist_ok=True)

            # compute the per-layer temporal stride S once, just the T_cam
            # len_x.item() is the original input length (e.g. 200), T_cam is the number of
            # CAM maps (e.g.50)
            S = len_x.item() / T_cam
            offset = S // 2 # center of each receptive field
            for t in range(T_cam):
                # map each CAM time-step t back to its "center" input frame
                frame_idx = int(min(N - 1, t * S + offset))
                rgb_img = frames[frame_idx]
                if rgb_img.max() > 1:
                    rgb_img = rgb_img.astype(np.float32) / 255
                cam_img = upsample_cam(cam_maps_layer[t], size=(rgb_img.shape[0], rgb_img.shape[1]))
                visualization = show_cam_on_image(rgb_img, cam_img, use_rgb=True)

                # Overlay the gloss text
                pil = Image.fromarray(visualization)
                draw = ImageDraw.Draw(pil)
                text = f"{gloss}"
                draw.text((5,5), text, fill=(255,255,255))
                visualization = np.array(pil)

                save_path = target_folder / f"frame_{frame_idx:04d}_cam_{t:02d}.png"
                Image.fromarray(visualization).save(save_path)

            # kill the old graph, free activations
            del grayscale_cam
            torch.cuda.empty_cache()

        ##########################################################################


    for h in forward_handles:
        h.remove()
    for h in backward_handles:
        h.remove()

if __name__ == "__main__":
    main()
