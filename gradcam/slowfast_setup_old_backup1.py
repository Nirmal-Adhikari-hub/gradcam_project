import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in PYTHONPATH for slowfast imports
sys.path.append(str(Path(__file__).resolve().parent))

from slowfast.slr_network_multi import SLRModel
from slowfast.evaluation.slr_eval.wer_calculation import wer_calculation, filter_stm_for_predicted_ctm
from slowfast.seq_scripts import write2file
from slowfast.dataset.dataloader_video import BaseFeeder
from slowfast import utils



mode = 'test'
slowfast_ckpt = '/home/nirmal/SlowFast/GradCAMs/checkpoints/slow_fast_phoenix2014_dev_18.01_test_18.28.pt'



if __name__ == "__main__":
    sparser = utils.get_parser()
    p = sparser.parse_args()

    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./slowfast/configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # print(f"Arguments: {args}")

    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()

    model = SLRModel(
        num_classes=args.model_args['num_classes'], 
        c2d_type=args.model_args['c2d_type'], 
        conv_type=args.model_args['conv_type'], 
        load_pkl=None, 
        slowfast_config=args.slowfast_config,
        slowfast_args=[],
        gloss_dict=gloss_dict,
    )

    # Load model weights
    state_dict = torch.load(slowfast_ckpt, map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict["model_state_dict"].items()}
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ Trained model weights loaded.")

    feeder = BaseFeeder(
        prefix=args.dataset_info['dataset_root'],
        gloss_dict=gloss_dict,
        dataset=args.dataset,
        datatype=args.feeder_args['datatype'],
        kernel_size=['K5','P2','K5','P2'],
        transform_mode=False,
        mode=mode
    )

    #### Feeder Indices [0,1,2,62,414] -> Test
    #### Feeder Indices [0,68,5670,2275] -> Train
    #### Feeder Indices [0,21,539,107] -> Dev
    data, len_x, label, label_len, info = feeder.collate_fn([feeder[2]])
    print("data shape before permute:", data.shape)



    model.eval()
    with torch.no_grad():
        output = model(data, len_x)
        decoded = output["recognized_sents"]  # list of [(gloss, timestep), ...]
        # print("\n🧪 Decoded glosses:", decoded)
        # print(f"Labels: {label}, Label Lengths: {label_len}, len_x: {len_x}, Info: {info}")
        pred_glosses = [g for g, _ in decoded[0]]

        # print("\n🧪 Decoded glosses:", pred_glosses)

    # Save predicted CTM
    out_dir = f"./slowfast/work_dir/gradcam_eval"
    os.makedirs(out_dir, exist_ok=True)
    hyp_path = f"{out_dir}/output-hypothesis-{mode}.ctm"
    # Extract clean video ID
    video_id = info[0].split("|")[0]
    write2file(hyp_path, [video_id], decoded)

    # Filter STM file to include only that video
    stm_path = f"{args.dataset_info['evaluation_dir']}/{args.dataset_info['evaluation_prefix']}-{mode}.stm"
    filtered_stm = f"{out_dir}/filtered-{mode}.stm"
    filter_stm_for_predicted_ctm(
        stm_path=stm_path,
        ctm_path=hyp_path,
        filtered_stm_out_path=filtered_stm
    )

    # Evaluate WER only on this sample
    filtered_ctm = f"{out_dir}/out.output-hypothesis-{mode}.ctm"
    os.system(f"cp {hyp_path} {filtered_ctm}")

    wer_result = wer_calculation(gt_path=filtered_stm, primary_pred=filtered_ctm)
    print("\n✅ Final WER (single sample):", round(wer_result, 2), "%")


