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


def load_yaml(path: str):
    with open(path, 'r') as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            return yaml.load(f)


def load_dataset(args, gloss_dict):
    # Load dataset info from YAML
    feeder = BaseFeeder(
        prefix=args.dataset_info['dataset_root'],
        gloss_dict=gloss_dict,
        dataset=args.dataset,
        datatype=args.feeder_args['datatype'],
        kernel_size=['K5','P2','K5','P2'],
        transform_mode=False,
        mode=args.mode,
    )

    #### Feeder Indices [0,1,2,62,414] -> Test
    #### Feeder Indices [0,68,5670,2275] -> Train
    #### Feeder Indices [0,21,539,107] -> Dev
    return feeder


def load_model(args, gloss_dict, checkpoint_path: str):
    model = SLRModel(
        num_classes=args.model_args['num_classes'],
        c2d_type=args.model_args['c2d_type'],
        conv_type=args.model_args['conv_type'],
        load_pkl=None,
        slowfast_config=args.slowfast_config,
        slowfast_args=[],
        gloss_dict=gloss_dict
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state['model_state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("✅ Trained model weights loaded.")
    return model


def evaluate_single_sample(model, feeder, args, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Collate one sample
    data, len_x, label, label_len, info = feeder.collate_fn([feeder[args.index]])

    # Forward pass and decode
    with torch.no_grad():
        output = model(data, len_x)
        decoded = output['recognized_sents']

    # Write hypothesis CTM
    video_id = info[0].split('|')[0]
    hyp_path = f"{out_dir}/hyp-{args.mode}.ctm"
    write2file(hyp_path, [video_id], decoded)

    # Filter STM to this video
    stm_path = f"{args.dataset_info['evaluation_dir']}/{args.dataset_info['evaluation_prefix']}-{args.mode}.stm"
    filtered_stm = f"{out_dir}/stm-{args.mode}.stm"
    filter_stm_for_predicted_ctm(stm_path, hyp_path, filtered_stm)

    # Copy CTM for WER
    filtered_ctm = f"{out_dir}/ctm-{args.mode}.ctm"
    os.system(f"cp {hyp_path} {filtered_ctm}")

    # Compute WER
    wer = wer_calculation(gt_path=filtered_stm, primary_pred=filtered_ctm)
    return wer, decoded, info

def parse_args():
    # Parse arguments
    sparser = utils.get_parser()
    sparser.add_argument('--mode', type=str, default='test',
                        help='Dataset split to use (train/dev/test)')
    sparser.add_argument('--index', type=int, default=0,
                        help='Index of the sample in the feeder to evaluate')
    sparser.add_argument('--verbose', action='store_true',
                        help='Print progress messages')
    sparser.add_argument('--slowfast_ckpt', type=str,
                        default='/shared/home/xvoice/nirmal/gradcam/checkpoints/slow_fast_phoenix2014_dev_18.01_test_18.28.pt')
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
    return args



def main(mode: str, index: int):
    args = parse_args()

    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()

    # Update args with explicit mode and index
    args.mode = mode
    args.index = index

    # Load feeder and dataset info
    feeder = load_dataset(args, gloss_dict)
    # Determine checkpoint path
    ckpt = args.slowfast_ckpt
    # Load model
    model = load_model(args, gloss_dict,ckpt)

    # Evaluate WER for single sample
    out_dir = './slowfast/work_dir/gradcam_eval'
    wer, decoded, info = evaluate_single_sample(model, feeder, args, out_dir)
    print(f"Single-sample WER: {wer:.2f}% | Info: {info}")


if __name__ == '__main__':
    # Default mode and index can be overridden here or via CLI flags
    #### Feeder Indices [0,1,2,62,414] -> Test
    #### Feeder Indices [0,68,5670,2275] -> Train
    #### Feeder Indices [0,21,539,107] -> Dev
    main(mode='test', index=2)
