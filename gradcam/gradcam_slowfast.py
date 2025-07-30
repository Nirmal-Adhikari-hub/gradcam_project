import os, sys, pprint
from pathlib import Path
import numpy as np
import torch

mode = 'dev'
slowfast_ckpt = '/home/nirmal/SlowFast/GradCAMs/checkpoints/slow_fast_phoenix2014_dev_18.01_test_18.28.pt'
# yaml_path = "SLOWFAST_64x2_R101_50_50.yaml"
# gloss_dict = np.load('/home/nirmal/SlowFast/gradcam_project/slowfast/preprocess/phoenix2014/gloss_dict.npy', allow_pickle=True).item()
# inv_gloss_dict = {v[0]: k for k, v in gloss_dict.items()}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slowfast.slr_network_multi import SLRModel
from slowfast.evaluation.slr_eval.wer_calculation import evaluate
from slowfast.seq_scripts import write2file


if __name__ == "__main__":
    from slowfast.dataset.dataloader_video import BaseFeeder
    from slowfast import utils
    import yaml

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

    print(f"Arguments: {args}")

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
    data, len_x, label, label_len, info = feeder.collate_fn([feeder[21]])
    print("data shape before permute:", data.shape)

    # Load model weights
    state_dict = torch.load(slowfast_ckpt, map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict["model_state_dict"].items()}
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ Trained model weights loaded.")

    model.eval()
    with torch.no_grad():
        output = model(data, len_x)

        # print(output["sequence_logits"][0].shape)
        print(f"Output INFO: {output.keys()}")

        logits = output["sequence_logits"][0]   # [T, 1, V]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # print("\n🧪 Probabilities shape:", probs.shape, probs)  # [T, 1, V]
        pred_ids = probs.argmax(dim=-1).squeeze(1)  # [T]

        # # 🧪 DEBUG: Print raw prediction before decoding
        print("\n🧪 Predicted token IDs:", pred_ids)  # [T]

        # print(gloss_dict, inv_gloss_dict)
        # print(f"\nLENGTHS: gloss_dict {len(gloss_dict)}, inv_gloss_dict {len(inv_gloss_dict)}")

        # # print("\n🧪 Corresponding glosses:")
        # raw_glosses = [inv_gloss_dict.get(idx.item(), f"<UNK-{idx.item()}>") for idx in pred_ids]
        # print(raw_glosses)

        decoded = output["recognized_sents"]  # list of [(gloss, timestep), ...]
        print("\n🧪 Decoded glosses:", decoded)
        print(f"Labels: {label}, Label Lengths: {label_len}, len_x: {len_x}, Info: {info}")

        # 🧪 DEBUG: Decoded glosses after CTC collapse
        print("\n🧪 Decoded collapsed glosses:")
        print([g for g, _ in decoded[0]])

        # Prepare video ID
        info = [file_name.split("|")[0] for file_name in info]

        # Save decoded prediction
        hyp_path = f"./slowfast/work_dir/gradcam_eval/output-hypothesis-{mode}.ctm"
        write2file(hyp_path, info, decoded)

        ############## Evaluate WER ##############
        print("##################################")
        print("num_classes (model):", args.model_args['num_classes'])
        print("len(gloss_dict):", len(gloss_dict))
        print("\n🔍 WRITING TO CTM:")
        print("Video ID:", info[0])
        print("Predicted glosses:", [g for g, _ in decoded[0]])

        stm_path = f"{args.dataset_info['evaluation_dir']}/{args.dataset_info['evaluation_prefix']}-{mode}.stm"
        with open(stm_path, "r") as f:
            stm_lines = [line for line in f if info[0] in line]
        print("Reference STM Line:")
        print("\n".join(stm_lines))
        ##########################################

        # Run WER
        wer_result = evaluate(
            prefix = "./slowfast/work_dir/gradcam_eval/",
            mode = mode,
            output_file = f"output-hypothesis-{mode}.ctm",
            evaluate_dir = "./slowfast/evaluation/slr_eval/",
            evaluate_prefix = "phoenix2014-groundtruth",
            output_dir = 'tmp_eval_result/',
            python_evaluate = True,
            triplet = False,
        )

        print("\n✅ Final WER:", wer_result, "%")



        


    # print("Output shape:", output)
    # keys = [key for key in output.keys()]
    # print("Output keys:", keys)
    # print(f"{keys[0]} - {output[keys[0]]}")  # feat_len
    # print(f"{keys[1]} - {output[keys[1]][0].shape}")  
    # print(f"{keys[2]} - {output[keys[2]][0].shape}")  
    # print(f"{keys[3]} - {output[keys[3]]}")
    # print(f"{keys[4]} - {output[keys[4]]}")
    
    ########## without model.eval() ##########
    # with torch.no_grad():
    #     output = model(data, len_x)
    # print(f"{keys[0]} - {output[keys[0]].shape}")

    # print(f"{keys[1]} - {output[keys[1]][0].shape}")
    # print(f"{keys[1]} - {output[keys[1]][1].shape}")
    # print(f"{keys[1]} - {output[keys[1]][2].shape}")

    # print(f"{keys[2]} - {output[keys[2]][0].shape}")
    # print(f"{keys[2]} - {output[keys[2]][1].shape}")
    # print(f"{keys[2]} - {output[keys[2]][2].shape}")

    # print(f"{keys[3]} - {output[keys[3]]}")
    # print(f"{keys[4]} - {output[keys[4]]}")

    ''' 
    ########### Output example ###########
    feat_len - torch.Size([1])

    conv_logits - torch.Size([33, 1, 1296])
    conv_logits - torch.Size([33, 1, 1296])
    conv_logits - torch.Size([33, 1, 1296])

    sequence_logits - torch.Size([33, 1, 1296])
    sequence_logits - torch.Size([33, 1, 1296])
    sequence_logits - torch.Size([33, 1, 1296])

    conv_sents - None
    recognized_sents - None
    '''

    ## Main Grad-CAM code
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


