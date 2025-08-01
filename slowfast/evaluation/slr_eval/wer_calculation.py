import os
import pdb
from .python_wer_evaluation import wer_calculation


####### NON-ORIGINAL FUNCTIONS #########
def filter_stm_for_predicted_ctm(stm_path, ctm_path, filtered_stm_out_path):
    """
    Filters the STM (ground truth) file to only include lines corresponding to the
    video IDs present in the predicted CTM file.

    Args:
        stm_path (str): Path to the full .stm file
        ctm_path (str): Path to the .ctm file that contains predicted video IDs
        filtered_stm_out_path (str): Path to save the filtered STM file
    """
    with open(ctm_path, "r") as f:
        ctm_lines = f.readlines()
    predicted_ids = {line.split()[0] for line in ctm_lines}

    with open(stm_path, "r") as f:
        stm_lines = f.readlines()
    filtered_lines = [line for line in stm_lines if line.split()[0] in predicted_ids]

    with open(filtered_stm_out_path, "w") as f:
        f.writelines(filtered_lines)
    print(f"✅ Filtered STM saved to: {filtered_stm_out_path}")
########## END OF NON-ORIGINAL FUNCTIONS #########



def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, python_evaluate=False,
             triplet=False):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if python_evaluate:
        ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", f"{prefix}out.{output_file}")
        if triplet:
            wer_calculation(
                f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm",
                f"{prefix}out.{output_file}",
                f"{prefix}out.{output_file}".replace(".ctm", "-conv.ctm")
            )
        return ret
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return float(ret.split("=")[1].split("%")[0])


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")
