# %%
import torch
import matplotlib
import os
import sys
from pathlib import Path

preroot_dir = Path(__file__).parents[4]
sys.path.append(str(preroot_dir))
sys.path.insert(0, str(preroot_dir / "ISPRS_HD_NET"))
from ISPRS_HD_NET.Train_HDNet import main  # type: ignore # noqa


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
matplotlib.use("tkagg")

torch.set_num_threads(16)

root_dir = Path(__file__).parents[3]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch HDNet training")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        metavar="N",
        help="number of total epochs to train",
    )
    parser.add_argument("--data-path", default=root_dir / "data/ML_training/")
    parser.add_argument("--numworkers", default=8, type=int)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--base-channel", default=48, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--read-name", default="HDNet_Inria_best")
    parser.add_argument("--save-name", default="HDNet_NOCI_0.3")
    parser.add_argument("--DataSet", default="NOCI")
    parser.add_argument("--image-folder", default="train/image")
    args = parser.parse_args()

    return args


# %%
if __name__ == "__main__":
    args = parse_args()
    dir_checkpoint = str(root_dir) + "/data/ML_model/save_weights/run_6/"
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    main(args, dir_checkpoint)
