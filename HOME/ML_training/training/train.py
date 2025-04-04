# %%
import torch
import matplotlib
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

preroot_dir = Path(__file__).parents[4]
sys.path.append(str(preroot_dir))
sys.path.insert(0, str(preroot_dir / "ISPRS_HD_NET"))
from ISPRS_HD_NET.Train_HDNet import main  # type: ignore # noqa


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
matplotlib.use("tkagg")

torch.set_num_threads(16)

root_dir = Path(__file__).parents[3]


def parse_args(parser, args):
    bw_str = "BW" if args.BW else "C"
    bw_str_ = "_BW" if args.BW else ""
    res = args.res

    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--data-path", default=root_dir / "data/ML_training/")
    parser.add_argument("--numworkers", default=8, type=int)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--base-channel", default=48, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--read-name", default=args.read_name)
    parser.add_argument("--save-name", default=f"HDNet_NOCI_{res}_{bw_str}")
    parser.add_argument("--DataSet", default="NOCI" + bw_str_)
    parser.add_argument("--image-folder", default=f"train{bw_str_}/image")
    parser.add_argument("--label-folder", default="train/label")
    parser.add_argument("--boundary-folder", default="boundary")
    parser.add_argument("--train_txt", default="train.txt")
    parser.add_argument("--val_txt", default="val.txt")

    args = parser.parse_args()

    return args


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pytorch HDNet training")
    parser.add_argument("-n", "--numrun", required=True, type=int)
    parser.add_argument("-r", "--res", required=False, type=float, default=0.3)
    parser.add_argument(
        "-bw", "--BW", action=argparse.BooleanOptionalAction, type=bool, default=False
    )
    parser.add_argument("-rn", "--read_name", required=False, type=str, default="")
    parser.add_argument(
        "-e",
        "--epochs",
        default=150,
        type=int,
        metavar="N",
        help="number of total epochs to train",
    )
    parser.add_argument(
        "-nr",
        "--read_num",
        required=False,
        type=int,
        help="directory number to read weights from",
    )
    parser.add_argument(
        "-pet",
        "--percentage_empty",
        required=False,
        type=float,
        help="percentage empty tiles in the tiling",
    )
    parser.add_argument(
        "-lr", "--lr", default=0.001, type=float, help="initial learning rate"
    )
    args = parser.parse_args()

    dir_checkpoint = str(root_dir) + f"/data/ML_model/save_weights/run_{args.numrun}/"
    if args.read_num > 0:
        read_dir = str(root_dir) + f"/data/ML_model/save_weights/run_{args.read_num}/"
    else:
        read_dir = dir_checkpoint
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    if args.read_name == "":
        if args.read_num > 0:
            read_name = [
                f for f in os.listdir(read_dir) if ("best" in f and "NOCI" in f)
            ][0][:-4]
        else:
            read_name = ""
    else:
        read_name = ""

    args = parse_args(parser, args)

    description = (
        f"Tile resolution: {args.res} \nBlack and White: {args.BW} \nFrom Existing:"
        f" {args.read_name} in {read_dir}\nDate: {datetime.now()}\n\n"
        f"max_epochs: {args.epochs}\n, learning rate: {args.lr}\n"
    )
    description += (
        f"percentage empty tiles: {args.percentage_empty}"
        if args.percentage_empty
        else ""
    )
    with open(dir_checkpoint + "training_description.txt", "w") as file:
        file.write(description)

    main(args, dir_checkpoint, read_dir, read_name)
