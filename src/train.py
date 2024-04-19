# %%
import torch
import matplotlib
import os
import sys
from pathlib import Path
grandparent_dir = Path(__file__).parents[2]
sys.path.append(str(grandparent_dir))
sys.path.insert(0, str(grandparent_dir / 'ISPRS_HD_NET'))
from ISPRS_HD_NET.Train_HDNet import main  # noqa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
matplotlib.use('tkagg')

torch.set_num_threads(16)

root_dir = Path(__file__).parents[1]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch HDNet training")
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument(
        "--data-path",
        default="data/model/original/")
    parser.add_argument("--numworkers", default=8, type=int)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--base-channel", default=48, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--read-name", default='')
    parser.add_argument("--save-name", default='HDNet_NOCI')
    parser.add_argument("--DataSet", default='NOCI')
    parser.add_argument("--image-folder", default='train/image')
    args = parser.parse_args()

    return args


# %%
if __name__ == '__main__':
    args = parse_args()
    dir_checkpoint = str(root_dir) + '/data/model/save_weights/run_4/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    main(args, dir_checkpoint)
