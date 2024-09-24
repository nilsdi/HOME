import argparse
from glob import glob
import os.path as osp
from tqdm import tqdm
from HOME.ML_training.preprocessing.step_05_distance_map import process

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", dest="datadir", default="data/ML_training/")
parser.add_argument("--outname", default="boundary_tune")
# parser.add_argument('--split', nargs='+', default=['train', 'test'])
parser.add_argument("--metric", default="euc", choices=["euc", "taxicab"])
args = parser.parse_args()

label_list = [0, 255]


indir = osp.join(args.datadir, f"tune", "label")
outdir = osp.join(args.datadir, args.outname)
args_to_apply = [
    (indir, outdir, osp.basename(basename))
    for basename in glob(osp.join(indir, "*.tif"))
]
print("Processing {} files".format(len(args_to_apply)))
for i in tqdm(range(0, len(args_to_apply))):
    process(args_to_apply[i])
