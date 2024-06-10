# %%
import torch
import logging
import matplotlib
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
grandparent_dir = Path(__file__).parents[2]
sys.path.append(str(grandparent_dir))
from ISPRS_HD_NET.model.HDNet import HighResolutionDecoupledNet  # noqa
from ISPRS_HD_NET.utils.sync_batchnorm.batchnorm import convert_model  # noqa
from ISPRS_HD_NET.utils.dataset import BuildingDataset  # noqa
from ISPRS_HD_NET.eval.eval_HDNet import eval_net  # noqa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use('tkagg')

root_dir = Path(__file__).parents[1]
dir_checkpoint = str(grandparent_dir) + '/ISPRS_HD_NET/save_weights/pretrain/'
data_dir = str(root_dir / "data/model/")

batchsize = 16
num_workers = 8

datasets = ['WHU', 'Inria', 'Mass', 'NOCI']
weights = ['WHU', 'Inria', 'Mass', 'NOCI']


def eval_HRBR(net,
              device,
              batch_size):
    testdataset = BuildingDataset(
        dataset_dir=data_dir,
        training=False,
        txt_name="test.txt",
        data_name=Dataset)
    test_loader = DataLoader(testdataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=False)
    best_score = eval_net(net, test_loader, device,
                          savename=Dataset + '_' + read_name)  #
    print('Best iou:', best_score)
    return best_score


ious = pd.Series(index=weights)

if __name__ == '__main__':
    for weight in weights[-1:]:

        read_name = f'HDNet_{weight}_best'
        Dataset = weight
        assert Dataset in ['WHU', 'Inria', 'Mass', 'NOCI']
        net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
        print('Number of parameters: ', sum(p.numel()
                                            for p in net.parameters()))

        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s: %(message)s')
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        if read_name != '':
            net_state_dict = net.state_dict()
            state_dict = torch.load(
                dir_checkpoint + read_name + '.pth', map_location=device)
            net_state_dict.update(state_dict)
            net.load_state_dict(net_state_dict, strict=False)  # 删除了down1-3
            logging.info('Model loaded from ' + read_name + '.pth')

        logging.info('Dataset means std from ' + Dataset)
        net = convert_model(net)
        net = torch.nn.parallel.DataParallel(net.to(device))
        torch.backends.cudnn.benchmark = True
        ious.loc[weight] = eval_HRBR(net=net,
                                     batch_size=batchsize,
                                     device=device)
    print(ious)
