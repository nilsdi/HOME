# %%
import torch
import matplotlib
import os
import logging
import sys
from pathlib import Path
grandparent_dir = Path(__file__).parents[2]
sys.path.append(str(grandparent_dir))
sys.path.append(str(grandparent_dir / 'ISPRS_HD_NET'))
from ISPRS_HD_NET.predict import predict  # noqa
from ISPRS_HD_NET.utils.sync_batchnorm.batchnorm import convert_model  # noqa
from ISPRS_HD_NET.model.HDNet import HighResolutionDecoupledNet  # noqa

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

matplotlib.use('tkagg')
batchsize = 16
num_workers = 0
read_name = 'HDNet_Inria_best'
Dataset = 'NOCI'
assert Dataset in ['WHU', 'Inria', 'Mass', 'NOCI']
net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
print('Number of parameters: ', sum(p.numel() for p in net.parameters()))


data_dir = "data/model/"
dir_checkpoint = str(grandparent_dir) + '/ISPRS_HD_NET/save_weights/pretrain/'
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if read_name != '':
        net_state_dict = net.state_dict()
        state_dict = torch.load(
            dir_checkpoint + read_name + '.pth', map_location=device)
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict, strict=False)  # 删除了down1-3
        logging.info('Model loaded from ' + read_name + '.pth')

    net = convert_model(net)
    net = torch.nn.parallel.DataParallel(net.to(device))
    torch.backends.cudnn.benchmark = True
    predict(net=net,
            batch_size=batchsize,
            device=device,
            data_dir=data_dir)
