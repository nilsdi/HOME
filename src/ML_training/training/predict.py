# %%
import torch
import matplotlib
import os
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

preroot_dir = Path(__file__).parents[4]
sys.path.append(str(preroot_dir))
sys.path.append(str(preroot_dir / "ISPRS_HD_NET"))
from ISPRS_HD_NET.utils.sync_batchnorm.batchnorm import convert_model  # type: ignore # noqa
from ISPRS_HD_NET.model.HDNet import HighResolutionDecoupledNet  # type: ignore # noqa
from ISPRS_HD_NET.utils.dataset import BuildingDataset  # type: ignore # noqa
from ISPRS_HD_NET.eval.eval_HDNet import eval_net  # type: ignore # noqa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use("tkagg")

# %%
root_dir = Path(__file__).parents[3]
# data_dir = str(root_dir) + "/data/model/topredict/"
data_dir = str(root_dir) + "/data/ML_prediction/"

dir_checkpoint = str(root_dir) + "/data/ML_model/save_weights/run_3/"
# dir_checkpoint = "../ISPRS_HD_NET/save_weights/pretrain/"
predict = True
# prediction_folder = 'predictions/BW_RGB_training/'
# image_folder = 'train/image'

prediction_folder = "predictions/"
image_folder = "topredict/image/"
txt_file = "pred_trondheim_2019_0.3_i_lzw_25.txt"

batchsize = 2
num_workers = 16
read_name = "HDNet_NOCI_best"
# Dataset = 'NOCI'
Dataset = "NOCI"
# assert Dataset in ['WHU', 'Inria', 'Mass', 'NOCI']
net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
print("Number of parameters: ", sum(p.numel() for p in net.parameters()))


# %%
def predict_and_eval(
    net,
    device,
    batch_size,
    data_dir,
    predict=False,
    prediction_folder=prediction_folder,
    image_folder=image_folder,
):
    dataset = BuildingDataset(
        dataset_dir=data_dir,
        training=False,
        txt_name=txt_file,
        data_name=Dataset,
        image_folder=image_folder,
        predict=predict,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    if predict:
        save_path = os.path.join(data_dir, prediction_folder)
        print("Saving predictions in ", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for batch in tqdm(loader):
            imgs = batch["image"]
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(imgs)
                pred1 = (pred[0] > 0).float()
                label_pred = pred1.squeeze().cpu().int().numpy().astype("uint8") * 255

                for i in range(len(pred1)):
                    img_name = batch["name"][i].split("/")[-1]
                    # print('Saving to', os.path.join(save_path, img_name))
                    wr = cv2.imwrite(os.path.join(save_path, img_name), label_pred[i])
                    if not wr:
                        print("Save failed!")
    else:
        best_score = eval_net(
            net, loader, device, savename=Dataset + "_" + read_name
        )  #
        print("Best iou:", best_score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    if read_name != "":
        net_state_dict = net.state_dict()
        state_dict = torch.load(
            dir_checkpoint + read_name + ".pth", map_location=device
        )
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict, strict=False)
        logging.info("Model loaded from " + read_name + ".pth")

    net = convert_model(net)
    net = torch.nn.parallel.DataParallel(net.to(device))
    torch.backends.cudnn.benchmark = True
    predict_and_eval(
        net=net,
        batch_size=batchsize,
        device=device,
        data_dir=data_dir,
        predict=predict,
        prediction_folder=prediction_folder,
        image_folder=image_folder,
    )
