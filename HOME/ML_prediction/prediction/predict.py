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
import argparse
import numpy as np
from HOME.get_data_path import get_data_path
import json
from datetime import datetime


# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)
grandparent_dir = Path(__file__).parents[4]
sys.path.append(str(grandparent_dir))
sys.path.append(str(grandparent_dir / "ISPRS_HD_NET"))
from ISPRS_HD_NET.utils.sync_batchnorm.batchnorm import convert_model  # type: ignore # noqa
from ISPRS_HD_NET.model.HDNet import HighResolutionDecoupledNet  # type: ignore # noqa
from ISPRS_HD_NET.utils.dataset import BuildingDataset  # type: ignore # noqa
from ISPRS_HD_NET.eval.eval_HDNet import fast_hist  # type: ignore # noqa
from ISPRS_HD_NET.eval.eval_HDNet import eval_net  # type: ignore # noqa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use("tkagg")

# %%
data_dir = data_path / "ML_prediction/"


# %%
def predict_and_eval(
    project_name,
    tile_id,
    BW=False,
    image_folder="topredict/image/",
    label_folder="topredict/label/",
    predict=True,
    evaluate=False,
    batchsize=16,
    num_workers=8,
    data_dir=data_dir,
    prediction_folder=data_path / "ML_prediction/predictions",
    Dataset="NOCI",
):

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    if BW:
        model_weights = "run_5"
        dir_checkpoint = data_path / f"ML_model/save_weights/{model_weights}/"
        Dataset = "NOCI_BW"
        read_name = [
            f for f in os.listdir(dir_checkpoint) if ("best" in f) & ("NOCI" in f)
        ][0][:-4]
    else:
        model_weights = "run_6"
        dir_checkpoint = data_path / f"ML_model/save_weights/{model_weights}/"
        Dataset = "NOCI"
        read_name = [
            f for f in os.listdir(dir_checkpoint) if ("best" in f) & ("NOCI" in f)
        ][0][:-4]

    pred_name = f"pred_{project_name}_{tile_id}.txt"

    # Add project metadata to the log
    with open(data_path / "metadata_log/predictions_log.json", "r") as file:
        predictions_log = json.load(file)
    highest_tile_key = int(max([int(key) for key in predictions_log.keys()]))
    prediction_key = highest_tile_key + 1

    predictions_log[prediction_key] = {
        "project_name": project_name,
        "tile_id": tile_id,
        "BW": BW,
        "evaluate": evaluate,
        "batchsize": batchsize,
        "num_workers": num_workers,
        "model weights": model_weights,
        "read_name": read_name,
        "pred_name": pred_name,
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_folder": str(prediction_folder / project_name / f"tiles_{tile_id}"),
    }

    net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)

    if read_name != "":
        net_state_dict = net.state_dict()
        state_dict = torch.load(
            dir_checkpoint / f"{read_name}.pth", map_location=device
        )
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict, strict=False)
        logging.info("Model loaded from " + str(read_name) + ".pth")

    net = convert_model(net)
    net = torch.nn.parallel.DataParallel(net.to(device))
    torch.backends.cudnn.benchmark = True

    print("Number of parameters: ", sum(p.numel() for p in net.parameters()))

    dataset = BuildingDataset(
        dataset_dir=data_dir,
        training=False,
        txt_name=pred_name,
        data_name=Dataset,
        image_folder=image_folder,
        label_folder=label_folder,
        predict=predict & (not evaluate),
    )

    loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    hist = 0
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
                img_name = "/".join(batch["name"][i].split("/")[-3:])
                img_path = os.path.join(save_path, img_name)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                wr = cv2.imwrite(img_path, label_pred[i])
                if not wr:
                    print("Save failed!")

            if evaluate:
                true_labels = batch["label"]
                hist += fast_hist(
                    pred1.flatten().cpu().detach().int().numpy(),
                    true_labels.flatten().cpu().int().numpy(),
                    2,
                )
                # rec += recall_score(
                #     true_labels.flatten().cpu().int().numpy(),
                #     pred1.flatten().cpu().detach().int().numpy(),
                # )

    if evaluate:
        if predict:
            acc_R = np.diag(hist) / hist.sum(1) * 100
            print("Recall: ", acc_R)
            # print("Recall from sklearn: ", rec / len(loader))
        else:
            acc_R = eval_net(net, loader, device)
    else:
        acc_R = None

    # Recall for class 1
    predictions_log[prediction_key]["recall"] = acc_R[1]
    with open(data_path / "metadata_log/predictions_log.json", "w") as file:
        json.dump(predictions_log, file, indent=4)

    return prediction_key


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Tile raw orthophotos for prediction with ML model"
    )
    parser.add_argument("-p", "--project_name", required=True, type=str)
    parser.add_argument("--tile_id", required=True, type=str)
    parser.add_argument("-r", "--res", required=False, type=float, default=0.3)
    parser.add_argument(
        "-c", "--compression", required=False, type=str, default="i_lzw_25"
    )
    parser.add_argument(
        "-bw",
        "--BW",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-if", "--image_folder", required=False, type=str, default="topredict/image"
    )
    parser.add_argument(
        "-lf", "--label_folder", required=False, type=str, default="topredict/label"
    )
    parser.add_argument(
        "--predict",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--evaluate",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()
    print(args)
    predict_and_eval(
        project_name=args.project_name,
        tile_id=args.tile_id,
        BW=args.BW,
        image_folder=args.image_folder,
        label_folder=args.label_folder,
        predict=args.predict,
        evaluate=args.evaluate,
    )
