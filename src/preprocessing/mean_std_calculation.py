# %%
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os


class FilteredDataset(Dataset):
    def __init__(self, img_dir, txt_path, transform=None):
        if txt_path is not None:
            with open(txt_path, 'r') as file:
                self.img_names = [
                    name + '.tif' for name in file.read().splitlines()]
        else:
            self.img_names = [name for name in os.listdir(img_dir)]
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_dir / self.img_names[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_names)


def calculate_mean_std(dataset_path, txt_path=None, batch_size=32):
    # Define the dataset
    dataset = FilteredDataset(dataset_path, txt_path,
                              transform=transforms.ToTensor())

    # Define the data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


# Usage
if __name__ == '__main__':
    root_dir = Path(__file__).parents[2]
    path_train_data = root_dir / 'data/model/original/train_poor/image/'
    path_train_txt = root_dir / 'data/model/original/dataset/train.txt'

    mean, std = calculate_mean_std(path_train_data, path_train_txt)

    print('Mean:', mean)
    print('Std:', std)
