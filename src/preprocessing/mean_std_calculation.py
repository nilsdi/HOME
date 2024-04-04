# %%
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image


class FilteredDataset(Dataset):
    def __init__(self, img_dir, txt_path, transform=None):
        with open(txt_path, 'r') as file:
            self.img_names = [
                name + '.tif' for name in file.read().splitlines()]
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


# Usage
root_dir = Path(__file__).parents[2]
path_train_data = root_dir / 'data/model/train/image/'
path_train_txt = root_dir / 'data/model/dataset/train.txt'
dataset = FilteredDataset(path_train_data, path_train_txt,
                          transform=transforms.ToTensor())

# path_train_data = root_dir / 'data/model/train/image/'
# dataset = datasets.ImageFolder(path_train_data,
#                                transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

print('Mean:', mean)
print('Std:', std)
