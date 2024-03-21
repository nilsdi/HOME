# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

root_dir = Path(__file__).parents[2]
path_train_data = root_dir / 'data/model/train'
dataset = datasets.ImageFolder(path_train_data,
                               transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

mean = 0.
std = 0.
nb_samples = 0.

for data, _ in tqdm(dataloader):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print('Mean:', mean)
print('Std:', std)
