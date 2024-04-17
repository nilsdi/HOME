# %% Run evaluation with the weights from each epoch, and plot

import matplotlib.pyplot as plt
from pathlib import Path

# Initialize lists to store epochs, batch losses, and IoUs
epochs = []
batch_losses = []
IoUs = []

root_dir = Path(__file__).parents[2]

# Open the file and read the data
with open(root_dir / 'results20240415-173102.txt', 'r') as f:
    for line in f:
        if line.startswith('[epoch:'):
            epochs.append(int(line.split()[1][:-1]))
        elif line.startswith('batch_loss:'):
            batch_losses.append(float(line.split()[1]))
        elif line.startswith('val_IoU:'):
            IoUs.append(float(line.split()[1]))

# Create a figure and a single subplot
fig, ax1 = plt.subplots()

# Plot batch loss over epochs on the first y-axis
ax1.plot(epochs, batch_losses, marker='o', color='b')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Batch Loss', color='b')
ax1.tick_params('y', colors='b')

# Create a second y-axis and plot IoU over epochs on it
ax2 = ax1.twinx()
ax2.plot(epochs, IoUs, marker='o', color='r')
ax2.set_ylabel('IoU', color='r')
ax2.tick_params('y', colors='r')

plt.title('Batch Loss and IoU over Epochs')
plt.tight_layout()
plt.show()

# %%
