# %% Run evaluation with the weights from each epoch, and plot

import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.ticker import FuncFormatter

# Initialize lists to store epochs, batch losses, and IoUs
epochs = []
batch_losses = []
IoUs = []
lrs = []

root_dir = Path(__file__).parents[2]

# Open the file and read the data
with open(root_dir / 'results20240418-174149.txt', 'r') as f:
    for line in f:
        if line.startswith('[epoch:'):
            epochs.append(int(line.split()[1][:-1]))
        elif line.startswith('epoch_loss:'):
            batch_losses.append(float(line.split()[1]))
        elif line.startswith('val_IoU:'):
            IoUs.append(float(line.split()[1]))
        elif line.startswith('lr'):
            lrs.append(float(line.split()[1]))


# %% Create a figure and a single subplot
fig, ax1 = plt.subplots()

# Plot batch loss over epochs on the first y-axis
ax1.plot(epochs, batch_losses, marker='o', color='b')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cost', color='b')
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

# Define a function to format the ticks


def format_ticks(x, pos):
    return f'{x * 1e4:.0f}e-04'


# Create a FuncFormatter object
formatter = FuncFormatter(format_ticks)

# Create a figure and a single subplot
host = host_subplot(111, axes_class=AA.Axes)

# Plot batch loss over epochs on the first y-axis
plt.subplots_adjust(right=0.75)
par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))
par1.axis["left"] = par1.new_fixed_axis(loc="right", offset=(0, 0))
par1.axis["left"].toggle(all=True)

# Plot the data on each axes
host.plot(epochs, batch_losses, marker='o', color='b')
par1.plot(epochs, IoUs, marker='o', color='r')
par2.plot(epochs, lrs, marker='o', color='g')

# Set labels and colors for each axis
host.set_xlabel('Epochs')
host.set_ylabel('Cost', color='b')
par1.set_ylabel('IoU', color='r')
par2.yaxis.set_major_formatter(formatter)
par2.set_ylabel('Learning Rate', color='g')

host.tick_params('y', colors='b')
par1.tick_params('y', colors='r')
par2.tick_params('y', colors='g')

host.axis["left"].major_ticklabels.set_color('b')
par1.axis["left"].major_ticklabels.set_color('r')
par2.axis["right"].major_ticklabels.set_color('g')

plt.title('Batch Loss, IoU, and Learning Rate over Epochs')
plt.tight_layout()
plt.show()
# %%
