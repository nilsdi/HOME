#%%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Step 1: Access the original colormap
original_cmap = plt.cm.get_cmap('Blues')

# Step 2: Create a new colormap by modifying the original
# Let's say we want to add a yellow color in the middle of the gradient
# We'll create a new colormap from the original with our modifications

# Define the colors at specific points and the positions of those colors in the colormap
colors = [(1, 1, 1), (0.03137254901960784, 0.18823529411764706, 0.4196078431372549),
            (1, 1, 0), (0.03137254901960784, 0.11372549019607843, 0.34509803921568627)]  
            # White to blue, yellow, and then to dark blue
positions = [0.0, 0.5, 0.75, 1.0]  # Positions where these colors appear

colors = [(1, 1, 1), (0.03137254901960784, 0.18823529411764706, 0.4196078431372549),
            (0.03137254901960784, 0.11372549019607843, 0.34509803921568627), (0,0,0)]  
            # White to blue, yellow, and then to dark blue
colors = [(1, 1, 1), 
            (0.10137254901960784, 0.58823529411764706, 0.7196078431372549),
               (0.05137254901960784, 0.38823529411764706, 0.5196078431372549),
             (0.03137254901960784, 0.28823529411764706, 0.4196078431372549), 
            (0.03137254901960784, 0.18823529411764706, 0.4196078431372549), (0,0,0)]
# Create a new colormap
new_cmap = mcolors.LinearSegmentedColormap.from_list("CustomBlues", colors)

# Step 3: Use the new colormap
# Generate some data
x = np.random.randn(10000)
y = np.random.randn(10000)
fig, ax = plt.subplots(figsize=(6, 6))
# Plot with the new colormap
plt.hexbin(x, y, gridsize=30, cmap= new_cmap)

norm = mcolors.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
cbar = plt.colorbar(sm, ax = ax)
#plt.colorbar()
plt.show()
# %%
# Print some individul colors as a new bar on a plot
fig, ax = plt.subplots(figsize=(9, 6))
colors = [(1, 1, 1), (0.03137254901960784, 0.18823529411764706, 0.4196078431372549)]
# translate colors to matplotlib format
positions = [0.0, 1.0]
#plt.plot([0, 1], [0, 1], color='black', linewidth=1)
for i in range(1):
    plt.plot([i, i], [0, 1], color=colors[i], linewidth=50)
plt.show()

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))
colors = [(1, 1, 1), 
            (0.10137254901960784, 0.58823529411764706, 0.7196078431372549),
               (0.05137254901960784, 0.38823529411764706, 0.5196078431372549),
             (0.03137254901960784, 0.24823529411764706, 0.4196078431372549), 
            (0.03137254901960784, 0.18823529411764706, 0.3696078431372549), 
            (0,0,0)]
# Create a custom colormap from the defined colors
CustomBlues = mcolors.LinearSegmentedColormap.from_list("CustomBlues", colors)

for i, color in enumerate(colors):
    plt.plot([i, i+1], [0.5, 0.5], color=color, linewidth=20)  # Adjust linewidth for bar width
# draw a box around the plot
plt.plot([0, 0, len(colors)+1, len(colors)+1, 0], [0.45, 0.55, 0.55, 0.45, 0.45], color='black', linewidth=1)

# Create a gradient for the colormap bar
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Draw the colormap bar
plt.imshow(gradient, aspect='auto', cmap=CustomBlues, extent=[0, len(colors), 0.1, 0.2])

plt.xlim(0, len(colors))  # Adjust x-axis limits to fit the bars
plt.ylim(0, 1)  # Adjust y-axis limits if needed
plt.axis('off')  # Optional: Remove axis for a cleaner look
plt.show()