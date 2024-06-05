import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

images = np.load("brain_tumor_dataset/norm_aug_gray/images.npy", allow_pickle=True)
masks = np.load("brain_tumor_dataset/norm_aug_gray/masks.npy", allow_pickle=True)
labels = np.load("brain_tumor_dataset/norm_aug_gray/labels.npy")
integer_to_class = {1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'}

print(f"images:{images.shape}, \
masks:{masks.shape}, \
labels:{labels.shape}")

# Data preparation

# Split data
# Determine the total number of samples
num_samples = len(images)

train_ratio = 0.8
val_ratio = 0.75

# Shuffle the indexes
indexes = np.arange(num_samples)

# Split data
train_indexes, test_indexes, train_labels, test_labels = train_test_split(indexes, labels, train_size = train_ratio, stratify = labels, random_state = 11)
test_indexes, val_indexes, test_labels, val_labels = train_test_split(test_indexes, test_labels, train_size = val_ratio, stratify = test_labels, random_state = 11)

print("Images"
      "\nTrain:", images[train_indexes].shape,
      "\nVal:", images[val_indexes].shape,
      "\nTest:", images[test_indexes].shape)

_, counts_train_u = np.unique(train_labels, return_counts=True)
_, counts_val_u = np.unique(val_labels, return_counts=True)
_, counts_test_u = np.unique(test_labels, return_counts=True)
labels_u = ["Meningioma (1)", "Glioma (2)", "Pituitary Tumor (3)"]

fig, axs = plt.subplots(1, 3, figsize = (15,5))

axs[0].bar(labels_u, counts_train_u, color=["skyblue", "palevioletred", "yellowgreen"], zorder = 3.5)
axs[0].grid(axis = "y", zorder = 2.5)
axs[0].set_title("Train set")

axs[1].bar(labels_u, counts_val_u, color=["skyblue", "palevioletred", "yellowgreen"], zorder = 3.5)
axs[1].grid(axis = "y", zorder = 2.5)
axs[1].set_title("Validation set")

axs[2].bar(labels_u, counts_test_u, color=["skyblue", "palevioletred", "yellowgreen"], zorder = 3.5)
axs[2].grid(axis = "y", zorder = 2.5)
axs[2].set_title("Test set")

plt.savefig("barplot_test_val_train_aug_norm.png")