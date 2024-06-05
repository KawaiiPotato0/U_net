import matplotlib.colors as mcolors
import numpy as np
from model import segnet, unet, unet_multiclass, create_sliding_window_model
from keras.optimizers import Adam
from utils import DiceCoefficient, predict_on_batch_windows
from matplotlib import cm, pyplot as plt
import seaborn as sns

def predict(data_path, weight_path, model_name, binary = True, window_size=31):
    checkpoint_path = f"{weight_path}/cp.weights.h5"

    images = np.load(f"{data_path}/images.npy", allow_pickle=True)
    masks = np.load(f"{data_path}/masks.npy", allow_pickle=True)
    labels = np.load(f"{data_path}/labels.npy", allow_pickle=True)
    num_classes = 2

    if not binary:
        num_classes = 4 # 0: 'no tumor', 1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'
        masks = np.array([np.array(np.array(img, dtype=np.uint8)*labels[i], dtype=np.float32) for i, img in enumerate(masks)])

    indexes = [np.random.choice(np.where(labels == value)[0]) for value in [1, 2, 3]]

    if model_name == "unet":
        if binary:
            model = unet((128, 128, 1))
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[DiceCoefficient(num_classes=2)])
        else:
            model = unet_multiclass((128, 128, 1), num_classes)
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[DiceCoefficient(num_classes=4)])
    elif model_name == "segnet":
        model = segnet((128, 128, 1), num_classes)
        if binary:
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[DiceCoefficient(num_classes=2)])
        else:
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[DiceCoefficient(num_classes=4)])
    elif model_name == "window":
        model = create_sliding_window_model(window_size, num_classes)
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                        loss='categorical_crossentropy', 
                        metrics=[DiceCoefficient(num_classes=num_classes)])
            
    # Loads the weights
    model.load_weights(checkpoint_path)

    # Evaluate the model
    if model_name == "unet" or model_name == "segnet":
        pred_prob = model.predict(images[indexes])  # Predicted probabilities for each class
    elif model_name == "window":
        pred_prob = predict_on_batch_windows(images[indexes], masks[indexes], model, window_size, num_classes)
    # Get the most probable label for each pixel
    pred_labels = np.argmax(pred_prob, axis=-1)  # Take the index of the maximum probability along the channel axis

    # Plot
    # Set up the figure and axes
    fig, axes = plt.subplots(len(indexes), 3, figsize=(15, 5*len(indexes)), gridspec_kw={'width_ratios': [1, 1, 1]})
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    norm = mcolors.Normalize(vmin=0, vmax=num_classes-1)
    cmap = cm.get_cmap('viridis') 

    title_fontsize = 40
    label_fontsize = 20
    tick_label_fontsize = 15
    # Plot each row
    for i, idx in enumerate(indexes):
        # Reference image
        sns.heatmap(images[idx], cmap='gray', ax=axes[i, 0], cbar=False)
        axes[i, 0].set_title("Reference image", fontsize=label_fontsize)
        axes[i, 0].axis('off')  # Remove ticks for reference image

        # Predicted mask
        sns.heatmap(pred_labels[i], cmap=cmap, norm=norm, ax=axes[i, 1], cbar=False)
        axes[i, 1].set_title("Predicted mask", fontsize=label_fontsize)
        axes[i, 1].axis('off')  # Remove ticks for predicted mask

        # Ground truth mask
        sns.heatmap(masks[idx], cmap=cmap, norm=norm, ax=axes[i, 2], cbar=False)
        axes[i, 2].set_title("Ground truth", fontsize=label_fontsize)
        axes[i, 2].axis('off')  # Remove ticks for ground truth

    # Add color bar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cbar.ax.tick_params(labelsize=tick_label_fontsize)
    if binary:
        custom_tick_labels = ['no tumor', 'tumor']  # Replace with your custom labels
    else:
        custom_tick_labels = ['no tumor', 'meningioma', 'glioma', 'pituitary tumor']  # Replace with your custom labels
    cbar.set_ticks(np.arange(num_classes))  # Place ticks in the middle of each class
    cbar.set_ticklabels(custom_tick_labels)
    
    # Save and show the plot
    plt.suptitle("Prediction visualization", fontsize=title_fontsize)
    plt.savefig(f"{weight_path}/prediction.png", bbox_inches='tight')