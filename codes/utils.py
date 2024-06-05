import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
from keras.metrics import Metric, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from keras.backend import epsilon
from skimage.util.shape import view_as_windows
from tqdm import tqdm
from collections import Counter

def plot_training_history(history, title, filename, binary):
    num_classes = 2 if binary else 4
    loss = "Dice Loss" if binary else "Categorical Crossentropy Loss"

    plt.figure(figsize=(12, 8))

    # Plot Dice
    plt.subplot(2, 2, 1)
    plt.plot(history['dice_coefficient'], label='Training Dice Coefficient')
    plt.plot(history['val_dice_coefficient'], label='Validation Dice Coefficient')
    plt.ylim(0, 1.1)
    plt.grid()
    plt.title(title + ' - Training and Validation Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice coefficient')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 2)
    for i in range(num_classes):
        plt.plot(history[f'precision_{i}'], label=f'Training Precision on class {i}')
        plt.plot(history[f'val_precision_{i}'], label=f'Validation Precision on class {i}')
    plt.ylim(0, 1.1)
    plt.grid()
    plt.title(title + ' - Training and Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 2, 3)
    for i in range(num_classes):
        plt.plot(history[f'recall_{i}'], label=f'Training Recall on class {i}')
        plt.plot(history[f'val_recall_{i}'], label=f'Validation Recall on class {i}')
    plt.ylim(0, 1.1)
    plt.grid()
    plt.title(title + ' - Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Plot loss
    plt.subplot(2, 2, 4)
    plt.plot(history['loss'], label=f'Training Loss')
    plt.plot(history['val_loss'], label=f'Validation Loss')
    plt.ylim(0, 1.1)
    plt.grid()
    plt.title(title + f' - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)

def normalize_image(img):
    img = np.array(img, dtype=np.float32)
    normalized_img = cv2.normalize(img, None, 0, 1, norm_type=cv2.NORM_MINMAX)
    return normalized_img


class BalancedAccuracyScore(Metric):
    def __init__(self, num_classes=None, name='balanced_accuracy_score', **kwargs):
        super(BalancedAccuracyScore, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        if self.num_classes > 2:
            self.true_positives = [TruePositives(name='true_positives_' + str(i)) for i in range(num_classes)]
            self.true_negatives = [TrueNegatives(name='true_negatives_' + str(i)) for i in range(num_classes)]
            self.false_positives = [FalsePositives(name='false_positives_' + str(i)) for i in range(num_classes)]
            self.false_negatives = [FalseNegatives(name='false_negatives_' + str(i)) for i in range(num_classes)]
        else:
            self.true_positives = TruePositives(name='true_positives_binary')
            self.true_negatives = TrueNegatives(name='true_negatives_binary')
            self.false_positives = FalsePositives(name='false_positives_binary')
            self.false_negatives = FalseNegatives(name='false_negatives_binary')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.float32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)
        if self.num_classes > 2:  # Multiclass
            for i in range(self.num_classes):
                self.true_positives[i].update_state(y_true == i, y_pred == i, sample_weight)
                self.true_negatives[i].update_state(y_true == i, y_pred == i, sample_weight)
                self.false_positives[i].update_state(y_true == i, y_pred == i, sample_weight)
                self.false_negatives[i].update_state(y_true == i, y_pred == i, sample_weight)
        else:  # Binary
            self.true_positives.update_state(y_true, y_pred, sample_weight)
            self.true_negatives.update_state(y_true, y_pred, sample_weight)
            self.false_positives.update_state(y_true, y_pred, sample_weight)
            self.false_negatives.update_state(y_true, y_pred, sample_weight)
            # tf.print(f"TP {self.true_positives.result()}")
            # tf.print(self.true_positives.result())

            # tf.print(f"TN {self.true_negatives.result()}")
            # tf.print(self.true_negatives.result())

            # tf.print(f"FP {self.false_positives.result()}")
            # tf.print(self.false_positives.result())
            
            # tf.print(f"FN {self.false_negatives.result()}")
            # tf.print(self.false_negatives.result())
    def result(self):
        if self.num_classes > 2:  # Multiclass
            recall_per_class = []
            for i in range(self.num_classes):
                recall_per_class.append((self.true_positives[i].result() / (self.true_positives[i].result() + self.false_negatives[i].result() + epsilon()) +
                                         self.true_negatives[i].result() / (self.true_negatives[i].result() + self.false_positives[i].result() + epsilon())) / 2)
            balanced_accuracy = tf.reduce_mean(recall_per_class)
        else:  # Binary
            recall = self.true_positives.result() / (self.true_positives.result() + self.false_negatives.result() + epsilon())
            specificity = self.true_negatives.result() / (self.true_negatives.result() + self.false_positives.result() + epsilon())
            balanced_accuracy = (recall + specificity) / 2
        return balanced_accuracy

    def reset_state(self):
        if self.num_classes > 2:  # Multiclass
            for i in range(self.num_classes):
                self.true_positives[i].reset_states()
                self.true_negatives[i].reset_states()
                self.false_positives[i].reset_states()
                self.false_negatives[i].reset_states()
        else:  # Binary
            self.true_positives.reset_states()
            self.true_negatives.reset_states()
            self.false_positives.reset_states()
            self.false_negatives.reset_states()
        
class DiceCoefficient(Metric):
    def __init__(self, num_classes, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        if self.num_classes > 2:
            self.true_positives = [TruePositives(name='true_positives_' + str(i)) for i in range(num_classes)]
            self.true_negatives = [TrueNegatives(name='true_negatives_' + str(i)) for i in range(num_classes)]
            self.false_positives = [FalsePositives(name='false_positives_' + str(i)) for i in range(num_classes)]
            self.false_negatives = [FalseNegatives(name='false_negatives_' + str(i)) for i in range(num_classes)]
        else:
            self.true_positives = TruePositives(name='true_positives_binary')
            self.true_negatives = TrueNegatives(name='true_negatives_binary')
            self.false_positives = FalsePositives(name='false_positives_binary')
            self.false_negatives = FalseNegatives(name='false_negatives_binary')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.float32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)

        if self.num_classes > 2:  # Multiclass
            for i in range(self.num_classes):
                self.true_positives[i].update_state(y_true == i, y_pred == i, sample_weight)
                self.true_negatives[i].update_state(y_true == i, y_pred == i, sample_weight)
                self.false_positives[i].update_state(y_true == i, y_pred == i, sample_weight)
                self.false_negatives[i].update_state(y_true == i, y_pred == i, sample_weight)
        else:  # Binary
            self.true_positives.update_state(y_true, y_pred, sample_weight)
            self.true_negatives.update_state(y_true, y_pred, sample_weight)
            self.false_positives.update_state(y_true, y_pred, sample_weight)
            self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        if self.num_classes > 2:  # Multiclass
            dice_per_class = []
            for i in range(self.num_classes):
                dice_per_class.append((2 * self.true_positives[i].result()) / (2 * self.true_positives[i].result() + self.false_positives[i].result() + self.false_negatives[i].result() + epsilon()))
            dice_coefficient = tf.reduce_mean(dice_per_class)
        else:  # Binary
            dice_coefficient = (2 * self.true_positives.result()) / (2 * self.true_positives.result() + self.false_positives.result() + self.false_negatives.result() + epsilon())
        return dice_coefficient

    def reset_state(self):
        if self.num_classes > 2:  # Multiclass
            for i in range(self.num_classes):
                self.true_positives[i].reset_states()
                self.true_negatives[i].reset_states()
                self.false_positives[i].reset_states()
                self.false_negatives[i].reset_states()
        else:  # Binary
            self.true_positives.reset_states()
            self.true_negatives.reset_states()
            self.false_positives.reset_states()
            self.false_negatives.reset_states()

def create_windows_with_labels(image, mask, window_size, num_classes):
    windows = []
    window_labels = []
    padded_image = handle_border_pixels(image, window_size)
    windows.extend(create_windows(padded_image, window_size))
    window_labels.extend(mask.reshape(-1, num_classes))
    return np.array(windows), np.array(window_labels)

def handle_border_pixels(image, window_size):
    border_size = window_size // 2
    return np.pad(image, border_size, mode='reflect')
    
def create_windows(image, window_size):
    windows = view_as_windows(image, (window_size, window_size))
    return windows.reshape(-1, window_size, window_size)

def compute_class_ratio(central_pixels):
    # Compute the class distribution
    class_counts = Counter(central_pixels)
    
    # Compute the ratio of samples in each class
    class_ratio = {cls: count / len(central_pixels) for cls, count in class_counts.items()}
    
    return class_ratio

def undersample_majority_class(image_windows, central_pixels):
    central_pixels_tuple = tuple(map(tuple, central_pixels))
    
    # Compute the class ratio
    class_ratio = compute_class_ratio(central_pixels_tuple)
    
    # Find the minority and majority class
    minority_class = min(class_ratio, key=class_ratio.get)
    majority_class = max(class_ratio, key=class_ratio.get)
    
    # Find the number of samples in the minority class
    minority_samples = sum(1 for cls in central_pixels_tuple if cls == minority_class)
    
    # Determine the number of samples to keep for the majority class
    majority_samples = sum(1 for cls in central_pixels_tuple if cls == majority_class)
    desired_majority_samples = minority_samples
    
    # Find indices of minority class samples
    minority_indices = [idx for idx, cls in enumerate(central_pixels_tuple) if cls == minority_class]
    
    # Find indices of majority class samples
    majority_indices = [idx for idx, cls in enumerate(central_pixels_tuple) if cls == majority_class]
    
    # Randomly select subset of majority class samples
    np.random.shuffle(majority_indices)
    selected_majority_indices = majority_indices[:desired_majority_samples]
    
    # Keep all samples from the minority class
    undersampled_indices = minority_indices + selected_majority_indices
    
    # Keep the selected samples
    undersampled_image_windows = image_windows[undersampled_indices]
    undersampled_central_pixels = central_pixels[undersampled_indices]
    
    return undersampled_image_windows, undersampled_central_pixels


def train_on_batch_windows(images, masks, model, window_size, num_classes, epochs=1, callbacks=None, validation_data=None, batch_size=128):
    history = {}
    if callbacks:
        for callback in callbacks:
            callback.set_model(model)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Extract compiled metrics
        compiled_metrics_names = model.metrics_names
        # Define metrics dictionaries
        epoch_metrics = {name: [] for name in compiled_metrics_names}
        validation_metrics = {f'val_{name}': [] for name in compiled_metrics_names}
        
        epoch_losses = []
        validation_losses = []
        # Shuffle data before each epoch
        perm = np.random.permutation(len(images))
        shuffled_images = images[perm]
        shuffled_masks = masks[perm]

        with tqdm(total=len(shuffled_images), desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
            for image, mask in zip(shuffled_images, shuffled_masks):
                # Create windows and undersample
                image_windows, central_pixels = create_windows_with_labels(image, mask, window_size, num_classes)
                image_windows, central_pixels = undersample_majority_class(image_windows, central_pixels)

                # Iterate over windows in batches
                for i in range(0, len(image_windows), batch_size):
                    batch_windows = image_windows[i:i+batch_size]
                    batch_central_pixels = central_pixels[i:i+batch_size]

                    metrics = model.train_on_batch(batch_windows, batch_central_pixels, return_dict=True)
                    if compiled_metrics_names == []:
                        # Extract compiled metrics
                        compiled_metrics_names = model.metrics_names
                        # Define metrics dictionaries
                        epoch_metrics = {name: [] for name in compiled_metrics_names}
                        validation_metrics = {f'val_{name}': [] for name in compiled_metrics_names}
                    epoch_losses.append(metrics['loss'])
                    for name, value in metrics.items():
                        epoch_metrics[name].append(value)

                pbar.update(1)

        # Compute validation metrics
        if validation_data is not None:
            print("Validation")
            val_images, val_masks = validation_data
            with tqdm(total=len(val_images), desc=f'Epoch {epoch + 1}/{epochs}') as pbar:    
                for val_image, val_mask in zip(val_images, val_masks):
                    val_image_windows, val_central_pixels = create_windows_with_labels(val_image, val_mask, window_size, num_classes)
                    # Iterate over windows in batches
                    for i in range(0, len(image_windows), batch_size):
                        val_batch_windows = val_image_windows[i:i+batch_size]
                        val_batch_central_pixels = val_central_pixels[i:i+batch_size]
                        val_metrics = model.test_on_batch(val_batch_windows, val_batch_central_pixels, return_dict=True)

                        validation_losses.append(val_metrics['loss'])
                        for name, value in val_metrics.items():
                            validation_metrics[f'val_{name}'].append(value)

                    pbar.update(1)

        epoch_loss = np.mean(epoch_losses)
        epoch_metrics = {name: np.mean(values) for name, values in epoch_metrics.items()}
        validation_loss = np.mean(validation_losses) if validation_losses else None
        validation_metrics = {name: np.mean(values) for name, values in validation_metrics.items()} if validation_metrics else None

        print(f"Loss: {epoch_loss}, Metrics: {epoch_metrics}, Validation Loss: {validation_loss}, Validation Metrics: {validation_metrics}")

        # Save epoch history
        for key, value in epoch_metrics.items():
            if key in history:
                history[key].append(value)
            else:
                history[key] = [value]

        for key, value in validation_metrics.items():
            if key in history:
                history[key].append(value)
            else:
                history[key] = [value]

        # Execute callbacks
        if callbacks:
            for callback in callbacks:
                callback.on_epoch_end(epoch)

    return history


def evaluate_on_batch_windows(images, masks, model, window_size, num_classes, batch_size=128):
    # Extract compiled metrics
    compiled_metrics_names = model.metrics_names
    # Define metrics dictionaries
    metrics = {name: [] for name in compiled_metrics_names}

    # Shuffle data before each epoch
    perm = np.random.permutation(len(images))
    shuffled_images = images[perm]
    shuffled_masks = masks[perm]

    with tqdm(total=len(shuffled_images), desc=f'Evaluation') as pbar:
        for i in range(len(shuffled_images)):
            image = shuffled_images[i]
            mask = shuffled_masks[i]
            image_windows, central_pixels = create_windows_with_labels(image, mask, window_size, num_classes)

            num_batches = (len(image_windows) + batch_size - 1) // batch_size

            for j in range(num_batches):
                start = j * batch_size
                end = min((j + 1) * batch_size, len(image_windows))
                batch_image_windows = image_windows[start:end]
                batch_central_pixels = central_pixels[start:end]

                test_metrics = model.test_on_batch(batch_image_windows, batch_central_pixels, return_dict=True)

                if compiled_metrics_names == []:
                    # Extract compiled metrics
                    compiled_metrics_names = model.metrics_names
                    # Define metrics dictionaries
                    metrics = {name: [] for name in compiled_metrics_names}
                for name, value in test_metrics.items():
                    metrics[name].append(value)

            pbar.update(1)

    metrics = {name: np.mean(values) for name, values in metrics.items()}

    print(f"Metrics: {metrics}")

    return metrics

def predict_on_batch_windows(images, masks, model, window_size, num_classes, batch_size=128):
    predictions = []
    shape = (images[0].shape[0], images[0].shape[1], num_classes)

    # Shuffle data before each epoch
    perm = np.random.permutation(len(images))
    shuffled_images = images[perm]
    shuffled_masks = masks[perm]

    with tqdm(total=len(shuffled_images), desc=f'Prediction') as pbar:
        for i in range(len(shuffled_images)):
            image = shuffled_images[i]
            mask = shuffled_masks[i]
            image_windows, central_pixels = create_windows_with_labels(image, mask, window_size, num_classes)

            batch_predictions = []
            num_batches = (len(image_windows) + batch_size - 1) // batch_size

            for j in range(num_batches):
                start = j * batch_size
                end = min((j + 1) * batch_size, len(image_windows))
                batch_image_windows = image_windows[start:end]

                preds = model.predict_on_batch(batch_image_windows)
                batch_predictions.append(preds)

            batch_predictions = np.concatenate(batch_predictions, axis=0)
            predictions.append(np.reshape(batch_predictions, shape))
            pbar.update(1)

    return predictions