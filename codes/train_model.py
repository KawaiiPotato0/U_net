from keras.optimizers import Adam
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from model import unet_multiclass, unet, create_sliding_window_model, segnet
from utils import DiceCoefficient, plot_training_history, BalancedAccuracyScore, train_on_batch_windows
from keras.metrics import Recall, Precision
from pathlib import Path
import numpy as np
import tensorflow as tf
import math

def train(data_path, result_path, title, model_name="unet", binary=True, random_state=11, window_size=31):
    images = np.load(f"{data_path}/images.npy", allow_pickle=True)
    masks = np.load(f"{data_path}/masks.npy", allow_pickle=True)
    labels = np.load(f"{data_path}/labels.npy", allow_pickle=True)
    num_classes = 4 # 0: 'no tumor', 1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'
    if not binary:
        # Encode classes: 1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'
        masks = np.array([np.array(np.array(img, dtype=np.uint8)*labels[i], dtype=np.float32) for i, img in enumerate(masks)])
    else:
        num_classes = 2
    # Convert the ground truth masks to one-hot encoded format
    masks = tf.keras.utils.to_categorical(masks, num_classes=num_classes)

    # Split data
    # Determine the total number of samples
    num_samples = len(images)

    train_ratio = 0.8 # fraction of whole dataset
    val_ratio = 0.75 # fraction of test dataset

    # Shuffle the indexes
    indexes = np.arange(num_samples)

    # Split data
    train_indexes, test_indexes, train_labels, test_labels = train_test_split(indexes, labels, train_size = train_ratio, stratify = labels, random_state = random_state)
    test_indexes, val_indexes, test_labels, val_labels = train_test_split(test_indexes, test_labels, train_size = val_ratio, stratify = test_labels, random_state = random_state)


    # TRAINING

    batch_size = 8

    Path(result_path).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{result_path}/cp.weights.h5"

    # Calculate the number of batches per epoch
    n_batches = len(train_indexes) / batch_size
    n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer
    
    if model_name == "unet":
        if binary:
            model = unet((128, 128, 1))
            model.summary()
            model.compile(optimizer=Adam(learning_rate=1e-4), 
                          loss='categorical_crossentropy', 
                          metrics=[DiceCoefficient(num_classes=2), BalancedAccuracyScore(num_classes=2),
                                   Recall(class_id=0, name='recall_0'), Precision(class_id=0, name='precision_0'),
                                   Recall(class_id=1, name='recall_1'), Precision(class_id=1, name='precision_1')])
        else:
            model = unet_multiclass((128, 128, 1), num_classes)
            model.summary()
            model.compile(optimizer=Adam(learning_rate=1e-4), 
                          loss='categorical_crossentropy', 
                          metrics=[DiceCoefficient(num_classes=4), BalancedAccuracyScore(num_classes=4),
                                   Recall(class_id=0, name='recall_0'), Precision(class_id=0, name='precision_0'), 
                                   Recall(class_id=1, name='recall_1'), Precision(class_id=1, name='precision_1'), 
                                   Recall(class_id=2, name='recall_2'), Precision(class_id=2, name='precision_2'), 
                                   Recall(class_id=3, name='recall_3'), Precision(class_id=3, name='precision_3')])

    elif model_name == "segnet":
        model = segnet((128, 128, 1), num_classes)
        model.summary()
        if binary:
            model.compile(optimizer=Adam(learning_rate=1e-4), 
                          loss='categorical_crossentropy', 
                          metrics=[DiceCoefficient(num_classes=2), BalancedAccuracyScore(num_classes=2),
                                   Recall(class_id=0, name='recall_0'), Precision(class_id=0, name='precision_0'),
                                   Recall(class_id=1, name='recall_1'), Precision(class_id=1, name='precision_1')])
        else:
            model.compile(optimizer=Adam(learning_rate=1e-4), 
                          loss='categorical_crossentropy', 
                          metrics=[DiceCoefficient(num_classes=4), BalancedAccuracyScore(num_classes=4),
                                   Recall(class_id=0, name='recall_0'), Precision(class_id=0, name='precision_0'), 
                                   Recall(class_id=1, name='recall_1'), Precision(class_id=1, name='precision_1'), 
                                   Recall(class_id=2, name='recall_2'), Precision(class_id=2, name='precision_2'), 
                                   Recall(class_id=3, name='recall_3'), Precision(class_id=3, name='precision_3')])
               
    elif model_name == "window":
        if binary:
            model = create_sliding_window_model(window_size, num_classes)
            model.summary()
            model.compile(optimizer=Adam(learning_rate=1e-4), 
                          loss='categorical_crossentropy', 
                          metrics=[DiceCoefficient(num_classes=2), BalancedAccuracyScore(num_classes=2),
                                   Recall(class_id=0, name='recall_0'), Precision(class_id=0, name='precision_0'), 
                                   Recall(class_id=1, name='recall_1'), Precision(class_id=1, name='precision_1')])
        else:
            model = create_sliding_window_model(window_size, num_classes)
            model.summary()
            model.compile(optimizer=Adam(learning_rate=1e-4),
                          loss='categorical_crossentropy', 
                          metrics=[DiceCoefficient(num_classes=4), BalancedAccuracyScore(num_classes=4),
                                   Recall(class_id=0, name='recall_0'), Precision(class_id=0, name='precision_0'), 
                                   Recall(class_id=1, name='recall_1'), Precision(class_id=1, name='precision_1'), 
                                   Recall(class_id=2, name='recall_2'), Precision(class_id=2, name='precision_2'), 
                                   Recall(class_id=3, name='recall_3'), Precision(class_id=3, name='precision_3')])

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True)

    callbacks = [cp_callback]

    model.save_weights(checkpoint_path)

    # Train the model with callback
    if model_name == "unet" or model_name == "segnet":
        history = model.fit(images[train_indexes], 
                masks[train_indexes],
                epochs=20, 
                batch_size=batch_size, 
                callbacks=callbacks,
                validation_data=(images[val_indexes], masks[val_indexes])).history
        
    elif model_name == "window":
        history = train_on_batch_windows(images[train_indexes], 
                masks[train_indexes],
                model=model,
                window_size=window_size,
                num_classes=num_classes,
                epochs=20, 
                callbacks=callbacks,
                validation_data=(images[val_indexes], masks[val_indexes]))
    plot_training_history(history, title, f"{result_path}/history.png", binary)