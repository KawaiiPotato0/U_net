import numpy as np
from sklearn.model_selection import train_test_split
from model import segnet, unet_multiclass, unet, create_sliding_window_model
from keras.optimizers import Adam
from utils import DiceCoefficient, BalancedAccuracyScore, evaluate_on_batch_windows, predict_on_batch_windows
from keras.metrics import Recall, Precision
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(data_path, weights_path, model_name, binary=True, random_state=11, window_size=31):
    checkpoint_path = f"{weights_path}/cp.weights.h5"

    images = np.load(f"{data_path}/images.npy", allow_pickle=True)
    masks = np.load(f"{data_path}/masks.npy", allow_pickle=True)
    labels = np.load(f"{data_path}/labels.npy", allow_pickle=True)

    num_classes = 4 # 0: 'no tumor', 1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'

    if not binary:
        # Encode classes: 1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'
        masks = np.array([np.array(np.array(img, dtype=np.uint8)*labels[i], dtype=np.float32) for i, img in enumerate(masks)])
        
    else:
        num_classes=2
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

    # Create a basic model instance
    if model_name == "unet":
        if binary:
            model = unet((128, 128, 1))
            model.summary()
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[DiceCoefficient(num_classes=2), BalancedAccuracyScore(num_classes=2),
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
        
    # Loads the weights
    model.load_weights(checkpoint_path)

    # Evaluate the model
    if model_name == "unet" or model_name=="segnet":
        metrics = model.evaluate(images[test_indexes], masks[test_indexes])
    elif model_name == "window":
        metrics = evaluate_on_batch_windows(images[test_indexes], masks[test_indexes], model, window_size, num_classes)
    
    # Predicted probabilities for each class
    if model_name == "unet" or model_name=="segnet":
        pred_prob = model.predict(images[test_indexes])  
    elif model_name == "window":
        pred_prob = predict_on_batch_windows(images[test_indexes], masks[test_indexes], model, window_size, num_classes)
    
    #Create confusion matrix and normalizes it over predicted (columns)
    conf_masks = np.argmax(masks[test_indexes], axis=-1).reshape(-1, 128*128).reshape(-1)
    conf_pred = np.argmax(pred_prob, axis=-1).reshape(-1, 128*128).reshape(-1)


    plt.clf()
    sns.set(font_scale=1)
    print(weights_path)
    conf_matrix = confusion_matrix(conf_masks, conf_pred, normalize='pred')
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f')

    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.savefig(f"{weights_path}/confusion_matrix.png", bbox_inches='tight')

    with open(f"{weights_path}/metrics.txt", 'w') as file:
        for i, metric in enumerate(metrics):
            file.write(f'{model.metrics_names[i]}: {metric}\n')

