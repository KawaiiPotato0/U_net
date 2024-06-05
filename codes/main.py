from train_model import train
from evaluate_model import evaluate
from plot_prediction import predict
import matplotlib as mpl

# BINARY
# nothing - 0
# data: brain_tumor_dataset/nothing
# results: UNET_BINARY_SEGMENTATION/NOTHING

# normalized - 1
# data: brain_tumor_dataset/normalized
# results: UNET_BINARY_SEGMENTATION/NORMALIZATION

# augumented - 2
# data: brain_tumor_dataset/augumented
# results: UNET_BINARY_SEGMENTATION/AUGUMENTATION

# norm_aug - 3
# data: brain_tumor_dataset/norm_aug
# results: UNET_BINARY_SEGMENTATION/NORM_AUG

# MULTICLASS
# normalized - 4
# data: brain_tumor_dataset/normalized
# results: UNET_MULTICLASS/NORMALIZATION

# norm_aug - 5
# data: brain_tumor_dataset/norm_aug
# results: UNET_MULTICLASS/NORM_AUG

data_paths = ["brain_tumor_dataset/nothing",
              "brain_tumor_dataset/normalized",
              "brain_tumor_dataset/augumented",
              "brain_tumor_dataset/norm_aug",
              "brain_tumor_dataset/norm_aug_gray",
              "brain_tumor_dataset/normalized",
              "brain_tumor_dataset/norm_aug",
              "brain_tumor_dataset/norm_aug_gray",
              "brain_tumor_dataset/nothing",
              "brain_tumor_dataset/normalized",
              "brain_tumor_dataset/augumented",
              "brain_tumor_dataset/norm_aug",
              "brain_tumor_dataset/norm_aug_gray",
              "brain_tumor_dataset/normalized",
              "brain_tumor_dataset/norm_aug",
              "brain_tumor_dataset/norm_aug_gray"]

result_paths = ["UNET_BINARY_SEGMENTATION/NOTHING",
                "UNET_BINARY_SEGMENTATION/NORMALIZATION",
                "UNET_BINARY_SEGMENTATION/AUGUMENTATION",
                "UNET_BINARY_SEGMENTATION/NORM_AUG",
                "UNET_BINARY_SEGMENTATION/NORM_AUG_GRAY",
                "UNET_MULTICLASS/NORMALIZATION",
                "UNET_MULTICLASS/NORM_AUG",
                "UNET_MULTICLASS/NORM_AUG_GRAY",
                "SEGNET_BINARY_SEGMENTATION/NOTHING",
                "SEGNET_BINARY_SEGMENTATION/NORMALIZATION",
                "SEGNET_BINARY_SEGMENTATION/AUGUMENTATION",
                "SEGNET_BINARY_SEGMENTATION/NORM_AUG",
                "SEGNET_BINARY_SEGMENTATION/NORM_AUG_GRAY",
                "SEGNET_MULTICLASS/NORMALIZATION",
                "SEGNET_MULTICLASS/NORM_AUG",
                "SEGNET_MULTICLASS/NORM_AUG_GRAY"]

for i in [7]:
    combination = i
    binary = True if combination <= 4 or 8 <= combination <= 12 else False
    model = "unet" if combination <= 7 else "segnet"
    if binary:
        title = f"Binary {model.capitalize()}"
    else: 
        title = f"Multiclass {model.capitalize()}"
    random_state = 11
    window_size = 39

    mpl.rcParams.update(mpl.rcParamsDefault)
    #train(data_paths[combination], result_paths[combination], title, model, binary, random_state, window_size)
    evaluate(data_paths[combination], result_paths[combination], model, binary, random_state, window_size)
    predict(data_paths[combination], result_paths[combination], model, binary, window_size)