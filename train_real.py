#################################################################################
#
#  Project Name  : Pediatric Bone Age Prediction System (Baseline CNN)
#  Description   : Loads pediatric hand X-ray images and bone age labels from
#                   the RSNA Bone Age dataset, builds an efficient tf.data
#                   pipeline, then trains a simple from-scratch CNN regression
#                   model to predict bone age (in months) from X-rays. This
#                   serves as the baseline to compare against the transfer
#                   learning model (train_transfer.py).
#  Date          : 09-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

Border = "-" * 50

# -----------------------------
# CONFIG - update paths if your setup differs
# -----------------------------
DATA_DIR = r"C:\Users\shubh\Desktop\Deep_Learning\data"
CSV_PATH = os.path.join(DATA_DIR, "boneage-training-dataset.csv")
IMG_DIR = os.path.join(DATA_DIR, "boneage-training-dataset")

IMG_SIZE = 128
SUBSET_SIZE = 1000
BATCH_SIZE = 16
EPOCHS = 10


#################################################################################
#
# Function Name : load_data
# Input :         csv_path, img_dir
# Description :   Loads the bone age CSV, builds the full image file path for
#                 each record (id -> id.png), and prints basic dataset info
# Return Value :  DataFrame with an added 'filepath' column
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    df['filepath'] = df['id'].apply(lambda x: os.path.join(img_dir, f"{x}.png"))

    print("Total available images : ", len(df))
    print("Age range (months) : ", df['boneage'].min(), "to", df['boneage'].max())

    return df


#################################################################################
#
# Function Name : prepare_model_data
# Input :         df, subset_size
# Description :   Takes a random subset of the full dataset (for fast CPU
#                 training), then splits it into train and validation sets
# Return Value :  train_df, val_df
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def prepare_model_data(df, subset_size):
    df_subset = df.sample(n=subset_size, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(df_subset, test_size=0.2, random_state=42)

    print("Train samples : ", len(train_df))
    print("Validation samples : ", len(val_df))

    return train_df, val_df


#################################################################################
#
# Function Name : load_and_preprocess_image
# Input :         filepath, label
# Description :   Reads an X-ray image from disk, decodes it as grayscale,
#                 resizes to IMG_SIZE, and normalizes pixel values (0-255 to
#                 0-1) for stable training
# Return Value :  img, label
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_and_preprocess_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, label


#################################################################################
#
# Function Name : build_dataset
# Input :         dataframe, shuffle
# Description :   Builds an efficient tf.data pipeline from a dataframe of
#                 filepaths and labels - loads, preprocesses, shuffles,
#                 batches, and prefetches, instead of loading all images into
#                 memory upfront
# Return Value :  tf.data.Dataset ready for training/evaluation
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_dataset(dataframe, shuffle=False):
    filepaths = dataframe['filepath'].values
    labels = dataframe['boneage'].values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


#################################################################################
#
# Function Name : build_baseline_cnn
# Input :         input_shape
# Description :   Builds a simple from-scratch CNN with 3 Conv2D+MaxPooling
#                 blocks, followed by Dense layers, ending in a single
#                 output neuron (no activation) for regression - predicting
#                 bone age as a continuous number
# Return Value :  Compiled-ready Keras Sequential model
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_baseline_cnn(input_shape=(128, 128, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(1)
    ])

    return model


#################################################################################
#
# Function Name : train_model
# Input :         model, train_ds, val_ds, epochs
# Description :   Compiles the model with MSE loss (for regression) and MAE
#                 metric (for interpretability), then trains it on the
#                 training set while validating on the held-out validation set
# Return Value :  history object from model.fit
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def train_model(model, train_ds, val_ds, epochs):
    print(Border)
    print("Training : Baseline CNN")
    print(Border)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return history


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full baseline CNN pipeline end-to-end: load data,
#                 prepare datasets, build model, train, and save
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data")
    print(Border)

    df = load_data(CSV_PATH, IMG_DIR)

    print(Border)
    print("Step 2 : Prepare Train/Validation Split")
    print(Border)

    train_df, val_df = prepare_model_data(df, SUBSET_SIZE)

    print(Border)
    print("Step 3 : Build Data Pipelines")
    print(Border)

    train_ds = build_dataset(train_df, shuffle=True)
    val_ds = build_dataset(val_df, shuffle=False)

    print(Border)
    print("Step 4 : Build Baseline CNN")
    print(Border)

    model = build_baseline_cnn()
    model.summary()

    print(Border)
    print("Step 5 : Train Model")
    print(Border)

    history = train_model(model, train_ds, val_ds, EPOCHS)

    print(Border)
    print("Step 6 : Final Results")
    print(Border)

    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    print(f"Final Training MAE : {final_train_mae:.2f} months")
    print(f"Final Validation MAE : {final_val_mae:.2f} months")

    model.save("baseline_model.keras")
    print("Model saved as baseline_model.keras")


if __name__ == "__main__":
    main()