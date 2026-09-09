#################################################################################
#
#  Project Name  : Pediatric Bone Age Prediction System (Transfer Learning)
#  Description   : Loads pediatric hand X-ray images and bone age labels from
#                   the RSNA Bone Age dataset, builds an efficient tf.data
#                   pipeline, then trains a CNN regression model using
#                   transfer learning (EfficientNetB0 pretrained on ImageNet)
#                   in two phases - frozen base training followed by
#                   fine-tuning - to predict bone age (in months) from X-rays.
#  Date          : 09-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
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
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 5


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
#                 resizes to IMG_SIZE, and converts to RGB (3-channel) since
#                 EfficientNetB0 expects RGB input
# Return Value :  img, label
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_and_preprocess_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.grayscale_to_rgb(img)
    return img, label


#################################################################################
#
# Function Name : augment_image
# Input :         img, label
# Description :   Applies random horizontal flip and brightness change to
#                 training images only, to help the model generalize instead
#                 of memorizing the limited training set
# Return Value :  img, label
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, label


#################################################################################
#
# Function Name : build_dataset
# Input :         dataframe, shuffle, augment
# Description :   Builds an efficient tf.data pipeline from a dataframe of
#                 filepaths and labels - loads, preprocesses, optionally
#                 augments, shuffles, batches, and prefetches
# Return Value :  tf.data.Dataset ready for training/evaluation
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_dataset(dataframe, shuffle=False, augment=False):
    filepaths = dataframe['filepath'].values
    labels = dataframe['boneage'].values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


#################################################################################
#
# Function Name : build_transfer_model
# Input :         input_shape
# Description :   Builds a transfer learning model using EfficientNetB0
#                 (pretrained on ImageNet, top classification layer removed)
#                 as a frozen feature extractor, with a custom regression
#                 head added on top for bone age prediction
# Return Value :  model, base_model (base_model returned separately so it can
#                 be unfrozen later for fine-tuning)
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_transfer_model(input_shape=(128, 128, 3)):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)

    return model, base_model


#################################################################################
#
# Function Name : train_phase1
# Input :         model, train_ds, val_ds, epochs
# Description :   Phase 1 training - base model is frozen, only the custom
#                 regression head is trained. Uses a standard learning rate
#                 since we're training new layers from scratch
# Return Value :  history object from model.fit
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def train_phase1(model, train_ds, val_ds, epochs):
    print(Border)
    print("Phase 1 : Training custom head only (base model frozen)")
    print(Border)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return history


#################################################################################
#
# Function Name : train_phase2
# Input :         model, base_model, train_ds, val_ds, epochs
# Description :   Phase 2 training - unfreezes the base model and fine-tunes
#                 the entire network with a much smaller learning rate, so
#                 the pretrained ImageNet weights are gently adjusted rather
#                 than destroyed
# Return Value :  history object from model.fit
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def train_phase2(model, base_model, train_ds, val_ds, epochs):
    print(Border)
    print("Phase 2 : Fine-tuning (base model unfrozen)")
    print(Border)

    base_model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return history


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full bone age transfer learning pipeline end-to-end:
#                 load data, prepare datasets, build model, train in two
#                 phases, and save the final model
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

    train_ds = build_dataset(train_df, shuffle=True, augment=True)
    val_ds = build_dataset(val_df, shuffle=False, augment=False)

    print(Border)
    print("Step 4 : Build Transfer Learning Model")
    print(Border)

    model, base_model = build_transfer_model()
    model.summary()

    print(Border)
    print("Step 5 : Train Model (Two Phases)")
    print(Border)

    train_phase1(model, train_ds, val_ds, PHASE1_EPOCHS)
    history2 = train_phase2(model, base_model, train_ds, val_ds, PHASE2_EPOCHS)

    print(Border)
    print("Step 6 : Final Results")
    print(Border)

    final_mae = history2.history['val_mae'][-1]
    print(f"Final Validation MAE : {final_mae:.2f} months")

    model.save("transfer_model.keras")
    print("Model saved as transfer_model.keras")


if __name__ == "__main__":
    main()