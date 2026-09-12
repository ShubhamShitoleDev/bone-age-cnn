#################################################################################
#
#  Project Name  : Pediatric Bone Age Prediction System (Grad-CAM Explainability)
#  Description   : Loads the trained transfer learning model and generates
#                   Grad-CAM heatmaps for sample X-ray images, showing which
#                   regions of the image (ideally growth plates) the model
#                   focused on to make its bone age prediction. Saves
#                   side-by-side comparisons of original X-ray vs heatmap
#                   overlay for use in the resume/portfolio README.
#  Date          : 09-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

Border = "-" * 50

# -----------------------------
# CONFIG - update paths if your setup differs
# -----------------------------
DATA_DIR = r"C:\Users\shubh\Desktop\Deep_Learning\data"
CSV_PATH = os.path.join(DATA_DIR, "boneage-training-dataset.csv")
IMG_DIR = os.path.join(DATA_DIR, "boneage-training-dataset")
MODEL_PATH = "transfer_model.keras"
OUTPUT_DIR = "gradcam_outputs"

IMG_SIZE = 128
NUM_SAMPLES_TO_VISUALIZE = 5


#################################################################################
#
# Function Name : load_and_prepare_image
# Input :         filepath
# Description :   Loads a single X-ray image from disk, resizes it, converts
#                 to RGB, and returns both the raw display version (0-255)
#                 and the model-ready preprocessed version (batched)
# Return Value :  display_img (for plotting), model_input (for prediction)
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_and_prepare_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.grayscale_to_rgb(img)

    display_img = tf.cast(img, tf.uint8).numpy()

    model_input = tf.expand_dims(img, axis=0)

    return display_img, model_input


#################################################################################
#
# Function Name : find_last_conv_layer
# Input :         model
# Description :   Automatically finds the name of the last convolutional
#                 layer inside the EfficientNetB0 base model - Grad-CAM needs
#                 this layer's feature maps and gradients
# Return Value :  Name of the last convolutional layer (string)
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def find_last_conv_layer(model):
    # Find the EfficientNetB0 base model by name/type instead of a fixed
    # index, since the exact layer ordering can vary depending on how the
    # functional model was built
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # nested model (e.g. EfficientNetB0)
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find a nested base model (e.g. EfficientNetB0) "
                          "inside the loaded model. Print model.summary() to inspect.")

    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            return base_model, layer.name

    raise ValueError("Could not find a convolutional layer in the base model.")


#################################################################################
#
# Function Name : generate_gradcam_heatmap
# Input :         model, base_model, last_conv_layer_name, model_input
# Description :   Computes the Grad-CAM heatmap for a given input image -
#                 gets the gradient of the model's output with respect to
#                 the last conv layer's feature maps, weights the feature
#                 maps by these gradients, and produces a 2D heatmap
# Return Value :  2D numpy heatmap, normalized between 0 and 1
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def generate_gradcam_heatmap(model, base_model, last_conv_layer_name, model_input):
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # Find the exact position of base_model within the outer model's layers,
    # so we know precisely which layers make up the "head" (Dense/Dropout/Dense)
    # - this avoids guessing a fixed index, since preprocess_input can expand
    # into several internal layers before the base_model layer
    head_layers = None
    for i, layer in enumerate(model.layers):
        if layer is base_model:
            head_layers = model.layers[i + 1:]
            break

    if head_layers is None:
        raise ValueError("Could not locate base_model within the outer model's layers.")

    # Run the preprocessing steps manually (same as inside build_transfer_model)
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(model_input)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(preprocessed)
        # Pass the pooled base output through the custom head to get final prediction
        x = base_output
        for layer in head_layers:
            x = layer(x)
        prediction = x

    grads = tape.gradient(prediction, conv_outputs)

    # Average the gradients across width/height - gives one weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


#################################################################################
#
# Function Name : overlay_heatmap
# Input :         display_img, heatmap, alpha
# Description :   Resizes the heatmap to match the original image size,
#                 applies a color map (jet - blue to red), and overlays it
#                 on top of the original grayscale X-ray for visualization
# Return Value :  Overlayed image as a numpy array (uint8)
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def overlay_heatmap(display_img, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], [IMG_SIZE, IMG_SIZE]
    ).numpy().squeeze()

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    overlayed = jet_heatmap * alpha + display_img * (1 - alpha)
    overlayed = np.uint8(overlayed)

    return overlayed


#################################################################################
#
# Function Name : visualize_predictions
# Input :         model, df, num_samples
# Description :   Picks random sample images, generates Grad-CAM heatmaps for
#                 each, and saves a side-by-side plot (original vs overlay)
#                 with actual age, predicted age, and error for each sample
# Return Value :  None (saves image files to OUTPUT_DIR)
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def visualize_predictions(model, df, num_samples):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_model, last_conv_layer_name = find_last_conv_layer(model)
    print("Using last conv layer : ", last_conv_layer_name)

    samples = df.sample(n=num_samples, random_state=1)

    for idx, row in samples.iterrows():
        filepath = row['filepath']
        actual_age = row['boneage']

        display_img, model_input = load_and_prepare_image(filepath)

        predicted_age = model.predict(model_input, verbose=0)[0][0]

        heatmap = generate_gradcam_heatmap(model, base_model, last_conv_layer_name, model_input)
        overlayed = overlay_heatmap(display_img, heatmap)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(display_img)
        axes[0].set_title("Original X-ray")
        axes[0].axis('off')

        axes[1].imshow(overlayed)
        axes[1].set_title(f"Grad-CAM\nActual: {actual_age}mo | Predicted: {predicted_age:.1f}mo")
        axes[1].axis('off')

        output_path = os.path.join(OUTPUT_DIR, f"gradcam_{row['id']}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved : {output_path}  (Actual: {actual_age}, Predicted: {predicted_age:.1f}, "
              f"Error: {abs(actual_age - predicted_age):.1f} months)")


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Loads the trained model and dataset, then generates and
#                 saves Grad-CAM visualizations for a handful of sample
#                 X-rays to demonstrate model interpretability
# Date :          09-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load Trained Model")
    print(Border)

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded from : ", MODEL_PATH)

    print(Border)
    print("Step 2 : Load Dataset")
    print(Border)

    df = pd.read_csv(CSV_PATH)
    df['filepath'] = df['id'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.png"))
    print("Total images available : ", len(df))

    print(Border)
    print("Step 3 : Generate Grad-CAM Visualizations")
    print(Border)

    visualize_predictions(model, df, NUM_SAMPLES_TO_VISUALIZE)

    print(Border)
    print(f"Done. Check the '{OUTPUT_DIR}' folder for saved visualizations.")
    print(Border)


if __name__ == "__main__":
    main()