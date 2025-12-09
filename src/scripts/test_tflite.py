import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os

def test_tflite_model(model_path, labels_path, image_path):
    print(f"Loading model: {model_path}...")
    try:
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Load Labels
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Prepare Image
        print(f"Processing image: {image_path}...")
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        input_data = np.array(img, dtype=np.float32)
        input_data = input_data / 255.0  # Normalize to [0, 1]
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get Output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = np.squeeze(output_data)
        top_k_indices = predictions.argsort()[-3:][::-1] # Top 3

        print("\n" + "="*30)
        print("🌱 PREDICTION RESULTS")
        print("="*30)
        for i in top_k_indices:
            print(f"{labels[i]}: {predictions[i]*100:.2f}%")
        print("="*30 + "\n")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/scripts/test_tflite.py <path_to_model.tflite> <path_to_labels.txt> <path_to_image.jpg>")
        print("Example: python src/scripts/test_tflite.py exports/model.tflite exports/labels.txt data_processed/Mandua_blast/test.jpg")
    else:
        test_tflite_model(sys.argv[1], sys.argv[2], sys.argv[3])
