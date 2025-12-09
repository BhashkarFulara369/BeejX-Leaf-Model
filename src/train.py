import os
import tensorflow as tf
import numpy as np
import yaml
from scripts.loader import BeejXDataLoader
from models.mobilenet import build_mobilenet_model

def load_config(path='configs/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("="*50)
    print("BeejX Leaf Disease Model - Professional Training Pipeline")
    print("="*50)

    # 1. Load Configuration
    config = load_config()
    print("Configuration loaded.")

    # 2. Prepare Data
    loader = BeejXDataLoader(config)
    datasets = loader.get_combined_dataset()

    if datasets is None:
        print("\n[ERROR] No training data found!")
        print("Please add images to: c:/Project/CDD/LeafModel/data/raw/")
        print("Structure: data/raw/CropName/DiseaseName/image.jpg")
        return
        
    train_ds, val_ds, class_names = datasets # Unpack tuple
    # class_names = train_ds.class_names # REMOVED: Caused AttributeError on PrefetchDataset
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 3. Build Model
    model = build_mobilenet_model(num_classes, config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4.1 Calculate Class Weights (Professional Handling of Imbalance)
    print("\nCalculating class weights to balance dataset...")
    
    y_train = []
    # Iterate only a subset or full dataset if small (Local is usually small)
    print("Scanning for labels (this might take a moment)...")
    try:
        # We unbatch to get exact labels. 
        # CAUTION: If dataset is huge, this is slow. For <10GB local data, it's fine.
        for _, labels in train_ds.unbatch():
            y_train.append(labels.numpy())
            
        from sklearn.utils import class_weight
        y_train = np.array(y_train)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(weights))
        print(f"Computed Class Weights: {class_weights}")
    except Exception as e:
        print(f"[WARNING] Could not compute class weights: {e}")
        print("Falling back to equal weights.")
        class_weights = None

    # 4. Train
    print("\nStarting training...")
    
    # Define Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config['paths']['output_dir'], "best_model.keras"),
        save_best_only=True,
        monitor='val_accuracy'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    epochs = config['training']['epochs']
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=val_ds, 
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    # 5. Export to TFLite
    print("\nExporting to TFLite...")
    export_dir = config['paths']['output_dir']
    os.makedirs(export_dir, exist_ok=True)
    
    # Save generic SavedModel
    model.save(os.path.join(export_dir, "saved_model"))

    # Convert with Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantization
    tflite_model = converter.convert()

    tflite_path = os.path.join(export_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    # Save Labels
    with open(os.path.join(export_dir, "labels.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"\nSUCCESS! Model saved to: {tflite_path}")
    print(f"Labels saved to: {os.path.join(export_dir, 'labels.txt')}")
    print("Copy 'model.tflite' and 'labels.txt' to your Android project's assets folder.")

if __name__ == "__main__":
    main()
