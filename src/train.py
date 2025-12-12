import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from scripts.loader import BeejXDataLoader
from models.mobilenet import build_mobilenet_model
from utils.logger import get_logger

logger = get_logger(__name__)

def get_class_weights(train_ds):
    """Computes class weights to handle data imbalance."""
    logger.info("Computing class weights...")
    y_train = []
    try:
        # Iterate over the dataset to collect labels
        for _, labels in train_ds.unbatch():
            y_train.append(labels.numpy())
            
        y_train = np.array(y_train)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        return dict(enumerate(weights))
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}. Using equal weights.")
        return None

def train_initial_model(model, train_ds, val_ds, class_weights, config, callbacks):
    """Trains the base model."""
    logger.info("Starting initial training...")
    history = model.fit(
        train_ds, 
        epochs=config['training']['epochs'], 
        validation_data=val_ds, 
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history

def fine_tune_model(model, train_ds, val_ds, class_weights, config, callbacks, initial_epoch):
    """Fine-tunes the model by unfreezing top layers."""
    if not config['fine_tuning']['enabled']:
        return None

    logger.info("Starting fine-tuning...")
    
    # Unfreeze the base model
    base_model = model.layers[1] 
    base_model.trainable = True
    
    # Freeze early layers
    unfreeze_from = config['fine_tuning']['unfreeze_from_layer']
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['fine_tuning']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_epochs = config['training']['epochs'] + config['fine_tuning']['epochs']
    
    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history_fine

def export_model(model, class_names, output_dir):
    """Exports the model to TFLite format."""
    logger.info(f"Exporting model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save generic SavedModel
    model.save(os.path.join(output_dir, "saved_model"))

    # Convert to TFLite with Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(output_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    # Save Labels
    with open(os.path.join(output_dir, "labels.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    logger.info(f"Model exported successfully: {tflite_path}")

def main():
    logger.info("Initializing Training Pipeline")
    
    from scripts.loader import load_config
    config = load_config()
    
    # Prepare Data
    loader = BeejXDataLoader(config)
    datasets = loader.get_combined_dataset()
    
    if datasets is None:
        logger.error("Training data not found. Please check data/raw directory.")
        return
        
    train_ds, val_ds, class_names = datasets
    num_classes = len(class_names)
    logger.info(f"Classes found: {num_classes}")

    # Calculate Weights
    class_weights = get_class_weights(train_ds)

    # Build Model
    model = build_mobilenet_model(num_classes, config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config['paths']['output_dir'], "best_model.keras"),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]

    # Train
    history = train_initial_model(model, train_ds, val_ds, class_weights, config, callbacks)
    
    # Fine-tune
    fine_tune_model(model, train_ds, val_ds, class_weights, config, callbacks, history.epoch[-1])

    # Export
    export_model(model, class_names, config['paths']['output_dir'])

if __name__ == "__main__":
    main()
