import tensorflow as tf
import os
import glob
from typing import Tuple, List, Optional
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise

class BeejXDataLoader:
    """
    Handles data loading and augmentation for the BeejX Leaf Model.
    """
    def __init__(self, config: dict):
        self.config = config
        self.img_size = tuple(config['model']['input_shape'][:2])
        self.batch_size = config['training']['batch_size']
        
        # Define Augmentation Layers
        self.augment_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config['augmentation']['rotation_range']),
            tf.keras.layers.RandomZoom(config['augmentation']['zoom_range']),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])

    def get_local_dataset(self) -> Optional[tf.data.Dataset]:
        """Loads local dataset from data/raw folder with Augmentation."""
        data_dir = self.config['data']['local_data_dir']
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return None

        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not classes:
            logger.error(f"No class folders found in {data_dir}")
            return None
            
        logger.info(f"detected {len(classes)} classes: {classes}")
        logger.info(f"Loading data from {data_dir}...")
        
        # Load Training Data
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config['training']['validation_split'],
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
            labels='inferred',
            label_mode='int'
        )
        
        # Load Validation Data
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config['training']['validation_split'],
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
            labels='inferred',
            label_mode='int' 
        )

        class_names = train_ds.class_names
        # Attach class names to the dataset object for easy access in train.py
        train_ds.class_names = class_names

        # Apply Augmentation to Training only
        train_ds = train_ds.map(lambda x, y: (self.augment_layers(x, training=True), y), 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        
        # MobileNetV2 Preprocessing
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds, class_names

    def get_combined_dataset(self):
        """Returns Local Train, Validation datasets and Class Names."""
        return self.get_local_dataset()
