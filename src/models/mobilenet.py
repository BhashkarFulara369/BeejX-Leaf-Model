import tensorflow as tf

def build_mobilenet_model(num_classes: int, config: dict):
    """
    Builds a MobileNetV2 model with a custom classification head.
    """
    input_shape = config['model']['input_shape']
    alpha = config['model']['alpha']
    weights = config['model']['weights']
    dropout_rate = config['model']['dropout']

    print(f"Building MobileNetV2 with alpha={alpha}, input_shape={input_shape}")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        alpha=alpha
    )

    # Freeze base model (optional: unfreeze later for fine-tuning)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model
