from tensorflow import keras
from tensorflow.keras import layers, regularizers

INPUT_SHAPE = (16, 112, 112, 6)


def build_model(
    input_shape=INPUT_SHAPE,
    hidden_units=1024,
    dropout=0.30,
    l2_strength=1e-6,
):
    """Build the fixed four-layer 3D-CNN used by the experiments."""
    reg = regularizers.l2(l2_strength)

    inputs = keras.Input(shape=input_shape, name="rgb_flow_clip")
    x = inputs

    for index, (filters, pool_size) in enumerate(
        [
            (64, (1, 2, 2)),
            (128, (2, 2, 2)),
            (256, (2, 2, 2)),
            (512, (2, 2, 2)),
        ],
        start=1,
    ):
        x = layers.Conv3D(
            filters,
            kernel_size=(3, 3, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name=f"conv3d_{index}",
        )(x)
        x = layers.BatchNormalization(name=f"batch_norm_{index}")(x)
        x = layers.Activation("swish", name=f"swish_{index}")(x)
        x = layers.MaxPooling3D(
            pool_size=pool_size,
            strides=pool_size,
            name=f"pool_{index}",
        )(x)

    average_features = layers.GlobalAveragePooling3D(
        name="global_average_pool"
    )(x)
    maximum_features = layers.GlobalMaxPooling3D(
        name="global_max_pool"
    )(x)

    x = layers.Concatenate(name="global_pool_concat")(
        [average_features, maximum_features]
    )
    x = layers.Dense(
        hidden_units,
        activation="swish",
        kernel_regularizer=reg,
        name="hidden_fc",
    )(x)
    x = layers.Dropout(dropout, name="dropout")(x)

    outputs = layers.Dense(
        2,
        activation="softmax",
        dtype="float32",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="four_conv3d_rgbflow",
    )
