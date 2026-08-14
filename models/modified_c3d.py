from tensorflow import keras
from tensorflow.keras import layers, regularizers

INPUT_SHAPE = (16, 112, 112, 6)

def build_modified_c3d(
    input_shape=INPUT_SHAPE,
    hidden_units=1024,
    dropout=0.30,
    l2_strength=1e-6,
):
    reg = regularizers.l2(l2_strength)

    inputs = keras.Input(
        shape=input_shape,
        name="rgb_flow_clip",
    )

    x = inputs

    for i, (filters, pool) in enumerate(
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
            (3, 3, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name=f"conv3d_{i}_{filters}",
        )(x)
        x = layers.BatchNormalization(
            name=f"bn_{i}"
        )(x)
        x = layers.Activation(
            "swish",
            name=f"swish_{i}",
        )(x)
        x = layers.MaxPooling3D(
            pool_size=pool,
            strides=pool,
            name=f"pool_{i}",
        )(x)

    avg = layers.GlobalAveragePooling3D(
        name="global_avg"
    )(x)

    mx = layers.GlobalMaxPooling3D(
        name="global_max"
    )(x)

    x = layers.Concatenate(
        name="avg_max_concat"
    )([avg, mx])

    # One hidden fully connected layer.
    x = layers.Dense(
        hidden_units,
        activation="swish",
        kernel_regularizer=reg,
        name="hidden_fc",
    )(x)

    x = layers.Dropout(
        dropout,
        name="dropout",
    )(x)

    # Two-class SoftMax output.
    outputs = layers.Dense(
        2,
        activation="softmax",
        dtype="float32",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs,
        outputs,
        name="modified_c3d_rgbflow",
    )
