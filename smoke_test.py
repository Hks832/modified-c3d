from models.four_conv3d import build_model


def main():
    model = build_model()

    conv3d_count = sum(
        layer.__class__.__name__ == "Conv3D"
        for layer in model.layers
    )
    dense_count = sum(
        layer.__class__.__name__ == "Dense"
        for layer in model.layers
    )

    assert model.input_shape == (None, 16, 112, 112, 6)
    assert model.output_shape == (None, 2)
    assert conv3d_count == 4
    assert dense_count == 2

    print("Smoke test passed.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Conv3D layers:", conv3d_count)
    print("Dense layers:", dense_count)


if __name__ == "__main__":
    main()
