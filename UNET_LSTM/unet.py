from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, ConvLSTM2D, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf


def conv_block(inputs, num_filters):
    x = Reshape((1, inputs.shape[1], inputs.shape[2], inputs.shape[3]))(inputs)
    x = ConvLSTM2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x1 = Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
    x1 = ConvLSTM2D(num_filters, 3, padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    return x1


def conv_block1(x, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def encoder_block1(inputs, num_filters):
    x = conv_block1(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def convlstm_block(inputs, num_filters):
    # Reshape inputs to add a time dimension
    x = tf.expand_dims(inputs, axis=1)  # Add time dimension
    x = ConvLSTM2D(num_filters, 3, padding="same", return_sequences=True)(x)
    x = tf.squeeze(x, axis=1)  # Remove the time dimension
    return x


feedback = None


def build_unet(input_shape):
    inputs = Input(input_shape)
    global feedback
    if feedback is not None:
        conInputs = Concatenate()([inputs, feedback])
    else:
        conInputs = inputs

    s1, p1 = encoder_block(conInputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block1(p2, 256)
    s4, p4 = encoder_block1(p3, 512)

    # Apply ConvLSTM on the bottleneck layer
    b1_lstm = convlstm_block(p4, 512)

    print(s1.shape, s2.shape, s3.shape, s4.shape)
    print(p1.shape, p2.shape, p3.shape, p4.shape)

    d1 = decoder_block(b1_lstm, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    print(d1.shape, d2.shape, d3.shape, d4.shape)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    print(outputs.shape)

    feedback = outputs

    model = Model(inputs, outputs, name="UNET")
    return model


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()