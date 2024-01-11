import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D



def load_model(input_shape, n_classes, model_path):
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = keras.applications.ResNet50(include_top=False,
                     weights=None,
                     input_shape=input_shape)


    top_model = conv_base(x)
    # Create a new 'top' of the model (i.e. fully-connected layers).
    top_model = GlobalAveragePooling2D()(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=inputs, outputs=output_layer)
    model.trainable = False
    model.load_weights(model_path)

    return model

