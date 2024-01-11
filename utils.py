import tensorflow as tf
import argparse
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

def dataloader(path, batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        seed=123,
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=batch_size)
    
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test set folder")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size", default=32)
    parser.add_argument("--model_path", type=str, required=False, default="./model.h5", help="Path to the model.h5")
    args = parser.parse_args()

    ds = dataloader(args.dataset_path, args.batch_size)
    model = load_model((224, 224, 3), 2, args.model_path)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.evaluate(ds)
