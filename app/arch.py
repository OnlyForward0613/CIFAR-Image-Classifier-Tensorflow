from tensorflow.keras import layers, models


def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Now we have the feature extracted, next is classification by passing the features to dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=10))
    return model


