import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image


model_object = load_model('models/model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


def show_image(img, label_correct, guess):
    img = (img * 255).astype(np.uint8)  # Scale the image values to [0, 255]
    img_pil = Image.fromarray(img)  # Convert to PIL Image
    plt.rcParams['text.color'] = 'green'
    plt.rcParams['axes.labelcolor'] = 'red'
    plt.figure()
    plt.imshow(img_pil)
    plt.title("Expected: " + label_correct)
    plt.xlabel("Guess : " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_image_number_choice():
    while True:
        number = input("Pick a number: ")
        if number.isdigit():
            number = int(number)
            if 0 <= number <= 1000:
                return int(number)
            else:
                print("Try again!")


def predict(model, image, correct_label):
    image = image / 255.0
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    print(predicted_class)
    show_image(image*255, class_names[correct_label[0]], predicted_class)


num = get_image_number_choice()
image_ = test_images[num]
label = test_labels[num]
predict(model=model_object, image=image_, correct_label=label)