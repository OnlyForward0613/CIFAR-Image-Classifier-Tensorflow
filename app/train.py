from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

from app.arch import get_model

# load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

'''
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[7][0]])
plt.show()'''

model = get_model()
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
model.save('models/model.h5')

# evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy : ", test_acc*100)

