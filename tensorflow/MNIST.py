import setuptools.dist
import tensorflow as tf
from tensorflow.keras import datasets, layers # type: ignore
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(10, kernel_size=5, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(20, kernel_size=5, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(rate=0.5),
    layers.Flatten(),
    layers.Dense(50, activation='relu'),
    layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

accuracy = model.evaluate(x_test,  y_test)
print(f"Dev Accuracy: {accuracy[1]:.4f}, Dev Loss: {accuracy[0]:.4f}")

def test_prediction(model, test_data, target_data, idx):
    data = test_data[idx]
    target = target_data[idx]
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)

    output = model(data)
    prediction = np.argmax(output.numpy(), axis=1)[0]
    image = data.squeeze(0)

    plt.figure()
    plt.imshow(image, cmap="gray", interpolation='nearest')
    plt.title(f"Prediction: {prediction}, Label: {target}")
    plt.show()

total_tests = 4

for i in range(total_tests):
    test_prediction(model, x_test, y_test, i)
import numpy as np

def test_prediction(model, test_data, target_data, idx):
    data = test_data[idx]
    target = target_data[idx]
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)

    output = model(data)
    prediction = np.argmax(output.numpy(), axis=1)[0]
    image = data.squeeze(0)

    plt.figure()
    plt.imshow(image, cmap="gray", interpolation='nearest')
    plt.title(f"Prediction: {prediction}, Label: {target}")
    plt.show()

total_tests = 4

for i in range(total_tests):
    test_prediction(model, x_test, y_test, i)
