import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFilter
import argparse

print(tf.__version__)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image input")
ap.add_argument("-t", "--thres", required=False, default=0.2, help="threshold for binary conversion", type=float)
args = vars(ap.parse_args())
args["image"] = "imgs/" + args["image"]

# loading mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train[:10000]
y_test  = y_test[:10000]
x_train, x_test = x_train[:10000] / 255.0, x_test[:10000] / 255.0

# creating model function
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

# creating fresh model
model = create_model()

# for visualisation

# print(x_train[0])
# print(y_train[0])
# plt.imshow(x_train[0], interpolation='nearest')
# plt.show()

# predictions = model(x_train[:1]).numpy()
# print(predictions)

# tf.nn.softmax(predictions).numpy()

# loss function
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# print(loss_fn(y_train[:1], predictions).numpy())

model.summary()
print(type(model))

# loading image
# img = mpimg.imread('num1.jpg')
grayimg = Image.open(args["image"]).convert('LA')
print(grayimg.size)
newImage = Image.new('L', (28,28), 255)
width = grayimg.size[0]
height = grayimg.size[1]
nheight, nwidth = 28, 28
newImage = grayimg.resize((28,28))
# if width > height:
#     nheight = int(round(20.0/width * height, 0))
#     if nheight == 0:
#         nheight = 1
#     img = grayimg.resize((20,nheight),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#     wtop = int(round((28-nheight)/2, 0))
#     newImage.paste(img, (4, wtop))
# else:
#     # Height is bigger. Heght becomes 20 pixels.
#     nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
#     if (nwidth == 0):  # rare case but minimum is 1 pixel
#         nwidth = 1
#         # resize and sharpen
#     img = grayimg.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#     wleft = int(round(((28 - nwidth) / 2), 0))  # calculate vertical position
#     newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
tv = list(newImage.getdata())
tva = [(255 - x[0]) / 255.0 for x in tv]
# tva = [(255 - x) * 1.0 / 255.0 for x in tv]
print(tva)
print(len(tva))

tva = [x if x > args["thres"] else 0 for x in tva]
finimg = np.array(tva).reshape(28,28)
plt.imshow(finimg)
plt.show()


# model.fit(x_train, y_train, epochs=5)

# creating model checkpoint
checkpoint_path = "training_1/cp.cpkt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

# model.fit(x_train, y_train,epochs=10, validation_data=(x_test, y_test), callbacks=[cp_callback])

# loss, acc = model.evaluate(x_test, y_test, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path)
outputlayer = model(np.array(tva).reshape(-1,28,28))
foutlayer = tf.nn.softmax(outputlayer)
output = np.argmax(foutlayer[0])
print(np.shape(outputlayer))
print(f'{output} : {foutlayer[0][output]*100:.2f}%')
# loss, acc = model.evaluate(x_test, y_test, verbose=2)

# print("Restored model, accuracy: {:5.2f}%".format(100*acc))