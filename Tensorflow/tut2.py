import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFilter
import argparse
print(tf.__version__)

# setting command line params
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image input")
ap.add_argument("-t", "--thres", required=False, default=0.2, help="threshold for binary conversion", type=float)
args = vars(ap.parse_args())
args["image"] = "imgs/" + args["image"]

# function for loading mnist dataset
def loadData():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # y_train = y_train[:10000]
    # y_test  = y_test[:10000]
    # x_train, x_test = x_train[:10000] / 255.0, x_test[:10000] / 255.0
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


# function for loading and preparing image
def readImg(inpImg, verbose=True, filter=True):
    img = Image.open(inpImg).convert('LA')
    # print(img.size)
    newImg = img.resize((28,28))
    tempArr = np.array(newImg.getdata())
    imgArr = ((255 - tempArr)/255.0)[:, :1]
    if filter:
        zeroIndxs = imgArr < args["thres"]
        imgArr[zeroIndxs] = 0
    finImg = imgArr.reshape(28, 28)
    if verbose:
        print(np.shape(imgArr))
        print(imgArr)
        plt.imshow(finImg)
        plt.show()
    return finImg


# function for creating model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


# loading MNIST database
training_data, training_labels, test_data, test_labels = loadData()

savePath = 'training_2/'
# ------------------------------------------------------------------------------------
# # creating fresh model
# model = create_model()
#
# # displaying model info
# model.summary()
# print(type(model))
#
# # training the model
# model.fit(training_data, training_labels, epochs=10, validation_data=(test_data, test_labels))
#
# # saving the model
# model.save(savePath + 'training1.h5')
# ------------------------------------------------------------------------------------

# loading saved model
loaded_model = tf.keras.models.load_model(savePath + 'training1.h5')

# evaluating loaded model
loss, acc = loaded_model.evaluate(test_data, test_labels, verbose=2)

# loading user image
img = readImg(args["image"], verbose=True, filter=False)

# function for passing user inputted image to the model
def getOutput(verbose=True):
    outputLayer = loaded_model(img.reshape(-1, 28, 28))
    probOutputLayer = tf.nn.softmax(outputLayer[0]).numpy()
    output = np.argmax(probOutputLayer)
    confidence = round(probOutputLayer[output]*100,2)
    # print(outputLayer[0])
    if verbose:
        print(probOutputLayer)
        print(np.shape(outputLayer))
        print(f'{output} : {confidence:.2f}%')
    return (output, confidence)

# loop try
outlist = []
for i in np.arange(0,1, step=0.01):
    img = readImg(args["image"], False, False)
    imgArr = img.reshape(784)
    zeroIndxs = imgArr < i
    imgArr[zeroIndxs] = 0
    (output, confidence) = getOutput(False)
    outlist.append((output, confidence))
print(outlist)
print(max(outlist, key=lambda item:item[1]))
