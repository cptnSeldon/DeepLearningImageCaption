""" ----------------
    DATA PREPARATION : may take a while
    ---------------- """
# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
from os import listdir
from pickle import dump
import h5py

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def extract_features(directory): # extract features from each photo in the directory
    """
    param:  dir_name
    return: features

    This function does the following:
    - loads the model, restructures & summarizes it
    - in the loop:
      - loads an image from a given picture file
      - converts its pixels to an array (numpy)
      - reshapes & prepares the image for the VGG model
      - finally retrieves the features and stores it using the image id
      (so that we know which photo's features they are)
    - returns a dictionary containing the features retrieved from all the images
    """
    model = VGG16() # load the model
    model.layers.pop() # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary()) # summarize
    features = dict() # extract features from each photo
    for name in listdir(directory):
        filename = directory + '/' + name # load an image from file
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image) # convert the image pixels to a numpy array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))    # reshape data for the model
        image = preprocess_input(image) # prepare the image for the VGG model
        feature = model.predict(image, verbose=0) # get features
        image_id = name.split('.')[0] # get image id
        features[image_id] = feature # store feature
        print('>%s' % name)
    return features

if __name__ == "__main__":

    directory = '../Flicker8k_Dataset' # extract features from all images
    features = extract_features(directory)
    print('Extracted Features: %d' % len(features))
    dump(features, open('features.pkl', 'wb')) # save to file
l
