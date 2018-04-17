# Automated Image Caption


This project's purpose is to study how (and how accurately) are automated images captions generated.

Jason Brownlee has written a complete tutorial on the matter where he proposes to create such a project from scratch using python and deep learning libraries (as keras for instance). The main idea is to follow this tutorial to grasp the bases through his example program, at first, then check other projects and compare their results.

## _Jason Brownlee_'s tutorial on developing a deep learning photo caption generator from _scratch_

This chapter is a resume of the actual post's content.

1. Datasets (photo + caption)

- use of the Flickr8K dataset, content:
  - 8000 images selected manually
  - great variety of scenes and situations
  - each image has 5 sentence-based description

2. Preparation of the data (photo + text)

__PHOTO__

- use of a pre-trained model
  - Keras already provides such a model
  - Oxford Visual Geometry, VGG (which is used here), ..

- pre-compute using the pre-trained model
  - the features are saved in a file
  - gain of time

```python
# 01_image_data_preparation.py

  def extract_features(dir_name):  
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
    pass # complete code in the file
```

- internal representation is done before the classification is made !
  - extraction by the model

- run the file on the terminal to prepare the image data contained in Flick'r8k directory using the following command (may take a while):
```
  python 01_text_data_preparation.py
```
  - it should look like this:

![While running the file](https://i.imgur.com/RqOYJdG.png

/!\ warning : h5py and Pillow have to be installed as well (foundable in the requirements)

__TEXT__

- each photo has several descriptions
- these descriptions have to be cleaned beforehand

```python
  # 02_text_data_preparation.py

  def load_doc(filename): # LOAD FILE CONTENT
    """
    param:  filename
    return: text

    This function does the following:
      - retrives all the text contained in the descriptions' file
      - returns the retrieved (loading into the memory)
    """
    pass # complete code in the file

  def load_description(doc):  # TOKENIZE id + descriptions
    """
    param:  doc (= text returned by the function above)
    return: mapping

    This function does the following:
      - creates a dictionary which will contain
        - the image identifier
        - the image description
      - retrieve the id and the description (parsing)
        - line composition : [image_id] space [description] new line
        - dictionary: mapping[image_id] = [description_1, description_2, ..]
      - returns the resulting dictionary
    """
      pass # complete code in the file

    def clean_descriptions(descriptions): # REMOVE UNNECESSARY INFORMATION
      """
      param:  descriptions
      return: none

      This function cleans the unnecessary information:
        - the sentence is cut into words (tokenization) then converted to lowercase
        - the punctuation is removed as well as the remaining a and s
          (we could also remove the other determinants here..)
        - the numbers are removed too
      """
      pass # complete code in the file

    def to_vocabulary(descriptions):
      """
      param:  descriptions
      return: all descriptions

      Creates a list containing all the descriptions in string.
      """
      pass # complete code in the file

    def save_descriptions(descriptions, filename):
      """
      param:  descriptions, filename
      return: none

      Saves the descriptions' list in the given file.
      """
      pass
```

Loaded: 8,092
Vocabulary Size: 8,763

3. Deep learning model development

Loading Data.
Defining the Model.
Fitting the Model.

4. Model evaluation
5. Caption generation







# __Links__ :

## Tutorials
- https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

## Misc documentation
- https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
- http://www.robots.ox.ac.uk/~vgg/research/very_deep/
- https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014
- http://caffe.berkeleyvision.org/model_zoo.html
- http://www.cs.toronto.edu/~frossard/post/vgg16/
- https://keras.io/applications/

## TensorFlow examples
- https://github.com/DeepRNN/image_captioning
- https://github.com/yunjey/show-attend-and-tell
