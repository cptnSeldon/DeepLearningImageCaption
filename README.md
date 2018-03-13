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

- use of a pre-trained model
  - Keras already provides such a model
  - Oxford Visual Geometry, VGG (which is used here), ..

- pre-compute using the pre-trained model
  - the features are saved in a file
  - gain of time

```python
# 01_data_preparation.py

  extract_features(dir_name)  
  """
  param: dir_name
  return: features

  This function does the following:
    - loads the model, restructures & summarizes it
    - in the loop: loads an image from a given picture file, converts its pixels to an array (numpy), reshapes & prepares the image for the VGG model, finally retrieves the features and stores it using the image id (so that we know which photo's features they are)
    - returns a dictionary containing the features retrieved from all the images
  """
```

- internal representation is done before the classification is made !
  - extraction by the model

3. Deep learning model development
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
