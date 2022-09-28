# Mnist Classifier with VGG19
## Problem Description
The mnist dataset contains handwritten digits, where it has a training set of 60,000 examples, and a test set of 10,000 examples.

Please download it here: http://yann.lecun.com/exdb/mnist/

Requirements:
- You are asked to construct a classification model based on convolutional neural networks for digit recognition. Please report the prediction accuracy for the 
testing set.
- Please add 10%, 20%, 30% of salt-and-pepper noise to the test images, and test them using the model you trained on clean images from the previous Question. 
- Please use VGG-16 as a backbone pre-trained on ImageNet to fine-tune your model for the classification and redo Q.1 and Q.2. 

To add noise to the images, you could use the example code below:
```
import random
import numpy as np
noise_lv = 0.1 % 0.1, 0.2, 0.3 
img_size = 28*28
for i in range(len(X_train)):
 ran_seq = random.sample([n for n in range(img_size)], np.int(img_size*noise_lv))
 x = X_train[i].reshape(-1, img_size)
 x[0, ran_seq]=255
 ```