# In[ ]:
#import libraries
import cv2, os
import numpy as np
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

# In[ ]:

# In[ ]:

#preprocess images which obtained from sumulator
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# In[ ]:
#zoom helper function
def zoom(img):
  zoom = iaa.Affine(scale=(1, 1.3))
  img = zoom.augment_image(img)
  return img
#pan helper function
def pan(img):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  img = pan.augment_image(img)
  return img

#brightbness alters function
def img_random_brightness(img):
    brightness = iaa.Multiply((0.2, 1.2))
    img = brightness.augment_image(img)
    return img
#flip function
def img_random_flip(img, steering_angle):
    image = cv2.flip(img,1)
    steering_angle = -steering_angle
    return img, steering_angle
# In[ ]:




