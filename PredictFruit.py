# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:09:56 2019

@author: Sahan Dilshan
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import cv2
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator


validation_data_dir = 'fruits/validation'
img_width, img_height, img_depth = 32,32,3
batch_size = 64 

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}



model = load_model('Trained Models/fruits_fresh_cnn_1.h5')
print("model was successfully loaded.")

def draw_test(name, pred, im, true_label):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 500 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, "predited - "+ pred, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.putText(expanded_image, "true - "+ true_label, (20, 120) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.imshow(name, expanded_image)


def getRandomImage(path, img_width, img_height):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path + "/" + image_name
    return image.load_img(final_path, target_size = (img_width, img_height)), final_path, path_class



files = []
predictions = []
true_labels = []
# predicting images
#for i in range(0, 10):
#    path = './fruits/validation/' 
#    img, final_path, true_label = getRandomImage(path, img_width, img_height)
#    files.append(final_path)
#    true_labels.append(true_label)
#    x = image.img_to_array(img)
#    x = x * 1./255
#    x = np.expand_dims(x, axis=0)
#    images = np.vstack([x])
#    classes = model.predict_classes(images)
#    predictions.append(classes)
#    
#for i in range(0, len(files)):
#    image = cv2.imread((files[i]))
#    draw_test("Prediction", class_labels[predictions[i][0]], image, true_labels[i])
#    cv2.waitKey(0)

model.summary()
#r_324_100
preImg = image.load_img('apple.png', target_size=(32, 32))
preImg = image.img_to_array(preImg)


#preImg = preImg.reshape((1, preImg.shape[0], preImg.shape[1], preImg.shape[2]))
#x = model.predict_classes(preImg)
#
#print(class_labels[x[0]])

preImg = preImg.astype("float") / 255.0
preImg = np.expand_dims(preImg, axis=0)

y = model.predict(preImg)
#x = x * 1./255
#x = np.expand_dims(x, axis=0)
#images = np.vstack([x])
for i in range(0,81):
    print(class_labels[i]," :",y[0][i]*1000)
    

pre = [i*100 for i in y[0] if i*100>10]

for d in pre:
    print(d)
cv2.destroyAllWindows()



