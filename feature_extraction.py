#packages to import 
from keras.applications import InceptionV3, ResNet50
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

imgs_path = "M:/VieApp/images/images/"
imgs_model_width, imgs_model_height = 224, 224
nb_closest_images = 5 # number of most similar images to retrieve

#model need to extract feature
model = ResNet50(weights='imagenet')

# remove the last layers in order to get features instead of predictions
feat_extractor = Model(inputs=model.input, outputs=vgg_model.get_layer("fc1000").output)

#feat_extractor.summary() # print the layers of the CNN

files = [imgs_path + x for x in os.listdir(imgs_path) if "jpg" in x] #files in the path
#print("number of images:",len(files))


# load all the images and prepare them for feeding into the CNN

importedImages = []

for f in files:
    filename = f
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    importedImages.append(image_batch)
images = np.vstack(importedImages)
processed_imgs = preprocess_input(images.copy())

# extract the images features
imgs_features = feat_extractor.predict(processed_imgs)
###save the img feature
np.save('feature.npy',imgs_features) #imgs_features.shape

# compute cosine similarities between images
imgs_features=np.load('images_feature.npy')
cosSimilarities = cosine_similarity(imgs_features)

# store the results into a pandas dataframe
cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)
#cos_similarities_df.head()
