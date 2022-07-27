##Install packages needed for running API
import os
import flask
from flask import Flask
import json
#from flask import request, render_template


##Install packages needed for running the recommedndation function
from keras.preprocessing.image import load_img,img_to_array
import shutil
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


imgs_path = "C:/Users/Anthony/Downloads/Dataset/images/" #link to image path
imgs_model_width, imgs_model_height = 224, 224  #resize the image when loaded
nb_closest_images = 5   #num of img for closest similarity

files = [imgs_path + x for x in os.listdir(imgs_path) if "jpg" in x]   #count of the img in path


##the img similarity function
def similar_products(click_img):
    # compute cosine similarities between images
    imgs_features=np.load('C:/Users/Anthony/images_feature.npy') #loading feature extraction array
    cosSimilarities = cosine_similarity(imgs_features)
    
    # store the results into a pandas dataframe
    
    cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)
   
    #original = load_img(click_img, target_size=(imgs_model_width, imgs_model_height)) #to show img
   
    closest_imgs = cos_similarities_df[click_img].sort_values(ascending=False)[1:nb_closest_images+5].index
    closest_imgs_scores = cos_similarities_df[click_img].sort_values(ascending=False)[1:nb_closest_images+5]
   
    #destination_folder = 'C:/Users/Anthony/Downloads/Dataset/me/' #destination to store result recommended
    names=[] 
    for i in range(0,len(closest_imgs)):
        #original = load_img(closest_imgs[i], target_size=(imgs_model_width, imgs_model_height)) #to show img
        
        if closest_imgs[i]:
            response = {'score':float(closest_imgs_scores[i]), 'url':closest_imgs[i]}
            names.append(response)
            #shutil.copy(closest_imgs[i],destination_folder) ## use to copy recommeded files into a folder               
            #print("similarity score : ",closest_imgs_scores[i]) 
    return names

#similar_products('C:/Users/Anthony/Downloads/Dataset/images/A16PennsylvaniaDrawingRoom.jpg')  #how to call the function 

#from recom_img import retrieve_most_similar_products

app = Flask(__name__)


@app.route('/similarity/<click_img>')
def retrieve(click_img):
  i=int(click_img)
  similary_file = similar_products(files[i])
  return(json.dumps(similary_file))
  
  #return "The similar images are " + str(files[i])


#@app.route('/')
#def main():
    
#    return (json.dumps({"tony": 1, "sql": 2, "python": 3}))

if __name__ == '__main__':
    app.run()
