#!/usr/local/bin/python3

import numpy as np
from scipy.spatial import distance
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input19
from keras.models import Model
from keras.optimizers import RMSprop
#import os, shutil, glob, os.path
from PIL import Image as pil_image

img_dir = '/Users/bchambless/eBay_ML_competition/image_100'

# data from: image_100_vgg19_kmeans_no-norm_clust40_block5_pool_run1
# index of target cluster (black shoes): 28
# images: img149, img161, img260, img347, img358, img369, img380, img391

# index of source cluster(purple/red shoes): 37 
# red shoes: img472, img473 
# purple shoes: img515, img516, img520, img521 

# load custer centers
centers = np.loadtxt("saved_clusters_pool5_run1.csv")
targ_clust = centers[28] # black shoes
src_clust = centers[37] # purple shoes
rand_clust = centers[36] # whatever

# tested reshaping (ref so shaping changes both)
#src_clust_shaped = src_clust
#src_clust_shaped.shape = (1,7,7,512)

#print(src_clust_shaped.shape)

# calculate image vectors
# target imgs
black_shoe_imgs = np.array(["img149.jpg","img161.jpg","img260.jpg","img347.jpg","img358.jpg","img369.jpg","img380.jpg","img391.jpg"])
target_imgs = black_shoe_imgs

# src imgs
red_shoe_imgs = np.array(["img472.jpg","img473.jpg"])
purple_shoe_imgs = np.array(["img515.jpg","img516.jpg","img520.jpg","img521.jpg"])
src_imgs = np.concatenate([red_shoe_imgs, purple_shoe_imgs], axis=None)
# using one of the purple shoes as our training vector
training_img = np.array(["img515.jpg"])

# load model
image.LOAD_TRUNCATED_IMAGES = True
base_model = VGG19(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# distances


dist = distance.euclidean(targ_clust, src_clust)
print("distance from src to targ cluster centers: %s" %(dist))

# calculate some metrics for sanity
for f in black_shoe_imgs: #purple_shoe_imgs:
    imagepath = img_dir + "/" + f
    #print(imagepath)
    img = image.load_img(imagepath, target_size=(224,224), interpolation='bicubic')
    #img = image.load_img(imagepath, target_size=(375,500), interpolation='nearest')
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input19(img_data)
    features = np.array(model.predict(img_data))
    print(features.shape)
    flattened = features.flatten()
    print(flattened.shape)
    
    dist = distance.euclidean(src_clust, flattened)
    #print("(%s) distance to src cluster: %s" %(f) %(dist))
    print(str(f) + ': distance to src cluster: ' + str(dist)) 
    
    dist = distance.euclidean(targ_clust, flattened)
    #print("(%s) distance to targ cluster: %s" %(f) %(dist))
    print(str(f) + ': distance to target cluster: ' + str(dist)) 

# generate training vector
orig_tar_clust = np.array(targ_clust) # save the flat shape
targ_clust.shape = (1,7,7,512)
train_img_path = img_dir + "/" + "img515.jpg"
print("train_img_path= %s" %(train_img_path))
train_img = image.load_img(train_img_path, target_size=(224,224), interpolation='bicubic')
train_img_data = image.img_to_array(train_img)
train_img_data = np.expand_dims(train_img_data, axis=0)
train_img_data = preprocess_input19(train_img_data)  # image data is ready
print(train_img_data.shape)

#generate test vector
test_img_path = img_dir + "/" + "img79.jpg"
print("test_img_path= %s" %(test_img_path))
test_img = image.load_img(test_img_path, target_size=(224,224), interpolation='bicubic')
test_img_data = image.img_to_array(test_img)
test_img_data = np.expand_dims(test_img_data, axis=0)
test_img_data = preprocess_input19(test_img_data)  # image data is ready
print(test_img_data.shape)

#train_vector = np.array([train_img_data, targ_clust])
# check some distances
train_features = np.array(model.predict(train_img_data))
print(train_features.shape)
train_flattened = train_features.flatten()
print(train_flattened.shape)

dist = distance.euclidean(orig_tar_clust, train_flattened)
print("original purp-targ(train) distance: %s" %(dist))

# pre-train test distance
test_features = np.array(model.predict(test_img_data))
print(test_features.shape)
test_flattened = test_features.flatten()
print(test_flattened.shape)

#dist = distance.euclidean(orig_tar_clust, test_flattened)
dist = distance.euclidean(np.array(rand_clust), test_flattened)
print("test distance: %s" %(dist))

# prep the model
model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=1e-4), metrics=['mse']) #metrics=['accuracy'])
# train
history = model.fit(train_img_data, targ_clust, epochs=200, verbose=1)
print(history)

# check some distances
new_train_features = np.array(model.predict(train_img_data))
new_train_flattened = new_train_features.flatten()

dist = distance.euclidean(orig_tar_clust, new_train_flattened)
print("new purp-targ distance: %s" %(dist))

# post-train test distance
test_features = np.array(model.predict(test_img_data))
print(test_features.shape)
test_flattened = test_features.flatten()
print(test_flattened.shape)

#dist = distance.euclidean(orig_tar_clust, test_flattened)
dist = distance.euclidean(np.array(rand_clust), test_flattened)
print("new test distance: %s" %(dist))



