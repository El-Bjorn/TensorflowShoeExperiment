#!/usr/local/bin/python3

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input as preprocess_input16
from keras.applications.vgg19 import preprocess_input as preprocess_input19
from keras.models import Model
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, MeanShift
import scipy.stats as stats
import os, shutil, glob, os.path
from PIL import Image as pil_image
#import Image

image.LOAD_TRUNCATED_IMAGES = True
#model = VGG16(weights='imagenet', include_top=False)
base_model = VGG19(weights='imagenet', include_top=True)
#base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
#jmodel = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
#model = Model(inputs=base_model.input, outputs=base_model.output)
#model = VGG16(weights='imagenet', include_top=False)
#
#exit()

img_dir = '/Users/bchambless/eBay_ML_competition/image_100'
target_dir = '/Users/bchambless/eBay_ML_competition/image_100_vgg19_kmeans_no-norm_clust40_block5__4run3'
num_clusters = 40

# create cluster directoriesr
clust_dirs = []
for i in range(num_clusters):
    dir = target_dir + '/' + str(i)
    print(dir)
    os.makedirs(dir)
    clust_dirs.append(dir)

filelist = glob.glob(os.path.join(img_dir, '*.jpg'))
filelist.sort()
#print(filelist)
featurelist = []

for i, imagepath in enumerate(filelist):
    print(" image: %s " %(imagepath))
    #img = image.load_img(imagepath, target_size=(224,224), interpolation='bicubic', color_mode='grayscale')
    img = image.load_img(imagepath, target_size=(224,224), interpolation='bicubic')
    #img = image.load_img(imagepath, target_size=(375,500), interpolation='nearest')
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input19(img_data)
    features = np.array(model.predict(img_data))
    flattened = features #.flatten()
    print(flattened.shape)
    #print(stats.describe(flattened))
    featurelist.append(flattened)

# norm the array
print("norming....")
f = np.array(featurelist)
#normf = (f - f.min(0)) / f.ptp(0)
normf = f / f.max(axis=0)
print(normf[0])
print(stats.describe(normf[0]))
print("------=--")
print(f[0])
print(stats.describe(f[0]))
#normf = f / ( f.max(axis=0), f)[ f.max(axis=0) != 0 ]


print("doing the clusterings...........")
#exit()

#kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(featurelist))
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(f)
#kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(normf)
#spectral = SpectralClustering(n_clusters=num_clusters).fit(normf)
#mean_shft = MeanShift().fit(normf)
#print("num bins= %s" %(mean_shft.labels_.size))
#
#try:
#    os.makedirs(target_dir)
#except OSError:
#    pass
for i, m in enumerate(kmeans.labels_):
#for i, m in enumerate(mean_shft.labels_):
    print(" %s -> %s" %(filelist[i],m))
    shutil.copy(filelist[i], clust_dirs[m])
	
print("cluster centers:")
print(kmeans.cluster_centers_)
print("saving clusters...")
np.savetxt("saved_clusters_black5_conv_run1_tst_no_flatten.csv", kmeans.cluster_centers_)
