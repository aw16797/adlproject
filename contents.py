import torch
from torch.utils import data
import numpy as np
import pickle
from PIL import Image

def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    arr_norm = (arr - arr_min) / (arr_max - arr_min)
    arr_scale = 255.0 * arr_norm
    return arr_scale

dataset = pickle.load(open('UrbanSound8K_test.pkl', 'rb'))
index = 352
mfcc = dataset[index]['features']['mfcc']
lm = dataset[index]['features']['logmelspec']
chroma = dataset[index]['features']['chroma']
spectral = dataset[index]['features']['spectral_contrast']
tonnetz = dataset[index]['features']['tonnetz']
print('MFCC')
print(mfcc.shape)
print('LM')
print(lm.shape)
print('Chr')
print(chroma.shape)
print('Spe')
print(spectral.shape)
print('Ton')
print(tonnetz.shape)

a = np.concatenate((mfcc,chroma,spectral,tonnetz),axis=0)
print(a.shape)
# Create image

a2 = rescale(a)
print(a2.astype(int))

img = Image.fromarray(np.uint8(a2.astype(int)), 'L')
img.save('mc.png')
img.show()
