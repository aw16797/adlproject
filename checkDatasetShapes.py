import torch
from torch.utils import data
import numpy as np
import pickle
from PIL import Image

dataset = pickle.load(open('UrbanSound8K_test.pkl', 'rb'))

mfcc = dataset[0]['features']['mfcc']
lm = dataset[0]['features']['logmelspec']
chroma = dataset[0]['features']['chroma']
spectral = dataset[0]['features']['spectral_contrast']
tonnetz = dataset[0]['features']['tonnetz']
print('MFCC')
print(mfcc.shape)
print('LMC')
print(lm.shape)
print('Chr')
print(chroma.shape)
print('Spe')
print(spectral.shape)
print('Ton')
print(tonnetz.shape)

a = np.concatenate((mfcc,chroma,spectral,tonnetz),axis=0)
print('?')
print(a.shape)

b = np.concatenate((lm, chroma, spectral, tonnetz), axis=0)
print('?')
print(b.shape)

# Creates PIL image
img = Image.fromarray(a, 'L')
img.show()
