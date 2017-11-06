import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
try:
    from keras.preprocessing.image import img_to_array, array_to_img
except:
    print(
        '''
        Warning! Can not import keras methods!
        Do Not Use font2arry_keras and font2arry_keras_fit
        '''
    )

def Font2Image(hanzi, size, font):
    im = Image.new("L", (size,size), color=0)
    draw = ImageDraw.Draw( im )
    font=ImageFont.truetype( font, size)
    draw.text ( (0,0), hanzi, fill=255, font=font )
    return im

def Font2array(font, size, encoding):
    
    '''
    Input font (.ttf), size(int), encoding list(list or string)
    Yeild numpy array (no chennel ie. shape=(size, size))
    
    need:
         numpy
         PIL
    '''
    
    for hanzi in encoding:
        im    = Font2Image(hanzi, size, font)
        array = np.array(im)
        array = array.reshape(size, size)
        yield array
        
def font2array_chennel(font, size, encoding):
    
    '''
    Input font (.ttf), size(int), encoding list(list or string)
    Yeild numpy array (have chennel ie. shape=(size, size,1))
    
    need:
         numpy
         PIL
    '''
    
    for hanzi in chrlist:
        im    = Font2Image(hanzi, size, font)
        array = np.array(im)
        array = array.reshape(size, size, 1)
        yield array


def font2array_keras(font, size, encoding):
    
    '''
    Input font (.ttf), size(int), encoding list(list or string)
    Yeild numpy array (have chennel ie. shape=(size, size,1))
    
    need:
         numpy
         PIL
         keras
    '''
    
    for hanzi in chrlist:
        yield img_to_array(Font2Image(hanzi, size, font))

def font2array_keras_fit(font, size, encoding, batchsize, suffle= True):
    
    '''
    Input font (.ttf), size(int), encoding list(list or string), batchsize
    Yeild numpy array (have chennel ie. shape=(size, size,1)) for keras fit_generator
    it will shuffle the chrlist
    need:
         numpy
         PIL
         keras
    '''
    
    #Make infinite loop for fit_generator
    while True:
        
        suffle_chrlist = list(chrlist)
        
        if suffle:
            np.random.shuffle(suffle_chrlist) #shuffle for fitting
        else:
            pass
        
        #make a batch array
        batch_num = len(suffle_chrlist)//batchsize
        for index in range(batch_num):
            batch_chrlist = suffle_chrlist[index*batchsize: (index+1)*batchsize]
            batch_array = []
            for hanzi in batch_chrlist:
                batch_array.append(img_to_array(Font2Image(hanzi, size, font)))
            batch_array_normalized = np.array(batch_array)/255
            yield batch_array_normalized, batch_array_normalized
