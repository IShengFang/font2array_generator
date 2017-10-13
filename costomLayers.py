import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.layers.merge import _Merge

#################
# Costom Layers #
#################
BATCH_SIZE = 64
LATENT_DIM = 300
epsilon_std = 1.0

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):

        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
def sampling(args):
    z_mean, z_log_var = args
    #epsilon = K.random_normal(shape=(K.shape(en_mean)[0], LATENT_DIM), mean=0., stddev=epsilon_std)
    epsilon = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM), mean=0., stddev=epsilon_std)
    print('epsilon')
    print(epsilon)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class GLCM(Layer):

    def __init__(self, BATCH_SIZE = 100, levels = 2, symmetric = True, normed = True, **kwargs ):
        super(GLCM, self).__init__(**kwargs)
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        self.batch_size = BATCH_SIZE

    def compute_output_shape(self, input_shape):

        return (self.batch_size, 4)

    def binarization(self, x):
        bi_x = K.sign(x)
        return bi_x
    
    def glcm(self, inputs, distance, angle):
        image = self.binarization(inputs)
        batch = image.shape[0].value
        
        #calculate GLCM(in a batch shape)
        rows = image.shape[1].value
        cols = image.shape[2].value
                
        row = int(round(np.sin(angle))) * distance
        col = int(round(np.cos(angle))) * distance
        if col > 0:
            a = image[:, :rows-row, :cols-col, :]
            b = image[:, row:, col:, :]
        else:
            a = image[:, :rows-row, -col:, :]
            b = image[:, row:, :cols+col, :]

        a_or_b = K.maximum(a, b)
        a_and_b = a*b

        one = K.ones(shape = a.shape)

        not_a = one-a
        not_b = one-b

        # not(a or b)
        M0_0 = K.map_fn(K.sum, (one-a_or_b))

        # b and (not a)
        M0_1 = K.map_fn(K.sum, (b*(not_a)))

        # a and (not b)
        M1_0 = K.map_fn(K.sum, (a*(not_b)))

        # a and b
        M1_1 = K.map_fn(K.sum, (a_and_b))
        
        if self.symmetric:
            temp = K.stack([M0_0*2, M0_1+M1_0, M0_1+M1_0, M1_1*2], axis = 1)
            if self.normed:
                output = temp / K.reshape(K.sum(temp, axis=1), (batch, 1))
            else:
                output = temp
        else:
            temp = K.stack([M0_0, M0_1, M1_0, M1_1], axis = 1)
            if self.normed:
                output = temp / K.reshape(K.sum(temp, axis=1), (batch, 1))
            else:
                output = temp

        return output
    
    def call(self, inputs):
        #make input becomes a binarized image
        # distances = [1, 2, 3]
        # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        g1_0 = self.glcm(inputs, 1, 0)
        g1_1 = self.glcm(inputs, 1, np.pi/4)
        g1_2 = self.glcm(inputs, 1, np.pi/2)
        g1_3 = self.glcm(inputs, 1, 3*np.pi/4)
        g2_0 = self.glcm(inputs, 2, 0)
        g2_1 = self.glcm(inputs, 2, np.pi/4)
        g2_2 = self.glcm(inputs, 2, np.pi/2)
        g2_3 = self.glcm(inputs, 2, 3*np.pi/4)
        g3_0 = self.glcm(inputs, 3, 0)
        g3_1 = self.glcm(inputs, 3, np.pi/4)
        g3_2 = self.glcm(inputs, 3, np.pi/2)
        g3_3 = self.glcm(inputs, 3, 3*np.pi/4)
        
        return K.concatenate([g1_0, g1_1, g1_2, g1_3, g2_0, g2_1, g2_2, g2_3, g3_0, g3_1, g3_2, g3_3], axis=1)
