import keras.backend as K
from keras import metrics

###################
# Define the loss #
###################
def wasserstein_real_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def wasserstein_fake_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)
