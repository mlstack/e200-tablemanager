# [Title] Autoencoder using Tensorflow.Keras
# [Author] Yibeck Lee(yibec.Lee@gmail.com)
# [Contents]
#  - Aucoencoder for Dimension Reduction
# [References]
#  - https://towardsdatascience.com/pca-vs-autoencoders-1ba08362f450
#  - https://medium.com/datadriveninvestor/deep-autoencoder-using-keras-b77cd3e8be95

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

trainFeatures = np.array([
  [1,1,1]
, [1,1,1]
, [1,1,1]
, [1,1,1]
, [1,2,3]
, [2,2,2]
, [2,2,2]
, [2,2,2]
, [2,2,2]
, [1,3,5]
])
print(trainFeatures)

encoding_dim = 2
input_img = Input(shape=(3,))
encoded = Dense(encoding_dim)(input_img)
decoded = Dense(3)(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(trainFeatures, trainFeatures,
                epochs=100,
                batch_size=2,
                verbose=0,
                shuffle=True)

decoded_imgs = autoencoder.predict(trainFeatures)
print(decoded_imgs)

error = trainFeatures - decoded_imgs

error_square = error**2
print(error_square)

error_square = np.sum(error_square, axis=1)
print(error_square)



import matplotlib.pyplot as plt
ind = np.arange(10) 
plt.bar(ind, error_square)
plt.show()

