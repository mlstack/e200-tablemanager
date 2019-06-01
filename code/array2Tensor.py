#! /c/Apps/Anaconda3/python

import numpy as np
import tensorflow as tf

_2d_arr = [[1,2,3],[4,5,6]]
_2d_np_arr = np.asarray(_2d_arr,np.float32)
print(_2d_np_arr)
tf_arr = tf.convert_to_tensor(_2d_np_arr, np.float32)
sess = tf.InteractiveSession()
print(tf_arr.eval())

sess.close()
