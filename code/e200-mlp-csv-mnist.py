#! /c/Apps/Anaconda3/python
"""
[Title] MLP with CSV format inputs
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Program Code Name] e200-mlp-csv-mnist.py  
[Description]
  - Local System의 CSV format 데이터 이용
  - Hadoop의 Flat 파일(csv, json, parquet 등)에 적용 가능
  - Naming 표준화로 가독성 확보
[History]
  - 2019-05-01 : 최초 작성
  - 2019-05-05 : Naming 표준화 개선
[References]
  - https://www.tensorflow.org
"""

import tensorflow as tf
import numpy as np
import pandas as pd

#from mlxtend.data import loadlocal_mnist

#from numpy import genfromtxt

# def convert(imgf, labelf, outf, n):
#     f = open(imgf, "rb")
#     o = open(outf, "w")
#     l = open(labelf, "rb")
#     f.read(16)
#     l.read(8)
#     images = []
#     for i in range(n):
#         image = [ord(l.read(1))]
#         for j in range(28*28):
#             image.append(ord(f.read(1)))
#         images.append(image)
#     for image in images:
#         o.write(",".join(str(pix) for pix in image)+"\n")
#     f.close()
#     o.close()
#     l.close()
# convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
#         "mnist_train.csv", 60000)
# convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
#         "mnist_test.csv", 10000)

# X, y = loadlocal_mnist(
#     images_path='train-images-idx3-ubyte'
#   , labels_path='train-labels-idx1-ubyte') 
# np.savetxt(fname='mnist-train-features.csv'
#   , X=X, delimiter=',', fmt='%d') 
# np.savetxt(fname='mnist-train-labels.csv', X=y, delimiter=',', fmt='%d')
# X, y = loadlocal_mnist(
#     images_path='t10k-images-idx3-ubyte'
#   , labels_path='t10k-labels-idx1-ubyte') 
# np.savetxt(fname='mnist-test-features.csv', X=X, delimiter=',', fmt='%d') 
# np.savetxt(fname='mnist-test-labels.csv', X=y, delimiter=',', fmt='%d')

# arrMnistTrainFeatures = genfromtxt('mnist-train-features.csv', delimiter=',')
# arrMnistTrainLabel= genfromtxt('mnist-train-labels.csv', delimiter=',')
# arrMnistTestFeatures= genfromtxt('t10k-train-features.csv', delimiter=',')
# arrMnistTestLabel= genfromtxt('t10k-train-labels.csv', delimiter=',')

dfTrainFeatures = pd.read_csv('mnist-train-features.csv',header=None)
dfTrainLabel = pd.read_csv('mnist-train-labels.csv',header=None)
dfTestFeatures = pd.read_csv('mnist-test-features.csv',header=None) 
dfTestLabel = pd.read_csv('mnist-test-labels.csv',header=None)

ndArrayTrainFeatures = dfTrainFeatures.values
ndArrayTrainLabel = dfTrainLabel.values
print('[type(ndArrayTrainLabel)]', type(ndArrayTrainLabel), '[ndArrayTrainLabel.shape]',ndArrayTrainLabel.shape)
numRowsTrain = ndArrayTrainLabel.shape[0]
print('[numRowsTrain]', numRowsTrain)
numLabelClass=len(np.unique(ndArrayTrainLabel))
print('[numLabelClass]', numLabelClass)
numFeatures=dfTrainFeatures.shape[1]
print('[numFeatures]', ndArrayTrainFeatures.shape[1])

featuresBeginPosition = numLabelClass

onehotTrainLabel=np.eye(numLabelClass)[ndArrayTrainLabel].reshape(-1,numLabelClass)
print('[onehotTrainLabel[:1,]]', onehotTrainLabel[:1,])
ndArrayTrain=np.concatenate([onehotTrainLabel, ndArrayTrainFeatures], axis=1)
print('[ndArrayTrain.shape]', ndArrayTrain.shape)


ndArrayTestFeatures = dfTestFeatures.values
ndArrayTestLabel = dfTestLabel.values
onehotTestLabel=np.eye(numLabelClass)[ndArrayTestLabel].reshape(-1,numLabelClass)
print('[onehotTestLabel[:1,]]', onehotTestLabel[:1,])
ndArrayTest=np.concatenate([onehotTestLabel, ndArrayTestFeatures], axis=1)
print('[ndArrayTest.shape]', ndArrayTest.shape)

numEpoches = 20
numRowsPerBatch = 100
numBatchesPerEpoch = int(numRowsTrain / numRowsPerBatch)
print('[numBatchesPerEpoch]', numBatchesPerEpoch)
learningRate=0.001
numNodesEachHiddenLayer={'numNodesH1':256, 'numNodesH2':256}

holderFeatures = tf.placeholder(shape=[None, numFeatures], dtype=tf.float64, name='holderFeatures')
holderLabel = tf.placeholder(shape=[None, numLabelClass], dtype=tf.float64, name='holderLabel')

inputToHidden1Matrices = tf.nn.tanh(tf.random_normal([numFeatures, numNodesEachHiddenLayer['numNodesH1']], dtype=tf.float64))
hidden1ToHidden2Matrices = tf.nn.tanh(tf.truncated_normal([numNodesEachHiddenLayer['numNodesH1'],numNodesEachHiddenLayer['numNodesH2']], dtype=tf.float64))
hidden2ToOutputMatrices = tf.nn.tanh(tf.truncated_normal([numNodesEachHiddenLayer['numNodesH2'],numLabelClass], dtype=tf.float64))

biasHidden1 = tf.zeros([numNodesEachHiddenLayer['numNodesH1']], dtype=tf.float64)
biasHidden2 = tf.zeros([numNodesEachHiddenLayer['numNodesH2']], dtype=tf.float64)
biasOutput = tf.zeros([numLabelClass], dtype=tf.float64)

Weights={
  'weightLayerInputToHidden1' : tf.Variable(inputToHidden1Matrices, dtype=tf.float64)
  ,'hidden1ToHidden2Matrices': tf.Variable(hidden1ToHidden2Matrices, dtype=tf.float64)
  ,'hidden2ToOutputMatrices': tf.Variable(hidden2ToOutputMatrices, dtype=tf.float64)
}

biases={
  'biasHidden1' : tf.Variable(biasHidden1, dtype=tf.float64)
, 'biasHidden2' : tf.Variable(biasHidden2, dtype=tf.float64)
, 'biasOutput' : tf.Variable(biasOutput, dtype=tf.float64)
}


def mlpModel(featureInputs):
  equationHidden1 = tf.add(tf.matmul(featureInputs, Weights['weightLayerInputToHidden1']), biases['biasHidden1'])
  equationHidden2 = tf.add(tf.matmul(equationHidden1, Weights['hidden1ToHidden2Matrices']), biases['biasHidden2'])
  equationOutput = tf.add(tf.matmul(equationHidden2, Weights['hidden2ToOutputMatrices']), biases['biasOutput'])
  return equationOutput

Hypothesis = mlpModel(featureInputs = holderFeatures)

lossCrossEntropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Hypothesis, labels=holderLabel))
optimizationGradientDescent = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(lossCrossEntropy)
init=tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for stepOfEpoch in range(5): #range(numBatchesPerEpoch):
    lossTotalPerEpoch=0
    for stepOfBatch in range(numBatchesPerEpoch):
      batchRowsBegin = stepOfBatch * numRowsPerBatch 
      batchRowsEnd = min(stepOfBatch * numRowsPerBatch + numRowsPerBatch, numRowsTrain)
      batchTrainFeatures=ndArrayTrain[batchRowsBegin:batchRowsEnd, featuresBeginPosition:]
      batchTrainLabel =  ndArrayTrain[batchRowsBegin:batchRowsEnd, :featuresBeginPosition]
      # print('[batchTrainLabel]\n', batchTrainLabel)
      _, lossPerBatch =sess.run(
          [optimizationGradientDescent, lossCrossEntropy]
        , feed_dict = {
                holderFeatures : batchTrainFeatures
            ,   holderLabel : batchTrainLabel
          }
      )

      lossTotalPerEpoch += lossPerBatch 
      lossMeanPerEpoch = lossTotalPerEpoch / numBatchesPerEpoch
    print('[stepOfEpoch]',stepOfEpoch, '[lossTotalPerEpoch]', lossTotalPerEpoch, '[lossMeanPerEpoch]', lossMeanPerEpoch)

    testFeatures = ndArrayTest[:, featuresBeginPosition:]
    testLabel = ndArrayTest[:, :featuresBeginPosition]
    lossMeanTest = sess.run(lossCrossEntropy, feed_dict={holderFeatures : testFeatures, holderLabel : testLabel})

    predictedLabel = tf.nn.softmax(Hypothesis)
    correctPrediction = tf.equal(tf.argmax(predictedLabel, 1),tf.argmax(holderLabel, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
    accuracyTest = sess.run(accuracy, feed_dict = {holderFeatures:testFeatures, holderLabel:testLabel})
    print('[lossMeanTest]',lossMeanTest,'[accuracyTest]',accuracyTest)

"""
    onehotPredictedLabel = tf.nn.softmax(Hypothesis)
    predictedlabel = tf.argmax(onehotPredictedLabel, 1)
    for i in range(len(testLabel)):
      print(i
           ,  sess.run(tf.argmax(testLabel[i:i+1],1))
           ,  sess.run(tf.argmax(predictedLabel.eval(feed_dict={holderFeatures:testFeatures[i:i+1]}),1))
           )
"""
