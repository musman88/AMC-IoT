import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D,  ZeroPadding2D
from keras.layers import BatchNormalization, MaxPooling2D, MaxPooling3D
from keras.regularizers import *
# from keras.optimizers import adam
import matplotlib.pyplot as plt
# from tensorflow.python.keras.optimizers import *
import seaborn as sns
from tensorflow.python.keras.optimizers import adam
import _pickle as cPickle
import random, sys, keras
import csv
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
from tensorflow.compat.v2.keras.utils import multi_gpu_model
from keras.utils import multi_gpu_model

with open("RML2016.10a_dict.pkl", 'rb') as f:       #Dataset = RML 2016.10a
#with open("RML2016.04C.multisnr.pkl", 'rb') as f:  #Dataset = RML2016.04C                                                                          
    Xd = cPickle.load(f, encoding="latin1")
from RML_Model import*
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    # mod is the label. mod = modulation scheme
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        # snr = signal to noise ratio
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)

# np.random.seed(2020)
n_examples = X.shape[0]
# looks like taking half the samples for training
n_train = int(n_examples * 0.8)     #80% samples in the training dataset 20% in testing
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]
def to_onehot(yy):
    data = list(yy)

    yy1 = np.zeros([len(data), max(data) + 1])
    yy1[np.arange(len(data)), data] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods

#dr = 0.5  # dropout rate (%)
dr = 0.6  # dropout rate (%)

sig_model = build_model_ConvNet(batch_normalization=False,activation='relu',in_shp=in_shp,dr=dr,classes=classes)       #Conventional CNN
# sig_model = build_model_Depthwise(batch_normalization=False,activation='relu',in_shp=in_shp,dr=dr,classes=classes)   #Conventional CNN+Depthwise
# sig_model = build_model_Separable(batch_normalization=True,activation='relu',in_shp=in_shp,dr=dr,classes=classes)    #Conventional CNN+Depthwise Separable
opt = keras.optimizers.Adam(learning_rate=0.01)
sig_model.compile( loss='categorical_crossentropy',optimizer=opt,  metrics=['acc'])
sig_model.build()
sig_model.summary()
nb_epoch = 100    # number of epochs to train on
batch_size = 1024  # training batch size

# # perform training...

#filepath = 'convmodrecnets1_CNN2_0.5.wts.h5'    #For dropout = 0.5
filepath = 'convmodrecnets1_CNN2_0.6.wts.h5'     #For dropout = 0.6
# parallel_model = multi_gpu_model(model, gpus=2)
sig_model.compile( loss='categorical_crossentropy',optimizer=opt,  metrics=['acc'])
history = sig_model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=[
                        # params determine when to save weights to file. Happens periodically during fit.
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='auto')
                    ]
                        )
# # # we re-load the best weights once training is finished. best means lowest loss values for test/validation
sig_model.load_weights(filepath)

# # # Show simple version of performance
score = sig_model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print (score)

# #
# # # Show loss curves
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()
plt.savefig('Loss_C.png')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds, labels=[]):   #Reds for CNN, Blues for Depthwise, Greens for Separable
    #plt.cm.Reds - color shades to use, Reds, Blues, etc.
    # made the image bigger- 800x800
    my_dpi=96
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    #key call- data, how to interpolate thefp vakues, color map
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #adds a color legend to right hand side. Shows values for different shadings of blue.
    plt.colorbar()
    # create tickmarks with count = number of labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrix

# pass in X_test value and it predicts test_Y_hat
test_Y_hat = sig_model.predict(X_test, batch_size=batch_size)
# fill matrices with zeros
conf = np.zeros([len(classes), len(classes)])
# normalize confusion matrix
confnorm = np.zeros([len(classes), len(classes)])

# this puts all the data into an 11 x 11 matrix for plotting.
for i in range(0, X_test.shape[0]):
    # j is first value in list
    j = list(Y_test[i, :]).index(1)
    # np.argmax gives the index of the max value in the array, assuming flattened into single vector
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1

# takes the data to plot and normalizes it
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
print(confnorm)
print(classes)
plot_confusion_matrix(confnorm, labels=classes)
# #
# #
# # # Plot confusion matrix
acc = {}
# #
# # # this create a new confusion matrix for each SNR
for snr in snrs:#range(1):#snrs:

    # extract classes @ SNR
    # changed map to list as part of upgrade from python2
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    # estimate classes
    test_Y_i_hat = sig_model.predict(test_X_i)

    # create 11x11 matrix full of zeroes
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1

    # normalize 0 .. 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="CNN Confusion Matrix (SNR=%d)" % (snr))
    plt.savefig('SNR_%d.png'%(snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc[snr] = 1.0 * cor / (cor + ncor)
#
# Save results to a pickle file for plotting later
print (acc)
#fd = open('results_cnn2_d0.5.dat','wb')    #For dropout = 0.5
#cPickle.dump( ("CNN2", 0.5, acc) , fd )    #For dropout = 0.5

fd = open('results_cnn2_d0.6.dat','wb')     #For dropout = 0.6
cPickle.dump( ("CNN2", 0.6, acc) , fd )     #For dropout = 0.6
#
# # Plot accuracy curve
# # map function produces generator in python3 which does not work with plt. Need a list.
list(map(chr,[66,53,0,94]))
plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Convolutional Neural Network")
plt.show()
plt.savefig('Accuracy.png')

#with open('acc_dictC_dr50.csv', 'w') as csv_file:  #For dropout = 0.5
with open('acc_dictC_dr60.csv', 'w') as csv_file:   #For dropout = 0.6
    writer = csv.writer(csv_file)
    for key, value in acc.items():
        writer.writerow([key, value])
