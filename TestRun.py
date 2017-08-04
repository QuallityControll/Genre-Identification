import numpy as np
import os
import pickle
from mygrad import Tensor
from mygrad.math import log
from mygrad.nnet.layers import dense
from mygrad.nnet.losses import multiclass_hinge
from mygrad.nnet.activations import softmax, relu
from MFCC.py import *

original_path = r"C:\Users\manusree\PycharmProjects\Alexa Skills\Genre-Identification\SongTraining"
changed_path = r"C:\Users\manusree\PycharmProjects\Alexa Skills\Genre-Identification\SongTraining"
a = os.listdir(r"C:\Users\manusree\PycharmProjects\Alexa Skills\Genre-Identification\SongTraining")

song_paths = []
ytrain = []
MFCCs = []
for genre in a:
    changed_path1 = original_path + "\\" + genre
    changed_path2 = original_path + "\\" + genre
    for song in os.listdir(changed_path1):
        changed_path2 = changed_path1 + "\\" + song
        song_paths.append(changed_path2)
        if genre == "Classical":
            ytrain.append([1,0,0,0,0])
        elif genre == "Jazz":
            ytrain.append([0,1,0,0,0])
        elif genre == "Pop":
            ytrain.append([0,0,1,0,0])
        elif genre == "Rap":
            ytrain.append([0,0,0,1,0])
        elif genre == "Rock":
            ytrain.append([0,0,0,0,1])

for i in range(len(song_paths)):
    print("It is " + str(100*i/len(song_paths)) + "% of the way done")
    a = file_to_array(song_paths[i])
    b = to_MFCC(a)
    MFCCs.append(b)
    del b
    del a

D = 12
K = 5

W1 = Tensor(he_normal((D, 100)))
b1 = Tensor(np.zeros((100,), dtype=W1.dtype))
W2 = Tensor(he_normal((100, K)))
b2 = Tensor(np.zeros((K,), dtype=W2.dtype))
params = [b1, W1, b2, W2]

rate = 1.

l = []
acc = []
for i in range(1000):
    o1 = relu(dense(xtrain, W1) + b1)
    p_pred = softmax(dense(o1, W2) + b2)

    loss = cross_entropy(p_pred=p_pred, p_true=ytrain)
    
    l.append(loss.data.item())
    loss.backward()

    acc.append(compute_accuracy(p_pred.data, ytrain))
    
    for param in params:
        sgd(param, rate)
    
    loss.null_gradients()