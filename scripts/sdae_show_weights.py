import theano, scipy, sys, os.path
import numpy as np
from PIL import Image
from pylearn2.utils import serial
from pylearn2.gui import patch_viewer

model_path = '/Users/kazjon/Dropbox/Documents/Research/UNCC/ComputationalCreativity/CCWorkshop/pylearn2/pylearn2/scripts/tutorials/stacked_autoencoders/dae'

def layerpath(l):
	return model_path+"_l"+str(i)+".pkl"

i = 1
models = []
weights = []
Xs = []
Ys = []
encode_functs = []
decode_functs = []
while os.path.isfile(layerpath(i)):
	models.append(serial.load(layerpath(i)))
	I = models[i-1].get_input_space().make_theano_batch()	
	E = models[i-1].encode(I)
	encode_functs.append(theano.function( [I], E ))
	H = models[i-1].get_output_space().make_theano_batch()
	D = models[i-1].decode(H)
	decode_functs.append(theano.function( [H], D ))
	weights.append(models[i-1].get_weights())
	i += 1

l1_acts = np.zeros(weights[0].T.shape)
for k in range(len(weights[1])):
	feature = weights[1][k]
	#feature = weights[1].T[k]
	l1_acts[k] = decode_functs[0](np.atleast_2d(feature))

pv = patch_viewer.make_viewer(l1_acts, patch_shape=[28,28])
pv.save("mnist_l2_weightsmat.png")
#scipy.misc.imsave('mnist7_l1_w0.png',l1_act.reshape([28,28]))

