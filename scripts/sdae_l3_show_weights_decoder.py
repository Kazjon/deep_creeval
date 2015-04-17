import theano, scipy, sys, os.path
import numpy as np
from PIL import Image
from pylearn2.utils import serial
from pylearn2.gui import patch_viewer

#model_path = '/Users/kazjon/Dropbox/Documents/Research/UNCC/ComputationalCreativity/CCWorkshop/pylearn2/pylearn2/scripts/tutorials/stacked_autoencoders/dae'

def layerpath(l,model_path):
	return model_path+"_l"+str(l)+".pkl"

def show_weights(model_path = "dae"):
	i = 1
	models = []
	weights = []
	Xs = []
	Ys = []
	encode_functs = []
	decode_functs = []
	while os.path.isfile(layerpath(i,model_path)):
		models.append(serial.load(layerpath(i,model_path)))
		I = models[i-1].get_input_space().make_theano_batch()	
		E = models[i-1].encode(I)
		encode_functs.append(theano.function( [I], E ))
		H = models[i-1].get_output_space().make_theano_batch()
		D = models[i-1].decode(H)
		decode_functs.append(theano.function( [H], D ))
		weights.append(models[i-1].get_weights())
		i += 1

	l1_acts = np.zeros([weights[2].shape[1],weights[0].shape[0]])
	for k in range(len(weights[2].T)):
		feature = np.zeros(len(weights[2].T))
		feature[k] = 1.0
		l3_acts = decode_functs[2](np.atleast_2d(feature.astype(np.dtype(np.float32))))
		l2_acts = decode_functs[1](l3_acts)
		l1_acts[k] = decode_functs[0](l2_acts)

	pv = patch_viewer.make_viewer(l1_acts, patch_shape=[28,28])
	pv.save("mnist_l3_weights_decoder.png")
	#scipy.misc.imsave('mnist7_l1_w0.png',l1_act.reshape([28,28]))

if __name__ == "__main__":
	show_weights()