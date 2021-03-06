import theano, scipy, sys, os.path
import numpy as np
from PIL import Image
from pylearn2.utils import serial

#model_path = '/Users/kazjon/Dropbox/Documents/Research/UNCC/ComputationalCreativity/CCWorkshop/pylearn2/pylearn2/scripts/tutorials/stacked_autoencoders/dae'

def layerpath(l,model_path):
	return model_path+"_l"+str(l)+".pkl"

def reconstruct(model_path = "dae"):
	i = 1
	models = []
	Xs = []
	Ys = []
	encode_functs = []
	decode_functs = []
	print layerpath(i,model_path)
	while os.path.isfile(layerpath(i,model_path)):
		models.append(serial.load(layerpath(i,model_path)))
		I = models[i-1].get_input_space().make_theano_batch()	
		E = models[i-1].encode(I)
		encode_functs.append(theano.function( [I], E ))

		H = models[i-1].get_output_space().make_theano_batch()
		D = models[i-1].decode(H)
		decode_functs.append(theano.function( [H], D ))
	
		i += 1

	img = Image.open('mnist7.png')
	flatimg = np.reshape(img,[1,28*28])
	print flatimg.shape
	l1_act = encode_functs[0](flatimg)
	print l1_act.shape
	l2_act = encode_functs[1](l1_act)
	print l2_act.shape
	l3_act = encode_functs[2](l2_act)
	print l3_act.shape

	l3_recon = decode_functs[2](l3_act)
	print l3_recon.shape
	l2_recon = decode_functs[1](l2_act)
	print l2_recon.shape
	l1_recon = decode_functs[0](l2_recon)
	print l1_recon.shape
	scipy.misc.imsave('mnist7_recon_l3.png',l1_recon.reshape([28,28]))

if __name__ == "__main__":
	reconstruct()