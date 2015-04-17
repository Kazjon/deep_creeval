__author__ = 'kazjon'


#Import the MNIST database again
from pylearn2.config import yaml_parse
from pylearn2.datasets.mnist import MNIST
import numpy as np
import model_inspector
import DUconfig,sys
import scipy.misc

img_size = [28,28]

sparse_coef = 0
sparse_p = 0.2

DUconfig.costs = yaml_parse.load('cost : !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {}')['cost']
data = MNIST('train',start=50000,stop=51000)
model = model_inspector.load_model(["cae_l1.pkl", "cae_l2.pkl"],DUconfig.costs)


model_inspector.save_as_patches(model_inspector.show_weights(model),[28,28],out_path="outpatches.png")
scipy.misc.imsave("hidden_weights.png",model_inspector.encode(data.X,model))
print model_inspector.recon_errors(data.X,model)

errors = np.zeros((data.X.shape[0],len(model['layers'])))

'''
originals = np.zeros([img_size[0]*data.X.shape[0],img_size[1]])
recons = np.zeros([img_size[0]*data.X.shape[0],img_size[1]])

for i,x in enumerate(data.X):
	originals[i*img_size[0]:(i+1)*img_size[0],:] = x.reshape(img_size)
	errors[i] = model_inspector.recon_errors(x,model)
	recons[i*img_size[0]:(i+1)*img_size[0],:] = model_inspector.reconstruct(model,x).reshape(img_size)
scipy.misc.imsave("comparison.png",np.concatenate((np.atleast_2d(originals*2),np.atleast_2d(recons)),axis=1))
#'''

#'''
originals = []
recons = []

for i,x in enumerate(data.X):
	originals.append(x.reshape(img_size))
	errors[i] = model_inspector.recon_errors(x,model)
	recons.append(model_inspector.reconstruct(model,x).reshape(img_size))
#'''
print errors

error_sums = [sum(x) for x in errors]
ary = zip(error_sums,originals,recons)
ary.sort(key=lambda row: row[0])
error_sums,originals,recons = zip(*ary)

originals = np.concatenate(originals)
recons = np.concatenate(recons)

num_to_print = 50

if len(error_sums) < 2 * num_to_print:
	scipy.misc.imsave("comparison.png",np.concatenate((np.atleast_2d(originals*2),np.atleast_2d(recons)),axis=1))
else:
	num_to_print *= img_size[0]
	scipy.misc.imsave("comparison_top.png",np.concatenate((np.atleast_2d(originals[0:num_to_print]*2),np.atleast_2d(recons[0:num_to_print])),axis=1))
	scipy.misc.imsave("comparison_bottom.png",np.concatenate((np.atleast_2d(originals[-num_to_print:]*2),np.atleast_2d(recons[-num_to_print:])),axis=1))

#Order samples by recon error
#Save 100 lowest and 100 highest as images