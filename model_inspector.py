import theano, scipy, sys, os.path, time, ebird_load_script
import numpy as np
from PIL import Image
from pylearn2.utils import serial
from pylearn2.gui import patch_viewer
import DUconfig
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import DataSpecsMapping


#model_path = '/Users/kazjon/Dropbox/Documents/Research/UNCC/ComputationalCreativity/CCWorkshop/pylearn2/pylearn2/scripts/tutorials/stacked_autoencoders/dae'

def load_model(model_paths, costs, batch_size=100):
	if type(costs) is not list:
		costs = len(model_paths) * [costs]
	model = {}
	model['layers'] = []
	model['costs'] = []
	model['weights'] = []
	model['encoders'] = []
	model['decoders'] = []
	for i,path in enumerate(model_paths):
		if os.path.isfile(path):
			model['layers'].append(serial.load(path))
			I = model['layers'][i].get_input_space().make_theano_batch(batch_size=batch_size)
			E = model['layers'][i].encode(I)
			model['encoders'].append(theano.function( [I], E ))

			H = model['layers'][i].get_output_space().make_theano_batch(batch_size=batch_size)
			D = model['layers'][i].decode(H)
			model['decoders'].append(theano.function( [H], D ))
			model['weights'].append(model['layers'][i].get_weights())

			data_specs = costs[i].get_data_specs(model['layers'][i])
			mapping = DataSpecsMapping(data_specs)
			space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
			source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
			# Build a flat tuple of Theano Variables, one for each space.
			# We want that so that if the same space/source is specified
			# more than once in data_specs, only one Theano Variable
			# is generated for it, and the corresponding value is passed
			# only once to the compiled Theano function.
			theano_args = []
			for space, source in safe_zip(space_tuple, source_tuple):
				arg = space.make_theano_batch(batch_size=batch_size)
				theano_args.append(arg)
			theano_args = tuple(theano_args)

			# Methods of `self.cost` need args to be passed in a format compatible
			# with data_specs
			nested_args = mapping.nest(theano_args)
			fixed_var_descr = costs[i].get_fixed_var_descr(model['layers'][i], nested_args)

			model['costs'].append(theano.function([nested_args], costs[i].expr(model['layers'][i], nested_args, ** fixed_var_descr.fixed_vars)))
		else:
			sys.exit("Whoa. "+path+" isn't a thing I know about!")

	return model

def save_as_patches(activations, shape, out_path = "patches_out.png", rescale=True):
	pv = patch_viewer.make_viewer(activations, patch_shape=shape, rescale=rescale)
	pv.save(out_path)

def reconstruct(model, data):
	return decode(encode(np.atleast_2d(data),model),model)

def reconstruct_and_save(model, img_path, out_path = "out.png"):
	img = Image.open(img_path)
	flatimg = np.reshape(img,(1,img.size[0]*img.size[1]))
	recon = reconstruct(flatimg,model)
	scipy.misc.imsave(out_path,recon.reshape([img.size[0],img.size[1]]))

def show_weights(model, subtract_average=True):
	l1_acts = np.zeros([model['weights'][-1].shape[1],model['weights'][0].shape[0]])
	if subtract_average:
		feature = np.zeros(len(model['weights'][-1].T))
		if str(model['layers'][-1].act_enc) == "Elemwise{tanh,no_inplace}":
			feature -= 1
		acts = feature
		for layer in reversed(model['decoders']):
			acts = layer(np.atleast_2d(acts.astype(np.dtype(np.float32))))
		l1_acts = np.tile(acts*-1,(l1_acts.shape[0],1))
		save_as_patches(l1_acts,[28,28],out_path="feature_averages.png",rescale=True)
	for k in range(len(model['weights'][-1].T)):
		feature = np.zeros(len(model['weights'][-1].T))
		if str(model['layers'][-1].act_enc) == "Elemwise{tanh,no_inplace}":
			feature -= 1
		feature[k] = 1.0
		acts = feature
		for layer in reversed(model['decoders']):
			acts = layer(np.atleast_2d(acts.astype(np.dtype(np.float32))))
		l1_acts[k,:] += acts[0]
		scipy.misc.imsave("weights/feature"+str(k)+".png",acts.reshape([28,28]))

	return l1_acts

#This method calculates, for each layer, the error of reconstructing just that layer's input.
def recon_errors(data, model, batch_size = 1000):
	results = []
	for batch in np.array_split(data,max(1,data.shape[0]/batch_size)):
		result = []
		for i,cost in enumerate(model['costs']):
			d = np.atleast_2d(batch)
			for l in range(i):
				d = model['encoders'][l](d)
			result.append(cost(d))
		results.append(result)
	return np.vstack(results)

#This method calculates, for each layer, the error of reconstructing from the input up to that layer and back again
def deep_recon_errors(data, model, batch_size = 1000):
	results = []
	for batch in np.array_split(data,max(1,data.shape[0]/batch_size)):
		result = []
		cost = model['costs'][0]
		for layer in range(len(model['layers'])):
			d = np.atleast_2d(batch)
			for l in range(layer):
				d = model['encoders'][l](d)
			for l in reversed(range(layer)):
				d = model['decoders'][l](d)
			result.append(cost(d))
		results.append(result)
	return np.vstack(results)

def encode(data, model, batch_size = 1000):
	acts = []
	for batch in np.array_split(data,max(1,data.shape[0]/batch_size)):
		act = batch
		for layer in model["encoders"]:
			act = layer(act)
		acts.append(act)
	return np.vstack(acts)

def decode(data, model, batch_size = 1000):
	acts = []
	for batch in np.array_split(data,max(1,data.shape[0]/batch_size)):
		act = batch
		for layer in reversed(model["decoders"]):
			act = layer(act)
		acts.append(act)
	return np.vstack(acts)

def represent(dataset,model):
	return encode(dataset.X,model)

from pylearn2.config import yaml_parse

def mnist_test(model_paths = ["dae_l1.pkl","dae_l2.pkl","dae_l3.pkl"], sparse_coef=0, sparse_p = 0.1):
	cost = yaml_parse.load('cost : !obj:pylearn2.costs.cost.SumOfCosts {costs: [!obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {}, !obj:pylearn2.costs.autoencoder.SparseActivation {coeff: '+str(sparse_coef)+',p: '+str(sparse_p)+'}]}')['cost']
	model = load_model(model_paths,cost)
	DUconfig.dataset = MNIST(which_set='train',start=0,stop=50000)
	#reconstructed_dataset = represent(DUconfig.dataset,model)
	#print reconstructed_dataset[0]

	print recon_errors(DUconfig.dataset.X,model)
	#reconstruct(model,img_path="mnist7.png",out_path="out7.png")
	ipaths = ["mnist1.png","mnist4.png","mnist5.png","mnist7.png"]
	imgs = []
	for img in ipaths:
		i = Image.open(img)
		imgs.append(np.reshape(i,(1,i.size[0]*i.size[1])))
	flatimgs = np.concatenate(imgs)
	out = decode(encode(flatimgs,model),model)
	scipy.misc.imsave("out_prime.png",out.reshape([flatimgs.shape[0]*28,28]))
	save_as_patches(show_weights(model),[28,28],out_path="outpatches.png")

def ebird_test():
	DUconfig.costs = yaml_parse.load('cost : !obj:pylearn2.costs.cost.SumOfCosts {costs: [!obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {}, !obj:pylearn2.costs.autoencoder.SparseActivation {coeff: 1,p: 0.15}]}')['cost']
	model = load_model(["ebird_dae_l1.pkl","ebird_dae_l2.pkl","ebird_dae_l3.pkl"],DUconfig.costs)
	dataset_start = 0
	dataset_stop = -1

	species_to_retrieve = ['Turdus_migratorius'] #['Picoides_pubescens', 'Turdus_migratorius']  # Empty list means all

	start_time = time.time()
	#DUconfig.dataset = ebird_load_script.load_ebird_data(dataset_start,dataset_stop,species_to_retrieve=species_to_retrieve)
	DUconfig.dataset = ebird_load_script.monary_load(dataset_start,dataset_stop,species_to_retrieve=species_to_retrieve)
	dataset_stop = DUconfig.dataset.X.shape[0]

	print 'Query took',time.time() - start_time,'seconds. Training with',dataset_stop,'examples.'

	print recon_errors(DUconfig.dataset.X.astype(np.float32),model)


if __name__ == "__main__":
	mnist_test(model_paths = ["cae_l1.pkl","cae_l2.pkl","cae_l3.pkl"])
	#ebird_test()