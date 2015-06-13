import theano, scipy, sys, os.path, time, ebird_load_script, sklearn.metrics, qutip, scipy.sparse, sklearn.cluster, matplotlib.cm, pprint, copy
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
	model['comparative_costs'] = []
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
			I2 = model['layers'][i].get_input_space().make_theano_batch(batch_size=batch_size)

			model['comparative_costs'].append(theano.function([I,I2], costs[i].costs[0].cost(I,I2)))
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

def show_weights(model, subtract_average=True, save_patches=True):
	l1_acts = np.zeros([model['weights'][-1].shape[1],model['weights'][0].shape[0]])
	if subtract_average:
		feature = np.zeros(len(model['weights'][-1].T))
		if str(model['layers'][-1].act_enc) == "Elemwise{tanh,no_inplace}":
			feature -= 1
		acts = feature
		for layer in reversed(model['decoders']):
			acts = layer(np.atleast_2d(acts.astype(np.dtype(np.float32))))
		l1_acts = np.tile(acts*-1,(l1_acts.shape[0],1))
		if save_patches:
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
		if save_patches:
			scipy.misc.imsave("weights/feature"+str(k)+".png",acts.reshape([28,28]))

	return l1_acts

#This method calculates, for each layer, the error of reconstructing just that layer's input.
def recon_errors_old(data, model, batch_size = 1000):
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

def recon_errors(data,model):
	results = np.zeros((data.shape[0],1))
	data_prime = decode(encode(data,model),model)
	for i,d in enumerate(data):
		result = []
		cost = model['comparative_costs'][0]
		d = np.atleast_2d(d)
		d_prime = np.atleast_2d(data_prime[i])
		results[i] = cost(d,d_prime)
	return results

#This method calculates, for each layer, the error of reconstructing from the input up to that layer and back again
def deep_recon_errors(data, model, batch_size = 1000):
	results = []
	for batch in np.array_split(data,max(1,data.shape[0]/batch_size)):
		result = []
		cost = model['costs'][0]
		for layer in range(len(model['layers'])):
			d = np.atleast_2d(batch)
			print cost(d)
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
	#print acts
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

def phones_test():
	import sklearn.preprocessing
	from conceptual_space import monary_load
	topmodel = monary_load("phones", ['Volume_(cubic_cm)','Display_Width(px)', 'Width_(mm)', 'Display_Diagonal_(in)', 'Display_Length(px)', 'Depth_(mm)', 'CPU_Clock_(MHz)', 'Mass_(grams)', 'ROM_Capacity_(Mb)', 'Length_(mm)', 'RAM_Capacity_(Mb)', 'Release_Year', 'Pixel_Density_(per_inch)'], [])
	F_by_O = topmodel.X.T
	print F_by_O.shape
	scipy.misc.imsave("F_by_O.png",F_by_O)

	print "Data extraction complete"
	#F_by_F = np.corrcoef(F_by_O)
	#O_by_O = np.corrcoef(F_by_O.T)
	#scipy.misc.imsave("O_by_O.png",O_by_O)

	#print "Corrcoefs complete"
	#SVD Calcs for O by O, comments taken from Som's community detection code.
	#k = 13
	for k in range(2,7):
		for t in [0.75]:
			U, s, V = scipy.sparse.linalg.svds(F_by_O.T,k=k) # Question remains whether this should be F by O or O by O!

			s = np.diagflat(s)
			print U.shape
			print s
			print V.shape
			# Perform dimensionality reduction, using parameter k. A good way to decide
			# on an optimal k value is to run the algorithm once, and plot the singular
			# values in decreasing order of magnitude. Then, choose k largest singular
			# values (the ones which show maximum gaps), and rerun algorithm at this k.
			#svdu = U[:,:k]
			#svds = s[:k,:k]
			#svdv = V[:,:k]

			# Generate US or SV products for generating reduced dimensional vectors
			# for nodes. This is the same if the matrix is symmetric and square.
			#u_red = svdu*svds
			u_red = np.matrix(U)*np.matrix(s)
			print "u_red: ", u_red.shape
			#v_red = svds*svdv.T
			#v_red = v_red.T
			v_red = np.matrix(s)*np.matrix(V)
			print "v_red: ", v_red.shape
			print "SVDs complete"

			#scipy.misc.imsave("u_red"+str(k)+"_"+str(t)+".png",u_red)
			#scipy.misc.imsave("v_red"+str(k)+"_"+str(t)+".png",v_red)
			# Compute cosine measurements between all US combinations. Produce the
			# cosine matrix in reduced k-space. Z_u will show communities for only Type
			# 1 nodes (rows of the original matrix).
			#Z_u = sklearn.metrics.pairwise.pairwise_distances(u_red,metric="cosine",n_jobs=-1)
			Z_u = np.corrcoef(u_red) / 2 + 0.5
			Z_u_labels = sklearn.cluster.MiniBatchKMeans(n_clusters=k).fit_predict(Z_u)
			clustersort = np.argsort(Z_u_labels)

			cm = matplotlib.cm.get_cmap("jet")
			cm.set_under(color="k", alpha=0)
			Z_u_clusters = np.zeros(Z_u.shape) - 0.01
			#Z_u_clusters = np.dstack([Z_u_clusters,Z_u_clusters,Z_u_clusters,np.ones(Z_u.shape)*255])
			#for i in range(k):
			#	lmask = np.matrix((Z_u_labels == i).astype(int))
			#	Z_u_clusters += (lmask.T * lmask) * (float(i+1)/(k+1))
			for i,l1 in enumerate(Z_u_labels):
				for j,l2 in enumerate(Z_u_labels):
					if l1 == l2:
						Z_u_clusters[i,j] = float(l1+1)/k
			#scipy.misc.imsave("Z_u_clusters_"+str(k)+".png",Z_u_clusters)

			#Z_u_overlaid = cm(Z_u_clusters, bytes=True)*np.dstack([Z_u,Z_u,Z_u,np.ones((Z_u.shape))])
			#TODO: WOrking on how to get this to add when the second image is 0s, and multiply when it's not

			#Z_u_labels.shape = (Z_u_labels.shape[0],1)
			#Z_u_overlaid = cm(Z_u_labels + Z_u)
			scipy.misc.imsave("Z_u_clusters"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters, bytes=True))
			#pprint.pprint(Z_u_overlaid)
			#scipy.misc.imsave("Z_u_clusters_"+str(k)+".png",cm(Z_u_clusters, bytes=True))
			#scipy.misc.imsave("Z_u_nocorr_"+str(k)+".png",Z_u)
			scipy.misc.imsave("Z_u_"+str(k)+".png",Z_u)
			Z_u_bin = sklearn.preprocessing.binarize(Z_u,t)
			#scipy.misc.imsave("Z_u_bin_nocorr_"+str(k)+"_"+str(t)+".png",Z_u_bin)
			scipy.misc.imsave("Z_u_bin_"+str(k)+"_"+str(t)+".png",Z_u_bin)


			from networkx.utils import reverse_cuthill_mckee_ordering
			rcm_order = qutip.reverse_cuthill_mckee(scipy.sparse.csr_matrix(1 - Z_u_bin)) # This *should* produce a good diagonalisation? Hasn't been tested.

			Z_u_rcm = Z_u[rcm_order,:]
			Z_u_rcm = Z_u_rcm[:,rcm_order]
			Z_u_clst = Z_u[clustersort,:]
			Z_u_clst = Z_u_clst[:,clustersort]
			#Z_u_bin_rcm = Z_u_bin[rcm_order,:]
			#Z_u_bin_rcm = Z_u_bin_rcm[:,rcm_order]
			Z_u_clusters_rcm = Z_u_clusters[rcm_order,:]
			Z_u_clusters_rcm = Z_u_clusters_rcm[:,rcm_order]
			Z_u_clusters_clst = Z_u_clusters[clustersort,:]
			Z_u_clusters_clst = Z_u_clusters_clst[:,clustersort]
			#Z_u_overlaid = Z_u_overlaid[rcm_order,:,:]
			#Z_u_overlaid = Z_u_overlaid[:,rcm_order,:]
			#O_by_O_rcm = O_by_O[rcm_order,:]
			#O_by_O_rcm = O_by_O_rcm[:,rcm_order]
			#scipy.misc.imsave("Z_u_rcm_nocorr_"+str(k)+"_"+str(t)+".png",Z_u_rcm)
			#scipy.misc.imsave("O_by_O_rcm_nocorr_"+str(k)+"_"+str(t)+".png",O_by_O_rcm)
			scipy.misc.imsave("Z_u_rcm_"+str(k)+"_"+str(t)+".png",Z_u_rcm)
			scipy.misc.imsave("Z_u_clst_"+str(k)+"_"+str(t)+".png",Z_u_clst)
			#scipy.misc.imsave("Z_u_bin_rcm_"+str(k)+"_"+str(t)+".png",Z_u_bin_rcm)
			#scipy.misc.imsave("Z_u_overlaid_rcm"+str(k)+"_"+str(t)+".png",Z_u_overlaid)
			scipy.misc.imsave("Z_u_clusters_rcm_"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters_rcm, bytes=True))
			scipy.misc.imsave("Z_u_clusters_clst_"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters_clst, bytes=True))
			#scipy.misc.imsave("Z_u_bin_rcm_"+str(k)+"_"+str(t)+".png",Z_u_bin_rcm)
			#scipy.misc.imsave("Z_u_clusters_rcm_"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters_rcm, bytes=True))
			#scipy.misc.imsave("O_by_O_rcm_"+str(k)+"_"+str(t)+".png",O_by_O_rcm)
			print "Reorder and image saving complete"
			#Now need to do a k-means on the matrices, pluck out groups, and then describe them.  Also want to pluck groupings at different sizes.
	sys.exit()

	print model_inspector.deep_recon_errors(model["training_data"].X,topmodel)
	errors = np.zeros((model["training_data"].X.shape[0],len(topmodel['layers'])))
	Os = []
	O_recons = []

	for i,x in enumerate(model["training_data"].X):
		Os.append(x)
		errors[i] = model_inspector.deep_recon_errors(x,topmodel)[-1]
		O_recons.append(model_inspector.reconstruct(topmodel,x))

if __name__ == "__main__":
	#mnist_test(model_paths = ["cae_l1.pkl","cae_l2.pkl","cae_l3.pkl"])
	#ebird_test()
	phones_test()