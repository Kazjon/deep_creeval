__author__ = 'kazjon'


#Import the MNIST database again
import sklearn.preprocessing
from pylearn2.config import yaml_parse
from pylearn2.datasets.mnist import MNIST
import numpy as np
import model_inspector
import DUconfig,sys
import scipy.misc, csv

img_size = [28,28]

sparse_coef = 0
sparse_p = 0.2

#path_prefixes = ["tanh_null","tanh_null_untied", "sigmoid_null", "sigmoid_null_untied","sigmoid_sigmoid", "sigmoid_sigmoid_untied"]
#path_prefixes += ["sigmoid_null_256_64_0.1","sigmoid_null_256_64_1","sigmoid_null_256_64_10"]
#path_prefixes += ["tanh_null_256_64_1","tanh_null_256_32_1","tanh_null_256_16_1"]
#path_prefixes += ["sigmoid_null_512_64_1","sigmoid_null_512_32_1","sigmoid_null_512_16_1"]
#path_prefixes += ["sigmoid_null_256_64_1","sigmoid_null_256_32_1","sigmoid_null_256_16_1"]
#path_prefixes += ["sigmoid_null_128_64_1","sigmoid_null_128_32_1","sigmoid_null_128_16_1"]
#path_prefixes += ["sigmoid_null_128_128_1","tanh_null_128_128_1"]
#path_prefixes += ["tanh_null_128_128_1_0.1","tanh_null_128_128_1_0.01", "tanh_null_128_128_1_0.005"]
#path_prefixes += ["sigmoid_null_128_128_1_0.1","sigmoid_null_128_128_1_0.01", "sigmoid_null_128_128_1_0.005", "sigmoid_null_128_128_1_0.5"]
#path_prefixes += ["sigmoid_null_128_128_1_0.2",'tanh_null_256_128_0.5_0.05','sigmoid_null_256_128_0.5_0.05']
#path_prefixes = ['tanh_null_256_128_0.5_0.05','tanh_null_256_128_0.5_0.05','tanh_null_256_128_1_0.05','tanh_null_256_128_2_0.05','tanh_null_256_128_5_0.05','tanh_null_256_128_10_0.05']
path_prefixes = ['tanh_null_256_128_0.1_0.05','tanh_null_256_128_0.02_0.05','tanh_null_256_128_0.01_0.05']
path_prefixes = ['sigmoid_null_256_64_0.05_0.05']

DUconfig.costs = yaml_parse.load('cost : !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {}')['cost']
data = MNIST('train',start=50000,stop=51000)

for path in path_prefixes:
	layerpaths = [path+"/cae_l1.pkl", path+"/cae_l2.pkl"]
	for l in range(len(layerpaths)):
		layerpaths_ = layerpaths[:l+1]
		model = model_inspector.load_model(layerpaths_,DUconfig.costs)
		#model = model_inspector.load_model([path+"/cae_l1.pkl"],DUconfig.costs)
		#'''
		model_inspector.save_as_patches(model_inspector.show_weights(model),[28,28],out_path=path+"/outpatches_l"+str(l+1)+".png")
		encoded = model_inspector.encode(data.X,model)
		scipy.misc.imsave(path+"/hidden_activations_l"+str(l+1)+".png",encoded)
		with open(path+"/encoded_l"+str(l+1)+".csv", "wb") as csvf:
			csvwriter = csv.writer(csvf)
			csvwriter.writerows(encoded)
		scaler = sklearn.preprocessing.StandardScaler().fit(encoded)
		normed_encoded = scaler.transform(encoded)
		scipy.misc.imsave(path+"/hidden_activations_normed_l"+str(l+1)+".png",normed_encoded)
		scaled_encoded = np.multiply(encoded,scaler.std_)
		scipy.misc.imsave(path+"/hidden_activations_scaled_l"+str(l+1)+".png",scaled_encoded)
		scaled_patches = np.multiply(model_inspector.show_weights(model).T,scaler.std_).T
		scaled_patches -= scaled_patches.min()
		scaled_patches /= scaled_patches.max()
		scaled_patches -= 0.5
		scaled_patches *= 2.0
		model_inspector.save_as_patches(scaled_patches,img_size,out_path=path+"/outpatches_scaled_l"+str(l+1)+".png", rescale=False)
		#'''
		print model_inspector.deep_recon_errors(data.X,model)
		errors = np.zeros((data.X.shape[0],len(model['layers'])))
		originals = []
		recons = []

		for i,x in enumerate(data.X):
			originals.append(x.reshape(img_size))
			errors[i] = model_inspector.deep_recon_errors(x,model)[-1]
			recons.append(model_inspector.reconstruct(model,x).reshape(img_size))

		error_sums = [sum(x) for x in errors]
		ary = zip(error_sums,originals,recons)
		ary.sort(key=lambda row: row[0])
		error_sums,originals,recons = zip(*ary)

		originals = np.concatenate(originals)
		recons = np.concatenate(recons)

		num_to_print = 50

		if len(error_sums) < 2 * num_to_print:
			scipy.misc.imsave(path+"/comparison_l"+str(l+1)+".png",np.concatenate((np.atleast_2d(originals*2),np.atleast_2d(recons)),axis=1))
		else:
			num_to_print *= img_size[0]
			scipy.misc.imsave(path+"/comparison_top_l"+str(l+1)+".png",np.concatenate((np.atleast_2d(originals[0:num_to_print]*2),np.atleast_2d(recons[0:num_to_print])),axis=1))
			scipy.misc.imsave(path+"/comparison_bottom_l"+str(l+1)+".png",np.concatenate((np.atleast_2d(originals[-num_to_print:]*2),np.atleast_2d(recons[-num_to_print:])),axis=1))

#Order samples by recon error
#Save 100 lowest and 100 highest as images