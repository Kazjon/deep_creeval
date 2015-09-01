# This class is an unabashed hack:
#   pylearn2's YAML files expect a callable function that will return their dataset, but we don't want to have to retrieve
#   our data multiple times for training and testing different layers.

dataset = []
test_dataset = []
costs = []

def get_train_dataset():
	return dataset

def get_test_dataset():
	return test_dataset

def get_dataset():
	return dataset

from pylearn2.models.autoencoder import ContractiveAutoencoder

def CAE(nvis, nhid, irange, act_enc, tied_weights, act_dec):
	#return ContractiveAutoencoder(nvis=nvis, nhid=nhid,irange=irange,corruptor=corruptor, act_enc=act_enc, act_dec=act_dec)
	return ContractiveAutoencoder(nvis=nvis, nhid=nhid,irange=irange, tied_weights=tied_weights, act_enc=act_enc, act_dec=act_dec)