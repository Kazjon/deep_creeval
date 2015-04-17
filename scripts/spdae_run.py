from pylearn2.config import yaml_parse
import DUconfig
from pylearn2.datasets.mnist import MNIST
import numpy as np
import model_inspector


layer1_yaml = open('spdae_l1.yaml', 'r').read()
hyper_params_l1 = {'train_stop' : 50000,
					'batch_size' : 100,
					'monitoring_batches' : 10,
					'nhid' : 500,
					'max_epochs' : 20,
					'corrupt' : 0.5,
					'sparse_coef' : 1,
					'sparse_p' : 0.05,
					'save_path' : '.'}
layer1_yaml = layer1_yaml % (hyper_params_l1)
print layer1_yaml

train = yaml_parse.load(layer1_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()
l1_obj = train.model.monitor.channels['objective'].val_record[-1]

'''
layer2_yaml = open('spdae_l2.yaml', 'r').read()
hyper_params_l2 = {'train_stop' : 50000,
					'batch_size' : 100,
					'monitoring_batches' : 10,
					'nvis' : hyper_params_l1['nhid'],
					'nhid' : 250,
					'max_epochs' : 20,
					'corrupt' : 0.1,
					'sparse_coef' : 10,
					'sparse_p' : 0.05,
					'save_path' : '.'}
layer2_yaml = layer2_yaml % (hyper_params_l2)
print layer2_yaml

train = yaml_parse.load(layer2_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()
l2_obj = train.model.monitor.channels['objective'].val_record[-1]
#'''
'''
layer3_yaml = open('spdae_l3.yaml', 'r').read()
hyper_params_l3 = {'train_stop' : 50000,
					'batch_size' : 100,
					'monitoring_batches' : 10,
					'nvis' : hyper_params_l2['nhid'],
					'nhid' : 100,
					'max_epochs' : 10,
					'corrupt' : 0.5,
					'sparse_coef' : hyper_params_l1['sparse_coef'],
					'sparse_p' : hyper_params_l1['sparse_p'],
					'save_path' : '.'}
layer3_yaml = layer3_yaml % (hyper_params_l3)
print layer3_yaml

train = yaml_parse.load(layer3_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()

l3_obj = train.model.monitor.channels['objective'].val_record[-1]
#'''

print l1_obj
#print l2_obj
#print l3_obj

'''
#model = model_inspector.load_model(["dae_l1.pkl","dae_l2.pkl"],DUconfig.costs)
model = model_inspector.load_model(["dae_l1.pkl","dae_l2.pkl","dae_l3.pkl"],DUconfig.costs)
model_inspector.save_as_patches(model_inspector.show_weights(model),[28,28],out_path="outpatches.png")

data = MNIST('train',start=50000,stop=50100)
errors = np.zeros((data.X.shape[0],len(model['layers'])))
for i,x in enumerate(data.X):
	errors[i] = model_inspector.reconstruction_errors(x,model)
	print errors[i]
	# Get reconstruction error
#Order samples by recon error
#Save 100 lowest and 100 highest as images
#'''