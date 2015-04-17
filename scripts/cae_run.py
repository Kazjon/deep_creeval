from pylearn2.config import yaml_parse
import DUconfig
from pylearn2.datasets.mnist import MNIST
import numpy as np
import model_inspector

layers = 2
errors = []

layer1_yaml = open('cae_l1.yaml', 'r').read()
hyper_params_l1 = {'train_stop' : 50000,
					'batch_size' : 100,
					'monitoring_batches' : 100,
					'nhid' : 256,
                    'irange' : 0.05,
					'max_epochs' : 25,
					'contract_coef' : 0.1,
					'save_path' : 'sigmoid_null_256_64_0.1_0.05',
					'act_enc' : 'sigmoid',
                    'act_dec' : 'null',
					'tied': True}
layer1_yaml = layer1_yaml % (hyper_params_l1)
layer2_yaml = open('cae_l2.yaml', 'r').read()
hyper_params_l2 = {'train_stop' : 50000,
					'batch_size' : 100,
					'monitoring_batches' : 100,
					'nvis' : hyper_params_l1['nhid'],
					'nhid' : 64,
                    'irange' : hyper_params_l1['irange'],
					'max_epochs' : hyper_params_l1['max_epochs'],
					'contract_coef' : 1,
					'save_path' : hyper_params_l1['save_path'],
					'act_enc' : hyper_params_l1['act_enc'],
					'act_dec' : hyper_params_l1['act_dec'],
					'tied' : hyper_params_l1['tied']}
layer2_yaml = layer2_yaml % (hyper_params_l2)
layer3_yaml = open('cae_l3.yaml', 'r').read()
hyper_params_l3 = {'train_stop' : 50000,
					'batch_size' : 5,
					'monitoring_batches' : 5000,
					'nvis' : hyper_params_l2['nhid'],
					'nhid' : 100,
					'max_epochs' : hyper_params_l1['max_epochs'],
					'contract_coef' : 1,
					'act_enc' : hyper_params_l1['act_enc'],
					'act_dec' : hyper_params_l1['act_dec'],
					'save_path' : hyper_params_l1['save_path'],
					'tied' : hyper_params_l1['tied']}
layer3_yaml = layer3_yaml % (hyper_params_l3)

print layer1_yaml
train = yaml_parse.load(layer1_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()
errors.append(train.model.monitor.channels['objective'].val_record[-1])

if layers > 1:
	print layer2_yaml
	train = yaml_parse.load(layer2_yaml)
	DUconfig.costs.append(train.algorithm.cost)
	train.main_loop()
	errors.append(train.model.monitor.channels['objective'].val_record[-1])

if layers > 2:
	print layer3_yaml
	train = yaml_parse.load(layer3_yaml)
	DUconfig.costs.append(train.algorithm.cost)
	train.main_loop()
	errors.append(train.model.monitor.channels['objective'].val_record[-1])
#'''

print errors

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