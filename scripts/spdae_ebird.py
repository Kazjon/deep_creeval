from pylearn2.config import yaml_parse
import numpy as np
import DUconfig, ebird_load_script, model_inspector, time

dataset_start = 0
dataset_stop = -1

species_to_retrieve = ['Turdus_migratorius'] #['Picoides_pubescens', 'Turdus_migratorius']  # Empty list means all

start_time = time.time()
#DUconfig.dataset = ebird_load_script.load_ebird_data(dataset_start,dataset_stop,species_to_retrieve=species_to_retrieve)
DUconfig.dataset = ebird_load_script.monary_load(dataset_start,dataset_stop,species_to_retrieve=species_to_retrieve)
dataset_stop = DUconfig.dataset.X.shape[0]

print 'Query took',time.time() - start_time,'seconds. Training with',dataset_stop,'examples.'


layer1_yaml = open('ebird_spdae_l1.yaml', 'r').read()
hyper_params_l1 = { 'batch_size' : 100,
					'monitoring_batches' : dataset_stop/1000,
					'nhid' : 100,
					'max_epochs' : 5,
					'corrupt' : 0.5,
					'sparse_coef' : 1,
					'sparse_p' : 0.15,
					'save_path' : '.'}
layer1_yaml = layer1_yaml % (hyper_params_l1)

print layer1_yaml

train = yaml_parse.load(layer1_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()
l1_obj = train.model.monitor.channels['objective'].val_record[-1]
#'''

layer2_yaml = open('ebird_spdae_l2.yaml', 'r').read()
hyper_params_l2 = { 'batch_size' :  hyper_params_l1['batch_size'],
					'monitoring_batches' : hyper_params_l1['monitoring_batches'],
					'nvis' : hyper_params_l1['nhid'],
					'nhid' : 100,
					'max_epochs' : 5,
					'corrupt' : 0.5,
					'sparse_coef' : 1,
					'sparse_p' : 0.15,
					'save_path' : '.'}
layer2_yaml = layer2_yaml % (hyper_params_l2)

print layer2_yaml

train = yaml_parse.load(layer2_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()
l2_obj = train.model.monitor.channels['objective'].val_record[-1]
#'''

layer3_yaml = open('ebird_spdae_l3.yaml', 'r').read()
hyper_params_l3 = {	'batch_size' : hyper_params_l1['batch_size'],
					'monitoring_batches' : hyper_params_l1['monitoring_batches'],
					'nvis' : hyper_params_l2['nhid'],
					'nhid' : 100,
					'max_epochs' : 5,
					'corrupt' : 0.5,
					'sparse_coef' : 1,
					'sparse_p' : 0.15,
					'save_path' : '.'}
layer3_yaml = layer3_yaml % (hyper_params_l3)
print layer3_yaml

train = yaml_parse.load(layer3_yaml)
DUconfig.costs.append(train.algorithm.cost)
train.main_loop()

l3_obj = train.model.monitor.channels['objective'].val_record[-1]

print l1_obj
print l2_obj
print l3_obj

model = model_inspector.load_model(["ebird_dae_l1.pkl","ebird_dae_l2.pkl","ebird_dae_l3.pkl"],DUconfig.costs)

print model_inspector.recon_errors(DUconfig.dataset.X.astype(np.float32),model)