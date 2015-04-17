
from pylearn2.config import yaml_parse
layer3_yaml = open('spdae_l3.yaml', 'r').read()
hyper_params_l3 = {'train_stop' : 50000,
                   'batch_size' : 100,
                   'monitoring_batches' : 5,
                   'nvis' : 200,
                   'nhid' : 10,
                   'max_epochs' : 20,
                   'corrupt' : 0.5,
                   'sparse_coef' : 10,
                   'sparse_p' : 0.1,
                   'save_path' : '.'}
layer3_yaml = layer3_yaml % (hyper_params_l3)
print layer3_yaml

train = yaml_parse.load(layer3_yaml)
train.main_loop()