try: import simplejson as json
except ImportError: import json
from sklearn.preprocessing import StandardScaler,Imputer
from monary import Monary
from copy import deepcopy

import os, pprint, textwrap, csv, importlib, sys, optparse , time, inspect, DUconfig

from collections import OrderedDict

from spearmint.utils.database.mongodb import MongoDB
from spearmint.resources.resource import print_resources_status
from spearmint.utils.parsing import parse_db_address

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from spearmint import main
import numpy as np
import cPickle as pickle
from pprint import pprint

def monary_load(collection, fields_x, fields_y, start=0,stop=-1, find_args={}):
	numfields = len(fields_x)+len(fields_y)
	with Monary("127.0.0.1") as monary:
		out = monary.query(
			"creeval",
			collection,
			find_args,
			fields_x+fields_y,
			["float32"] * numfields,
			limit=(stop-start),
			offset=start
		)
	for i,col in enumerate(out[0:len(fields_x)]):
		out[i] = np.ma.filled(col,np.ma.mean(col))
		#if any(np.isnan(col)):
	out = np.ma.row_stack(out).T
	X = out[:,0:len(fields_x)]
	y = out[:,len(fields_x):]
	y = (y > 0).astype(int)

	scaler = StandardScaler().fit(X)
	X = scaler.transform(X)
	pickle.dump(scaler,open(collection+"_scaler.pkl","wb"))
	y = np.asarray(y)

	return DenseDesignMatrix(X=X,y=y)

class ConceptualSpace():
	hyper_space = {}
	fixed_hypers = {"layer_fn": ""}
	scratch_path = ""
	spearmint_imports = ""
	spearmint_run = ""
	hypers = {}

	def __init__(self, hyper_space,  fixed_hypers , scratch_path = "", selected_hypers={}):
		self.hyper_space = hyper_space
		self.fixed_hypers = fixed_hypers
		self.scratch_path = scratch_path
		self.hypers = selected_hypers

	def pretrain(self,domain_name,metadata):
		self.fixed_hypers["layer_fn"] = domain_name+"_"+self.fixed_hypers["layer_fn"]
		pretrain_query = {'$and': [deepcopy(metadata["query"])]}
		pretrain_query['$and'].append({metadata["timefield"]: {"$gte": metadata["pretrain_start"]}})
		pretrain_query['$and'].append({metadata["timefield"]: {"$lt": metadata["pretrain_stop"]}})
		ddm = monary_load(domain_name,metadata["fields_x"],metadata["fields_y"],find_args=pretrain_query)
		#scaler = pickle.load(open(domain_name+"_scaler.pkl","rb"))
		DUconfig.dataset = ddm
		params = self.hypers
		params.update(self.fixed_hypers)
		params['save_path'] = self.scratch_path+"step_0/"
		params['yaml_path'] = params['yaml_path'].lstrip("./") #The existing yaml_path starts with "../../" in order to get out of the spearmint dir.
		params["train_stop"] = ddm.X.shape[0]
		result = self.run(ddm,params,logging=True)
		pprint(result)
		return result

	def stepwise_train(self,domain_name,metadata):
		start_time = metadata["pretrain_stop"]
		stop_time = metadata["pretrain_stop"] + metadata["time_slice"]
		while stop_time < metadata["train_stop"]:
			#query
			step_query = {'$and': [deepcopy(metadata["query"])]}
			step_query['$and'].append({metadata["timefield"]: {"$gte": start_time}})
			step_query['$and'].append({metadata["timefield"]: {"$lt": stop_time}})

			#train
			ddm = monary_load(domain_name,metadata["fields_x"],metadata["fields_y"],find_args=step_query)
			DUconfig.dataset = ddm
			params = self.hypers
			params.update(self.fixed_hypers)
			params['save_path'] = self.scratch_path+"step_"+str(len(metadata["steps"]))+"/"
			params['yaml_path'] = params['yaml_path'].lstrip("./") #The existing yaml_path starts with "../../" in order to get out of the spearmint dir.
			params["train_stop"] = ddm.X.shape[0]
			pretrain_prefix = self.scratch_path+"step_"+str(len(metadata["steps"])-1)+"/"
			pretrain_paths = [pretrain_prefix+"ebird_dae_l1.pkl", pretrain_prefix+"ebird_dae_l2.pkl", pretrain_prefix+"ebird_dae_l3.pkl"]
			result = self.run(ddm, params,logging=True, pretrained = pretrain_paths)

			#inspect
			self.inspect(result)
			metadata["steps"].append(result)
			start_time = stop_time
			stop_time += metadata["time_slice"]
			sys.exit()

	def predict(self, record):
		raise NotImplementedError

	def unexpect(self, record):
		raise NotImplementedError

	#This needs to be implemented by subclasses
	def run(self, data, params, logging=False, pretrained = []):
		raise NotImplementedError

	#This needs to be implemented by subclasses
	def inspect(self, model):
		raise NotImplementedError

	def train_mongo(self, collection, fields_x, fields_y, query, threshold = 0.01, look_back = 3):
		name = self.__class__.__name__+str(id(self))
		expdir = os.path.abspath(os.path.join(self.scratch_path,name))+"/"
		if not os.path.exists(expdir):
			os.makedirs(expdir)
		self.fixed_hypers["_mongo"] = {"collection": collection, "fields_x": fields_x, "fields_y": fields_y, "query": query}
		self._train(threshold,look_back)
		return self.hypers



	def train_csv(self, data, threshold = 0.01, look_back = 3):
		name = self.__class__.__name__+str(id(self))
		expdir = os.path.abspath(os.path.join(self.scratch_path,name))+"/"
		if not os.path.exists(expdir):
			os.makedirs(expdir)
		self.write_data(data, expdir)
		self._train(threshold,look_back)

	def _train(self, threshold, look_back):
		name = self.__class__.__name__+str(id(self))
		self.gen_spearmint_template(self.hyper_space, name)
		#self.gen_model_script(textwrap.dedent(self.spearmint_imports), textwrap.dedent(self.spearmint_run), name)
		run_source = inspect.getsourcelines(self.run)[0]
		run_source[0] = run_source[0].replace("self,","") #The written copy of the method is in a class, but the spearmint version is static.
		self.gen_model_script(textwrap.dedent("".join(run_source)), name, spearmintImports=self.spearmint_imports)
		self.run_spearmint(name, threshold=threshold, look_back=look_back)

	def write_data(self,data, name):
		with open(name+"/data.csv", "wb") as csvf:
			writer = csv.writer(csvf)
			writer.writerows(data)

		# Tuples indicate min and max values for real-valued params, lists indicate possible values -- lists of strings for categorical, lists of ints for ordinal
	def gen_spearmint_template(self, params, fname):
		data = {"language": "PYTHON", "main-file":fname+".py", "experiment-name": fname, "likelihood": "NOISELESS", "resources": {"my-machine": {"scheduler":"local","max-concurrent":1}}}
		paramdict = {}
		for k,v in params.iteritems():
			var = {}
			if type(v) is list:
				if type(v[0]) is str:
					var["type"] = "ENUM"
					var["size"] = 1
					var["options"] = v
				elif type(v[0]) is int:
					if len(v) > 2:
						var["type"] = "ENUM"
						var["size"] = 1
						var["options"] = v
					else:
						var["type"] = "INT"
						var["size"] = 1
						var["min"] = v[0]
						var["max"] = v[1]
			elif type(v) is tuple:
				var["type"] = "FLOAT"
				var["size"] = 1
				var["min"] = v[0]
				var["max"] = v[1]
			paramdict[k] = var
		data["variables"] = paramdict
		expdir = os.path.abspath(os.path.join(self.scratch_path,fname))+"/"
		with open(os.path.join(expdir,'config.json'), "w") as f:
			f.writelines(json.dumps(data, indent=4))

	def gen_model_script(self, spearmintRun, fname, spearmintImports = ""):
		experiment_dir = os.path.abspath(os.path.join(self.scratch_path,fname))
		with open(experiment_dir+"/"+fname+".py", "w") as f:
			f.write(spearmintImports+"\nimport numpy as np\nimport sys\nsys.path.append('..')\nfrom conceptual_space import monary_load\nimport DUconfig\nfrom monary import Monary\n\n")
			f.write(spearmintRun+"\n")
			f.write(textwrap.dedent("""\
				def main(job_id, params):
					asciiparams = dict()
					for k,v in params.iteritems():
						k_a = k
						v_a = v
						if type(k_a) is list:
							k_a = k_a[0]
						if type(v_a) is list:
							v_a = v_a[0]
						if type(k_a) is unicode:
							k_a = k_a.encode('ascii','ignore')
						if type(v_a) is unicode:
							v_a = v_a.encode('ascii','ignore')
						asciiparams[k_a] = v_a
					import pprint
					pprint.pprint(asciiparams)
					fixed_params = {0}
					import os
					if os.path.exists('{1}'):
						data = np.genfromtxt('{1}',delimiter=',')
					else:
						data = monary_load(fixed_params['_mongo']['collection'], fixed_params['_mongo']['fields_x'], fixed_params['_mongo']['fields_y'], find_args=fixed_params['_mongo']['query'])
						DUconfig.dataset = data
						fixed_params["train_stop"] = data.X.shape[0]
					hypers = fixed_params
					hypers.update(asciiparams)
					return run(data, hypers)
			""".format(str(self.fixed_hypers), experiment_dir+"/data.csv")))

	def run_spearmint(self, name, threshold = 1e-1, look_back=1):
		options, expt_dir = self.get_options([os.path.abspath(os.path.join(self.scratch_path,name))])

		resources = main.parse_resources_from_config(options)

		# Load up the chooser.
		chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
		chooser = chooser_module.init(options)
		experiment_name = options.get("experiment-name", 'unnamed-experiment')

		self.exp_path =  experiment_name
		# Connect to the database
		db_address = options['database']['address']
		sys.stderr.write('Using database at %s.\n' % db_address)
		db = MongoDB(database_address=db_address)

		stopping = False
		while not stopping:
			for resource_name, resource in resources.iteritems():
				if stopping:
					break
				jobs = main.load_jobs(db, experiment_name)
				# resource.printStatus(jobs)
				# If the resource is currently accepting more jobs
				# TODO: here cost will eventually also be considered: even if the
				#	   resource is not full, we might wait because of cost incurred
				# Note: I chose to fill up one resource and them move on to the next
				# You could also do it the other way, by changing "while" to "if" here

				while resource.acceptingJobs(jobs) and not stopping:
					# Load jobs from DB
					# (move out of one or both loops?) would need to pass into load_tasks
					jobs = main.load_jobs(db, experiment_name)
					#pprint.pprint(main.load_hypers(db, experiment_name))

					# Remove any broken jobs from pending.
					main.remove_broken_jobs(db, jobs, experiment_name, resources)

					# Get a suggestion for the next job
					suggested_job = main.get_suggestion(chooser, resource.tasks, db, expt_dir, options, resource_name)

					# Submit the job to the appropriate resource
					process_id = resource.attemptDispatch(experiment_name, suggested_job, db_address, expt_dir)

					# Set the status of the job appropriately (successfully submitted or not)
					if process_id is None:
						suggested_job['status'] = 'broken'
						main.save_job(suggested_job, db, experiment_name)
					else:
						suggested_job['status'] = 'pending'
						suggested_job['proc_id'] = process_id
						main.save_job(suggested_job, db, experiment_name)

					jobs = main.load_jobs(db, experiment_name)

					# Print out the status of the resources
					# resource.printStatus(jobs)
					print_resources_status(resources.values(), jobs)

					stalled = []
					for task in main.load_task_group(db, options, resource.tasks).tasks.values():
						performance = task.valid_normalized_data_dict["values"][::-1]
						stalled.append(0)
						if len(performance) > look_back:
							print performance[0:look_back+1]
							print "Diffs: ",
							within_thresh = True
							for i,run in enumerate(performance[0:look_back]):
								diff = abs(run - performance[i+1])
								print str(round(diff,2))+", ",
								if diff > threshold:
									within_thresh = False
									print "...No stall"
									break
							if within_thresh:
								stalled[len(stalled)-1] = 1
					if all(stalled):
						print "Stalled!"
						stopping = True
						obj_model = chooser.models[chooser.objective['name']]
						obj_mean, _ = obj_model.function_over_hypers(obj_model.predict, chooser.grid)
						current_best_location = chooser.task_group.from_unit(chooser.grid[np.argmin(obj_mean),:][None])
						for name,var in chooser.task_group.dummy_task.variables_meta.iteritems():
							val = current_best_location[0][var['indices']]
							if var['type'] == 'enum':
								self.hypers[name] = self.hyper_space[name][np.where(val)[0]]
							else:
								self.hypers[name] = float(val)
						print "Best:",self.hypers
			# If no resources are accepting jobs, sleep
			# (they might be accepting if suggest takes a while and so some jobs already finished by the time this point is reached)
			if main.tired(db, experiment_name, resources):
				time.sleep(options.get('polling-time', 5))


	def get_options(self, override_args = None):
		parser = optparse.OptionParser(usage="usage: %prog [options] directory")

		parser.add_option("--config", dest="config_file",
						  help="Configuration file name.",
						  type="string", default="config.json")

		if override_args is not None:
			(commandline_kwargs, args) = parser.parse_args(override_args)
		else:
			(commandline_kwargs, args) = parser.parse_args()

		# Read in the config file
		expt_dir  = os.path.realpath(args[0])
		if not os.path.isdir(expt_dir):
			raise Exception("Cannot find directory %s" % expt_dir)
		expt_file = os.path.join(expt_dir, commandline_kwargs.config_file)

		try:
			with open(expt_file, 'r') as f:
				options = json.load(f, object_pairs_hook=OrderedDict)
		except:
			raise Exception("config.json did not load properly. Perhaps a spurious comma?")
		options["config"]  = commandline_kwargs.config_file


		# Set sensible defaults for options
		options['chooser']  = options.get('chooser', 'default_chooser')
		if 'tasks' not in options:
			options['tasks'] = {'main' : {'type' : 'OBJECTIVE', 'likelihood' : options.get('likelihood', 'GAUSSIAN')}}

		# Set DB address
		db_address = parse_db_address(options)
		if 'database' not in options:
			options['database'] = {'name': 'spearmint', 'address': db_address}
		else:
			options['database']['address'] = db_address

		if not os.path.exists(expt_dir):
			sys.stderr.write("Cannot find experiment directory '%s'. "
							 "Aborting.\n" % (expt_dir))
			sys.exit(-1)

		return options, expt_dir

class SDAConceptualSpace(ConceptualSpace):
	hyper_space =  {"sparse_coef_l1": (0.1,2),
					"sparse_p_l1": (0.01,0.2),
					"nhid_l1": range(100,201,50),
					"nhid_l2": range(100,201,50),
					"sparse_coef_l2": (0.1,2),
					"sparse_p_l2": (0.01,0.2),
					"nhid_l3": range(100,201,50),
					"sparse_coef_l3": (0.1,2),
					"sparse_p_l3": (0.01,0.2)
	}
					#"corrupt_l1": (0.2,0.5),
					#"corrupt_l2": (0.2,0.5),
					#"corrupt_l3": (0.2,0.5),
					#"nhid_l1": range(100,1000,50),

	fixed_hypers = {"train_stop": 50000,
					"batch_size": 100,
					"monitoring_batch_size": 100,
					"max_epochs": 1,
					"save_path": ".",
					"yaml_path": "../../model_yamls/",
	                "layer_fn": "sdae",
	                "num_layers": 3,
					"corrupt_l1": 0.33,
					"corrupt_l2": 0.33,
					"corrupt_l3": 0.33,
	                "n_folds": 5
	}

#	spearmint_imports =  """\
#							from pylearn2.config import yaml_parse
#						"""

	def inspect(self,model):
		pass

	def run(self,data, hypers, logging=False, pretrained = []):
		from pylearn2.config import yaml_parse
		print hypers

		objectives = []
		model_files = []
		models = []
		for layer_num in range(1,hypers['num_layers']+1):
			layer_yaml = open(hypers["yaml_path"]+hypers["layer_fn"]+"_l"+str(layer_num)+".yaml", 'r').read()
			layer_hypers = {'train_stop': hypers["train_stop"],
								'batch_size': hypers["batch_size"],
								'n_folds': hypers["n_folds"],
								'monitoring_batch_size': hypers["monitoring_batch_size"],
								'nhid': hypers["nhid_l"+str(layer_num)],
								'max_epochs': hypers["max_epochs"],
								'corrupt': hypers["corrupt_l"+str(layer_num)],
								'sparse_coef': hypers["sparse_coef_l"+str(layer_num)],
								'sparse_p': hypers["sparse_p_l"+str(layer_num)],
								'save_path': hypers["save_path"],
								'layer_fn': hypers["layer_fn"]}
			if layer_num > 1:
				layer_hypers["nvis"] = hypers["nhid_l"+str(layer_num-1)]
			layer_yaml = layer_yaml % (layer_hypers)
			print "-----LAYER_"+str(layer_num)+"-----"
			print layer_yaml
			print "-----------------"
			train = yaml_parse.load(layer_yaml)
			if len(pretrained):
				train.model = pickle.load(open(pretrained[layer_num-1]))
				print "Replaced model with pretrained one from",pretrained[layer_num-1]
			train.main_loop()

			obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers])
			print obj
			objectives.append(obj)
			models.append(train)
			model_files.append(layer_hypers['save_path']+hypers["layer_fn"]+"_l"+str(layer_num)+".pkl")

		objective = float(np.sum(objectives))
		if logging:
			return {"objective": objective,
					"model_files": model_files,
					"models": models}
		else:
			return objective

		'''
		#l1_obj = train.model.monitor.channels['test_objective'].val_record[-1]
		layer2_yaml = open(hypers["yaml_path"]+'spdae_l2.yaml', 'r').read()
		hyper_params_l2 = {'train_stop' : hypers["train_stop"],
							'batch_size' : hypers["batch_size"],
							'n_folds' : hypers["n_folds"],
							'monitoring_batch_size' : hypers["monitoring_batch_size"],
							'nvis' : hyper_params_l1['nhid'],
							'nhid' : hypers["nhid_l2"],
							'max_epochs' :hypers["max_epochs"],
							'corrupt' : hypers["corrupt_l2"],
							'sparse_coef' : hypers["sparse_coef_l2"],
							'sparse_p' : hypers["sparse_p_l2"],
							'save_path' : hypers["save_path"]}
		layer2_yaml = layer2_yaml % (hyper_params_l2)
		print "-----LAYER_2-----"
		print layer2_yaml
		print "-----------------"
		train2 = yaml_parse.load(layer2_yaml)
		if len(pretrained):
			train2.model = pickle.load(open(pretrained[1]))
			print "Replaced model with pretrained one from",pretrained[1]
		train2.main_loop()
		l2_obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train2.trainers])
		print l2_obj
		layer3_yaml = open(hypers["yaml_path"]+'spdae_l3.yaml', 'r').read()
		hyper_params_l3 = {'train_stop' : hypers["train_stop"],
							'batch_size' : hypers["batch_size"],
							'n_folds' : hypers["n_folds"],
							'monitoring_batch_size' : hypers["monitoring_batch_size"],
							'nvis' : hyper_params_l2['nhid'],
							'nhid' : hypers["nhid_l3"],
							'max_epochs' :hypers["max_epochs"],
							'corrupt' : hypers["corrupt_l3"],
							'sparse_coef' : hypers["sparse_coef_l3"],
							'sparse_p' : hypers["sparse_p_l3"],
							'save_path' : hypers["save_path"]}
		layer3_yaml = layer3_yaml % (hyper_params_l3)
		print "-----LAYER_3-----"
		print layer3_yaml
		print "-----------------"
		train3 = yaml_parse.load(layer3_yaml)
		if len(pretrained):
			train3.model = pickle.load(open(pretrained[2]))
			print "Replaced model with pretrained one from",pretrained[2]
		train3.main_loop()
		l3_obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train3.trainers])
		print l3_obj

		objective = float(l1_obj + l2_obj + l3_obj)
		if logging:
			return {"objective": objective,
					"model_files": [hyper_params_l1['save_path']+"ebird_dae_l1.pkl", hyper_params_l2['save_path']+"ebird_dae_l2.pkl", hyper_params_l3['save_path']+"ebird_dae_l3.pkl"],
					"models": [train1,train2,train3]}
		else:
			return objective

		#'''



	spearmint_run =  """\
						def run(data, hypers):
							print hypers
							layer1_yaml = open(hypers["yaml_path"]+"spdae_l1.yaml", 'r').read()
							hyper_params_l1 = {'train_stop' : hypers["train_stop"],
												'batch_size' : hypers["batch_size"],
												'n_folds' : hypers["n_folds"],
												'monitoring_batch_size' : hypers["monitoring_batch_size"],
												'nhid' : hypers["nhid_l1"],
												'max_epochs' :hypers["max_epochs"],
												'corrupt' : hypers["corrupt_l1"],
												'sparse_coef' : hypers["sparse_coef_l1"],
												'sparse_p' : hypers["sparse_p_l1"],
												'save_path' : hypers["save_path"]}
							layer1_yaml = layer1_yaml % (hyper_params_l1)
							print "-----LAYER_1-----"
							print layer1_yaml
							print "-----------------"
							train = yaml_parse.load(layer1_yaml)
							train.main_loop()
							l1_obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers])
							print l1_obj
							#l1_obj = train.model.monitor.channels['test_objective'].val_record[-1]

							layer2_yaml = open(hypers["yaml_path"]+'spdae_l2.yaml', 'r').read()
							hyper_params_l2 = {'train_stop' : hypers["train_stop"],
												'batch_size' : hypers["batch_size"],
												'n_folds' : hypers["n_folds"],
												'monitoring_batch_size' : hypers["monitoring_batch_size"],
												'nvis' : hyper_params_l1['nhid'],
												'nhid' : hypers["nhid_l2"],
												'max_epochs' :hypers["max_epochs"],
												'corrupt' : hypers["corrupt_l2"],
												'sparse_coef' : hypers["sparse_coef_l2"],
												'sparse_p' : hypers["sparse_p_l2"],
												'save_path' : hypers["save_path"]}
							layer2_yaml = layer2_yaml % (hyper_params_l2)
							print "-----LAYER_2-----"
							print layer2_yaml
							print "-----------------"
							train = yaml_parse.load(layer2_yaml)
							train.main_loop()
							l2_obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers])
							print l2_obj
							layer3_yaml = open(hypers["yaml_path"]+'spdae_l3.yaml', 'r').read()
							hyper_params_l3 = {'train_stop' : hypers["train_stop"],
												'batch_size' : hypers["batch_size"],
												'n_folds' : hypers["n_folds"],
												'monitoring_batch_size' : hypers["monitoring_batch_size"],
												'nvis' : hyper_params_l2['nhid'],
												'nhid' : hypers["nhid_l3"],
												'max_epochs' :hypers["max_epochs"],
												'corrupt' : hypers["corrupt_l3"],
												'sparse_coef' : hypers["sparse_coef_l3"],
												'sparse_p' : hypers["sparse_p_l3"],
												'save_path' : hypers["save_path"]}
							layer3_yaml = layer3_yaml % (hyper_params_l3)
							print "-----LAYER_3-----"
							print layer3_yaml
							print "-----------------"
							train = yaml_parse.load(layer3_yaml)
							train.main_loop()
							l3_obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers])
							print l3_obj

							return float(l1_obj + l2_obj + l3_obj)
					"""

	def __init__(self, scratch_path = "", selected_hypers={}):
		ConceptualSpace.__init__(self, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)

if __name__ == "__main__":
	cs = SDAConceptualSpace("data/")
	cs.train({}, threshold = 0.01, look_back = 5)
	#from sdae_l3_reconstruct import reconstruct
	#reconstruct("data/"+cs.exp_path+"/dae")
	#from sdae_l3_show_weights_decoder import show_weights
	#show_weights("data/"+cs.exp_path+"/dae")
