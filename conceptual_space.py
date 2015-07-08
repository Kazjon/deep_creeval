try: import simplejson as json
except ImportError: import json
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.manifold import TSNE
from monary import Monary
from copy import deepcopy

import os, os.path, pprint, textwrap, csv, importlib, sys, optparse , time, inspect, DUconfig, model_inspector, scipy.misc, scipy.spatial, sklearn.mixture, matplotlib, qutip, itertools, copy, sklearn.preprocessing

from collections import OrderedDict

from spearmint.utils.database.mongodb import MongoDB
from spearmint.resources.resource import print_resources_status
from spearmint.utils.parsing import parse_db_address

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.monitor import Monitor

from spearmint import main
import numpy as np
import cPickle as pickle
from pprint import pprint
import types

import pandas as pd
from pandas.tools.plotting import radviz
import matplotlib.pyplot as plt

import bh_tsne.bhtsne as bh_tsne

plt.style.use('ggplot')
np.set_printoptions(linewidth=200)


from networkx.utils import reverse_cuthill_mckee_ordering

from scipy.spatial.distance import cosine
from scipy.spatial.distance import squareform
from sklearn.manifold import _utils
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import validation


MACHINE_EPSILON = np.finfo(np.double).eps

iterables = (types.DictType, types.ListType, types.TupleType, types.GeneratorType)
keepables = (types.TypeType, types.BooleanType, types.IntType, types.LongType, types.FloatType, types.ComplexType, types.StringType, types.UnicodeType)
def sanitise_for_str_out(d):
	if isinstance(d,types.DictType):
		#print "iterating through dict",d
		for k in d.keys():
			if isinstance(d[k],iterables):
				sanitise_for_str_out(d[k])
				#print "recursing to k:",k,"v:",d[k]
			elif not isinstance(d[k],keepables):
				del d[k]
				#print "deleted k:",k,"v:",d[k]
		#print "... kept",d
	elif isinstance(d,(types.ListType,types.TupleType,types.GeneratorType)):
		#print "iterating through",d
		for k,i in enumerate(d):
			if isinstance(i,iterables):
				sanitise_for_str_out(i)
				#print "recursing to item",i
			elif not isinstance(i,keepables):
				del d[k]
				#print "deleted item:",i
		#print "... kept",d

def _joint_probabilities(distances, desired_perplexity, verbose):
	"""Compute joint probabilities p_ij from distances.

	Parameters
	----------
	distances : array, shape (n_samples * (n_samples-1) / 2,)
		Distances of samples are stored as condensed matrices, i.e.
		we omit the diagonal and duplicate entries and store everything
		in a one-dimensional array.

	desired_perplexity : float
		Desired perplexity of the joint probability distributions.

	verbose : int
		Verbosity level.

	Returns
	-------
	P : array, shape (n_samples * (n_samples-1) / 2,)
		Condensed joint probability matrix.
	"""
	# Compute conditional probabilities such that they approximately match
	# the desired perplexity
	conditional_P = _utils._binary_search_perplexity(
		distances, desired_perplexity, verbose)
	P = conditional_P + conditional_P.T
	sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
	P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
	return P

def monary_load(collection, fields_x, fields_y, start=0,stop=0, find_args={}):
	print collection, fields_x,fields_y,start,stop,find_args
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

	if X.shape[0]:
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

	def __init__(self, domain_name, hyper_space,  fixed_hypers , scratch_path = "", selected_hypers={}):
		self.domain_name = domain_name
		self.hyper_space = hyper_space
		self.fixed_hypers = fixed_hypers
		self.scratch_path = scratch_path
		self.hypers = copy.deepcopy(selected_hypers)
		self.fixed_hypers["layer_fn"] = domain_name+"_"+self.fixed_hypers["layer_fn"]

	def pretrain(self,metadata):
		pretrain_query = {'$and': [deepcopy(metadata["query"])]}
		pretrain_query['$and'].append({metadata["timefield"]: {"$gte": metadata["pretrain_start"]}})
		pretrain_query['$and'].append({metadata["timefield"]: {"$lt": metadata["pretrain_stop"]}})
		ddm = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=pretrain_query)
		if not ddm.X.shape[0]:
			sys.exit("Pretrain failed as no examples were found within the pretraining time window of "+str(metadata["pretrain_start"])+" to "+str(metadata["pretrain_stop"]))
		#scaler = pickle.load(open(domain_name+"_scaler.pkl","rb"))
		DUconfig.dataset = ddm
		params = self.hypers
		params.update(self.fixed_hypers)
		params['save_path'] = self.scratch_path+"step_0/"
		params['yaml_path'] = params['yaml_path'].lstrip("./") #The existing yaml_path starts with "../../" in order to get out of the spearmint dir.
		params["train_stop"] = ddm.X.shape[0]
		result = self.run(ddm,params,logging=True, cv=False)
		result["start"] = metadata["pretrain_start"]
		result["stop"] =  metadata["pretrain_stop"]
		#pprint(result)
		return result

	def stepwise_train(self,metadata):
		#metadata["time_slice"] = 1 # Debug measure to speed shit up.
		start_time = metadata["pretrain_stop"]
		stop_time = metadata["pretrain_stop"] + metadata["time_slice"]
		while stop_time < metadata["train_stop"]:
			#query
			step_query = {'$and': [deepcopy(metadata["query"])]}
			step_query['$and'].append({metadata["timefield"]: {"$gte": start_time}})
			step_query['$and'].append({metadata["timefield"]: {"$lt": stop_time}})

			#train
			ddm = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=step_query)
			if ddm.X.shape[0]:
				DUconfig.dataset = ddm
				params = self.hypers
				params.update(self.fixed_hypers)
				params['save_path'] = self.scratch_path+"step_"+str(len(metadata["steps"]))+"/"
				params['yaml_path'] = params['yaml_path'].lstrip("./") #The existing yaml_path starts with "../../" in order to get out of the spearmint dir.
				params["train_stop"] = ddm.X.shape[0]
				pretrain_prefix = self.scratch_path+"step_"+str(len(metadata["steps"])-1)+"/"
				pretrain_paths = [pretrain_prefix+params["layer_fn"]+"_l"+str(i+1)+".pkl" for i in range(params["num_layers"])]
				result = self.run(ddm, params,logging=True, pretrained = pretrain_paths, cv=False)
			else:
				result = {"error": "NO DATA FOUND FOR THIS TIMESTEP."}
			result["start"] = start_time
			result["stop"] =  stop_time
			metadata["steps"].append(result)
			print metadata["steps"]
			start_time = stop_time
			stop_time += metadata["time_slice"]

	def stepwise_inspect(self,metadata,sample_size=50000,save_path="", resample=1):
		start_time = metadata["pretrain_start"]
		for i,step in enumerate(metadata["steps"][1:]):
			if i > 12:
				sys.exit()
			stop_time = step["stop"]
			print "Inspecting step "+str(i)+". "+str(start_time)+"--"+str(stop_time)+"."
			if "error" not in step.keys():
				#query
				print "pre deepcopy"
				step_query = {'$and': [deepcopy(metadata["query"])]}
				print "post deepcopy"
				step_query['$and'].append({metadata["timefield"]: {"$gte": start_time}})
				#step_query['$and'].append({metadata["timefield"]: {"$gte": metadata["pretrain_start"]}})
				step_query['$and'].append({metadata["timefield"]: {"$lt": stop_time}})
				print "   --- Query built."

				#sample
				data = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=step_query)
				for k in range(resample): #Testing multiple clusterings of each step with different samplings.
					if data.X.shape[0] > sample_size:
						data_sample = data.X[np.random.choice(data.X.shape[0],size=sample_size),:]
					else:
						data_sample = data.X
					#del data
					print "   --- Data loaded."

					#load models
					model1 = {}
					model1["model"] = model_inspector.load_model(step["model_files"],self.fixed_hypers["costs"])
					model1["F_by_A"] = model_inspector.show_weights(model1["model"],save_patches=False)
					model1["F_by_O"] = model_inspector.encode(data_sample,model1["model"]).T
					model1["F_by_O_scaler"] = StandardScaler().fit(model1["F_by_O"].T)
					model1["F_by_O_normed"] = model1["F_by_O_scaler"].transform(model1["F_by_O"].T).T
					model1["E_by_O"] = model_inspector.recon_errors(data_sample,model1["model"])
					model1["name"] = "current_"+str(k)
					print "   --- Model 1 load complete."

					prev = i # i is actually already the index of the previous model, since we're iterating from element 1 onwards.
					while "error" in metadata['steps'][prev].keys():
						prev -= 1
						if prev < 0:
							sys.exit("No valid previous steps found, cannot perform comparison.")
					model2 = {}
					model2["model"] = model_inspector.load_model(metadata['steps'][prev]["model_files"],self.fixed_hypers["costs"])
					model2["F_by_A"] = model_inspector.show_weights(model2["model"],save_patches=False)
					model2["F_by_O"] = model_inspector.encode(data_sample,model2["model"]).T
					model2["F_by_O_scaler"] = StandardScaler().fit(model2["F_by_O"].T)
					model2["F_by_O_normed"] = model2["F_by_O_scaler"].transform(model2["F_by_O"].T).T
					model2["E_by_O"] = model_inspector.recon_errors(data_sample,model2["model"])
					model2["name"] = "previous_"+str(k)
					print "   --- Model 2 load complete"

					self.compare_models(data_sample, model1, model2, save_path=os.path.dirname(step["model_files"][0]))
					del model1
					del model2
					del data_sample
			else:
				print "   --- Error: "+step["error"]


	def predict(self, record):
		raise NotImplementedError

	def unexpect(self, record):
		raise NotImplementedError

	#This needs to be implemented by subclasses
	def run(self, data, params, logging=False, pretrained = [], cv=True):
		raise NotImplementedError

	#This needs to be implemented by subclasses
	def inspect(self, model, save_path=""):
		raise NotImplementedError

	#This needs to be implemented by subclasses
	def compare_models(self, O_by_A, model1, model2,save_path=""):
		raise NotImplementedError

	def train_mongo(self, fields_x, fields_y, query, threshold = 0.01, look_back = 3):
		name = self.__class__.__name__+str(id(self))
		expdir = os.path.abspath(os.path.join(self.scratch_path,name))+"/"
		if not os.path.exists(expdir):
			os.makedirs(expdir)
		self.fixed_hypers["_mongo"] = {"collection": self.domain_name, "fields_x": fields_x, "fields_y": fields_y, "query": query}
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
			sanitised_fixed_hypers = deepcopy(self.fixed_hypers) # This gets rid of objects that seem to get left in the hypers -- particularly a Costs object.
			sanitise_for_str_out(sanitised_fixed_hypers)
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
						#fixed_params["yaml_path"] = fixed_params["yaml_path"] + fixed_params['_mongo']['collection'] + "_"
					hypers = fixed_params
					hypers.update(asciiparams)
					return run(data, hypers)
			""".format(str(sanitised_fixed_hypers), experiment_dir+"/data.csv")))

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
					"batch_size": 500,
					"monitoring_batch_size": 500,
					"max_epochs": 10,
					"save_path": ".",
					"yaml_path": "../../../../model_yamls/",
					"layer_fn": "sdae",
					"num_layers": 4,
					"corrupt_l1": 0.33,
					"corrupt_l2": 0.33,
					"corrupt_l3": 0.33,
					"corrupt_l4": 0.33,
					"n_folds": 5,
					"nhid_l4": 12,
					"sparse_coef_l4": 0,
					"sparse_p_l4": 0.1667
	}

#	spearmint_imports =  """\
#							from pylearn2.config import yaml_parse
#						"""

	def alignClusterings(self,model1,model2, some_thresh = 0.1, all_thresh = 0.9):
		if not "clustermodel" in model1.keys() or not "clustermodel" in model2.keys():
			sys.exit("ERROR: Tried to align two models that don't appear to have clusterings attached.")
		model1_alignments = np.zeros((len(model1["clustermeans"]),len(model2["clustermeans"])))
		for i in range(len(model1["clustermeans"])):
			for j in range(len(model2["clustermeans"])):
				#Set alignments[i,j] to the fraction of the probabilities assigned to points in cl1 that are also assigned (at any probability) in cl2.
				#  (Note, here we are using "assigned" to mean "given most likelihood under the probabilistic clustering"
				i_indices = np.nonzero(model1["clusterpreds"]==i)[0]
				i_members = model1["clusterreps"][i_indices,:]
				j_indices = np.nonzero(model2["clusterpreds"]==j)[0]
				j_members = model2["clusterreps"][j_indices,:]
				for k,im in enumerate(i_indices):
					if im in j_indices:
						model1_alignments[i,j] += i_members[k,i]
				model1_alignments[i,j] /= np.sum(i_members[:,i])

		#model1_alignments = model1_alignments[:,np.argmax(model1_alignments,axis=1)]
		pprint(np.around(model1_alignments,decimals=2))

		model2_alignments = np.zeros((len(model2["clustermeans"]),len(model1["clustermeans"])))
		for i in range(len(model2["clustermeans"])):
			for j in range(len(model1["clustermeans"])):
				#Set alignments[i,j] to the fraction of the probabilities assigned to points in cl1 that are also assigned (at any probability) in cl2.
				#  (Note, here we are using "assigned" to mean "given most likelihood under the probabilistic clustering"
				i_indices = np.nonzero(model2["clusterpreds"]==i)[0]
				i_members = model2["clusterreps"][i_indices,:]
				j_indices = np.nonzero(model1["clusterpreds"]==j)[0]
				j_members = model1["clusterreps"][j_indices,:]
				for k,im in enumerate(i_indices):
					if im in j_indices:
						model2_alignments[i,j] += i_members[k,i]
				model2_alignments[i,j] /= np.sum(i_members[:,i])

		#model2_alignments = model2_alignments[:,np.argmax(model2_alignments,axis=1)]
		pprint(np.around(model2_alignments,decimals=2))

		#alignments = np.array([(model1_alignments[i,j],model2_alignments[i,j]) for i in range(len(model1["clustermeans"])) for j in range(len(model2["clustermeans"]))])
		alignments = {}
		for i in range(len(model1["clustermeans"])):
			for j in range(len(model2["clustermeans"])):
				if model1_alignments[i,j]>= all_thresh:
					if model2_alignments[j,i]>= all_thresh:
						alignments[(i,j)] = "C" #Continuation
					elif model2_alignments[j,i] >= some_thresh:
						alignments[(i,j)] = "G" #Generalisation
					else:
						alignments[(i,j)] = "XG" #Extreme Generalisation
				elif model1_alignments[i,j] >= some_thresh:
					if model2_alignments[j,i]>= all_thresh:
						alignments[(i,j)] = "R" #Refinement
					elif model2_alignments[j,i] >= some_thresh:
						alignments[(i,j)] = "E" #Evolution
					else:
						alignments[(i,j)] = "EG" #Evolved Generalisation
				else:
					if model2_alignments[j,i]>= all_thresh:
						alignments[(i,j)] = "XR" #Extreme Refinement
					elif model2_alignments[j,i] >= some_thresh:
						alignments[(i,j)] = "ER" #Evolved Refinement
					else:
						alignments[(i,j)] = "-" #Nothing

		pprint(alignments)

	def compare_models(self, O_by_A, model1, model2, save_path=""):
		ObyA_tsne = np.array(list(bh_tsne.bh_tsne(O_by_A, no_dims=2, perplexity=30)))
		for alpha in [1000]:
			for m,model in enumerate([model1, model2]):
				model["tsne_joint_probs"] = squareform(_joint_probabilities(pairwise_distances(validation.check_array(model["F_by_O_normed"].T, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64), metric='euclidean', n_jobs=1, squared=True),30,2))
				model["clustermodel"],model["clusterreps"],model["clusterpreds"] = self.representation_clustering_VBGMM(model["F_by_O_normed"].T,n_components=6,alpha=alpha,tol=1e-9,n_iter=10000000)
				model["clustermeans"] = []
				model["clusterstdevs"] = []
				model["clustererrors"] = []
				for i,mean in enumerate(model["clustermodel"].means_):
					if i in model["clusterpreds"]:
						model["clustermeans"].append(model_inspector.decode(model["F_by_O_scaler"].inverse_transform(np.atleast_2d(np.float32(mean))),model["model"]).flatten())
						cl_indices = np.nonzero(model["clusterpreds"]==i)[0]
						members = O_by_A.T[:,cl_indices]
						model["clusterstdevs"].append(np.std(members,axis=1))
						members = model["E_by_O"][cl_indices]
						model["clustererrors"].append(np.mean(members))

				print model["clusterreps"][0:10,:]
				#tsne = TSNE(n_components=2, perplexity=30).fit_transform(model["F_by_O_normed"].T)
				model["tsne"] = np.array(list(bh_tsne.bh_tsne(model["F_by_O_normed"].T, no_dims=2, perplexity=30)))
				plt.figure(figsize=(20,20))
				plt.scatter(model["tsne"][:,0],model["tsne"][:,1], c=[float(p) for p in model["clusterpreds"]])
				plt.savefig(os.path.join(save_path,model["name"]+"_FbyO_TSNE.png"), bbox_inches='tight')
				plt.close("all")

				#tsne = TSNE(n_components=2, perplexity=30).fit_transform(O_by_A)
				plt.figure(figsize=(20,20))
				plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=[float(p) for p in model["clusterpreds"]])
				plt.savefig(os.path.join(save_path,model["name"]+"_ObyA_TSNE.png"), bbox_inches='tight')
				plt.close("all")

				'''
				plt.figure(figsize=(20,20))
				df = pd.DataFrame(O_by_A)
				df['class'] = model["clusterpreds"]
				radviz(df,"class",s=1, alpha=0.5)
				plt.savefig(os.path.join(save_path,model["name"]+"attrplot_"+str(alpha)+".png"), bbox_inches='tight')
				plt.close("all")

				plt.figure(figsize=(20,20))
				df = pd.DataFrame(model["F_by_O_normed"].T)
				df['class'] = model["clusterpreds"]
				radviz(df,"class",s=1, alpha=0.5)
				plt.savefig(os.path.join(save_path,model["name"]+"featureplot_"+str(alpha)+".png"), bbox_inches='tight')
				plt.close("all")
				'''


			#pairs = self.alignClusterings(model2,model1) #Swapped these around because alignClustering treats the first model as the past and the second as the current.

			'''
			plt.figure()
			df = pd.DataFrame(model1["clustermeans"]+model2["clustermeans"])
			df['class'] = [0]*len(model1["clustermeans"]) + [1]*len(model2["clustermeans"])
			radviz(df,"class")
			plt.savefig(os.path.join(save_path,model1["name"]+"_vs_"+model2["name"]+"_clustermeans_"+str(alpha)+".png"), bbox_inches='tight')
			plt.close("all")
			'''
			basesize = 10
			modsize = 100

			#probdiffs = np.absolute(model1["tsne_joint_probs"] - model2["tsne_joint_probs"])
			probdiffs = [cosine(u,v) for (u,v) in zip(model1["tsne_joint_probs"], model2["tsne_joint_probs"])]
			probdiffs = sklearn.preprocessing.MinMaxScaler().fit_transform(probdiffs)
			probdiffs = np.sqrt(probdiffs)
			cosdiffs = [cosine(u,v) for (u,v) in zip(model1["F_by_O_normed"].T, model2["F_by_O_normed"].T)]
			cosdiffs = sklearn.preprocessing.MinMaxScaler().fit_transform(cosdiffs)
			cosdiffs = np.sqrt(cosdiffs)

			for model in [model1,model2]:
				plt.figure(figsize=(20,20))
				df = pd.DataFrame(O_by_A)
				df['class'] = model["clusterpreds"]
				radviz(df,"class",s=[(basesize+(modsize*p))/10. for p in probdiffs], alpha=0.5)
				plt.savefig(os.path.join(save_path,model["name"]+"attrplot_probdiffs_"+str(alpha)+".png"), bbox_inches='tight')
				plt.close("all")
				plt.figure(figsize=(20,20))
				radviz(df,"class",s=[(basesize+(modsize*p))/10. for p in cosdiffs], alpha=0.5)
				plt.savefig(os.path.join(save_path,model["name"]+"attrplot_cosdiffs_"+str(alpha)+".png"), bbox_inches='tight')
				plt.close("all")

				plt.figure(figsize=(20,20))
				df = pd.DataFrame(model["F_by_O_normed"].T)
				df['class'] = model["clusterpreds"]
				radviz(df,"class",s=[(basesize+(modsize*p))/10. for p in probdiffs], alpha=0.5)
				plt.savefig(os.path.join(save_path,model["name"]+"featureplot_probdiffs_"+str(alpha)+".png"), bbox_inches='tight')
				plt.close("all")
				plt.figure(figsize=(20,20))
				radviz(df,"class",s=[(basesize+(modsize*p))/10. for p in cosdiffs], alpha=0.5)
				plt.savefig(os.path.join(save_path,model["name"]+"featureplot_cosdiffs_"+str(alpha)+".png"), bbox_inches='tight')
				plt.close("all")

			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=probdiffs, cmap="copper", s=[basesize+(modsize*p) for p in probdiffs])
 			plt.savefig(os.path.join(save_path,"ObyA_TSNE_probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(model1["tsne"][:,0],model1["tsne"][:,1], c=probdiffs, cmap="copper", s=[basesize+(modsize*p) for p in probdiffs])
			plt.savefig(os.path.join(save_path,model1["name"]+"_FbyO_TSNE_probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(model2["tsne"][:,0],model2["tsne"][:,1], c=probdiffs, cmap="copper", s=[basesize+(modsize*p) for p in probdiffs])
			plt.savefig(os.path.join(save_path,model2["name"]+"_FbyO_TSNE_probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=cosdiffs, cmap="copper", s=[basesize+(modsize*p) for p in cosdiffs])
 			plt.savefig(os.path.join(save_path,"ObyA_TSNE_cosdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(model1["tsne"][:,0],model1["tsne"][:,1], c=cosdiffs, cmap="copper", s=[basesize+(modsize*p) for p in cosdiffs])
			plt.savefig(os.path.join(save_path,model1["name"]+"_FbyO_TSNE_cosdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(model2["tsne"][:,0],model2["tsne"][:,1], c=cosdiffs, cmap="copper", s=[basesize+(modsize*p) for p in cosdiffs])
			plt.savefig(os.path.join(save_path,model2["name"]+"_FbyO_TSNE_cosdiffs.png"), bbox_inches='tight')
			plt.close("all")

		#sys.exit()
		'''
			for k in Ks:
				if k == float("inf"):
					print "*** k =",k
					for i in range(3):
						clusters,reps = self.representation_clustering_DPGMM(model["F_by_O_normed"].T,n_components=10,alpha=1e-2,tol=1e-6)

					return
				else:
					U, s, V = scipy.sparse.linalg.svds(model["F_by_O_normed"].T,k=k)
					s = np.diagflat(s)

					# Generate US or SV products for generating reduced dimensional vectors
					# for nodes. This is the same if the matrix is symmetric and square.
					#u_red = svdu*svds
					u_red = np.matrix(U)*np.matrix(s)
					print "   --- Dimensionality reduction completed."

					print u_red.shape
					print "*** k =",k
					for i in range(2):
						dpgmm_svd = sklearn.mixture.DPGMM(n_components=20,alpha=0.0001,covariance_type="diag",n_iter=1000, tol=1e-6).fit(u_red)
						Z = dpgmm_svd.predict_proba(u_red)
						print "model learnt: ",
						out = np.around(np.sort(np.sum(Z,axis=0))[::-1], decimals=0)
						print out[out>0]


					# Compute corrcoef measurements between all US combinations. Produce the
					# correlation matrix in reduced k-space. Z_u will show communities for only Type
					# 1 nodes (rows of the original matrix).
					Z_cosines = sklearn.metrics.pairwise.pairwise_distances(u_red,metric="cosine",n_jobs=1)
					Z_corrcoefs = np.corrcoef(u_red) / 2 + 0.5
					print "   --- Correlation matrix constucted."

					#scipy.misc.imsave(os.path.join(save_path,model["name"]+"_correlations_k="+str(k)+".png"),Z_u)

					for i in range(1):
						model["name"] = model["name"]+"_"+str(i)
						Z_corrcoefs_labels = sklearn.cluster.MiniBatchKMeans(n_clusters=k, n_init=20).fit_predict(Z_corrcoefs)
						Z_cosines_labels = sklearn.cluster.MiniBatchKMeans(n_clusters=k, n_init=20).fit_predict(Z_cosines)
						print "   --- Clusters generated."

						corr_clustermeans = []
						corr_clusterstdevs = []
						corr_clustercoefs = np.zeros((k,k))
						for cl1 in range(k):
							cl1_indices = np.nonzero(Z_corrcoefs_labels==cl1)[0]
							members = O_by_A.T[:,cl1_indices]
							corr_clustermeans.append(np.mean(members,axis=1))
							corr_clusterstdevs.append(np.std(members,axis=1))
							#''#ORDER BY RCM OF CLUSTER COEF MATRIX
							for cl2 in range(k):
								if cl1==cl2:
									clustercoefs[cl1,cl2] = 1
								else:
									cl2_indices = np.nonzero(Z_corrcoefs_labels==cl2)[0]
									coefs = Z_corrcoefs[zip(*itertools.product(cl1_indices,cl2_indices))]
									clustercoefs[cl1,cl2] = np.mean(coefs)
							#''
						model["corr_clustermeans_"+str(k)] = corr_clustermeans

						#ORDER BY SORTED CLUSTER MEANS
						corr_cluster_mean_sort = np.lexsort(np.array(corr_clustermeans).T)
						Z_corrcoefs_labels_sorted = np.zeros(Z_corrcoefs_labels.shape)
						for key,index in enumerate(corr_cluster_mean_sort):
							Z_corrcoefs_labels_sorted[np.nonzero(Z_corrcoefs_labels==index)] += key
						Z_corrcoefs_labels = Z_corrcoefs_labels_sorted
						#''

						cos_clustermeans = []
						cos_clusterstdevs = []
						cos_clustercoefs = np.zeros((k,k))
						for cl1 in range(k):
							cl1_indices = np.nonzero(Z_cosines_labels==cl1)[0]
							members = O_by_A.T[:,cl1_indices]
							cos_clustermeans.append(np.mean(members,axis=1))
							cos_clusterstdevs.append(np.std(members,axis=1))
							#''#ORDER BY RCM OF CLUSTER COEF MATRIX
							for cl2 in range(k):
								if cl1==cl2:
									clustercoefs[cl1,cl2] = 1
								else:
									cl2_indices = np.nonzero(Z_corrcoefs_labels==cl2)[0]
									coefs = Z_corrcoefs[zip(*itertools.product(cl1_indices,cl2_indices))]
									clustercoefs[cl1,cl2] = np.mean(coefs)
							#''
						model["cos_clustermeans_"+str(k)] = cos_clustermeans

						#ORDER BY SORTED CLUSTER MEANS
						cos_cluster_mean_sort = np.lexsort(np.array(cos_clustermeans).T)
						Z_cosines_labels_sorted = np.zeros(Z_cosines_labels.shape)
						for key,index in enumerate(cos_cluster_mean_sort):
							Z_cosines_labels_sorted[np.nonzero(Z_cosines_labels==index)] += key
						Z_cosines_labels = Z_cosines_labels_sorted
						#''

						print "   --- Cluster orderings calculated."

						#Replace all the labels in Z_corrcoefs_labels with those from the new ordering
						corr_clustersort = np.argsort(Z_corrcoefs_labels)
						Z_corrcoefs_clst = Z_corrcoefs[corr_clustersort,:]
						Z_corrcoefs_clst = Z_corrcoefs_clst[:,corr_clustersort]

						scipy.misc.imsave(os.path.join(save_path,model["name"]+"_clustered_correlations_k="+str(k)+".png"),Z_corrcoefs_clst)

						#Replace all the labels in Z_cosines_labels with those from the new ordering
						cos_clustersort = np.argsort(Z_cosines_labels)
						Z_cosines_clst = Z_cosines[cos_clustersort,:]
						Z_cosines_clst = Z_cosines_clst[:,cos_clustersort]

						scipy.misc.imsave(os.path.join(save_path,model["name"]+"_clustered_cosines_k="+str(k)+".png"),Z_cosines_clst)
						model["name"] = model["name"][:-2]
						#'''

		'''
		import pandas as pd
		from pandas.tools.plotting import radviz
		import matplotlib.pyplot as plt
		plt.style.use('ggplot')
		for k in Ks:
			plt.figure()
			df = pd.DataFrame(model1["corr_clustermeans_"+str(k)]+model2["corr_clustermeans_"+str(k)])
			df['class'] = [0]*len(model1["corr_clustermeans_"+str(k)]) + [1]*len(model2["corr_clustermeans_"+str(k)])
			radviz(df,"class")
			plt.savefig(os.path.join(save_path,model["name"]+"_corr_clustermeans_"+str(k)+".png"), bbox_inches='tight')

			plt.figure()
			df = pd.DataFrame(model1["cos_clustermeans_"+str(k)]+model2["cos_clustermeans_"+str(k)])
			df['class'] = [0]*len(model1["cos_clustermeans_"+str(k)]) + [1]*len(model2["cos_clustermeans_"+str(k)])
			radviz(df,"class")
			plt.savefig(os.path.join(save_path,model["name"]+"_cos_clustermeans_"+str(k)+".png"), bbox_inches='tight')
			plt.close("all")
		'''

	def representation_clustering_VBGMM(self,model,alpha=1e-3,tol=1e-5,n_components=20,n_iter=1000, verbose=False):
		vbgmm = sklearn.mixture.VBGMM(n_components=n_components,alpha=alpha,covariance_type="diag",n_iter=n_iter, tol=tol, verbose=verbose).fit(model)
		print "model learnt: ",
		Z = vbgmm.predict_proba(model)
		out = np.around(np.sort(np.sum(Z,axis=0))[::-1], decimals=0)
		print out[out>0]
		return (vbgmm,Z,vbgmm.predict(model))

	def representation_clustering_DPGMM(self,model,alpha=1e-3,tol=1e-5,n_components=20,n_iter=1000, verbose=False):
		dpgmm = sklearn.mixture.DPGMM(n_components=n_components,alpha=alpha,covariance_type="diag",n_iter=n_iter, tol=tol, verbose=verbose).fit(model)
		print "model learnt: ",
		Z = dpgmm.predict_proba(model)
		out = np.around(np.sort(np.sum(Z,axis=0))[::-1], decimals=0)
		print out[out>0]
		return (dpgmm,Z,dpgmm.predict(model))


	def inspect(self,model,save_path="",cluster_orders = {}):

		#Grab the top-most layer of the saved model
		topmodel = model_inspector.load_model(model["model_files"],model['models'][-1].algorithm.cost)
		F_by_A = model_inspector.show_weights(topmodel,save_patches=False)
		print F_by_A.shape
		scipy.misc.imsave("F_by_A.png",F_by_A)
		F_by_O = model_inspector.encode(model["training_data"].X,topmodel).T
		scipy.misc.imsave("F_by_O.png",F_by_O)

		print F_by_O.shape
		F_by_O = F_by_O[:,np.random.choice(F_by_O.shape[1],size=1000)]
		print F_by_O.shape

		print model_inspector.deep_recon_errors(F_by_O,topmodel)
		errors = np.zeros((F_by_O.shape[0],len(topmodel['layers'])))
		Os = []
		O_recons = []

		for i,x in enumerate(F_by_O):
			Os.append(x)
			errors[i] = model_inspector.deep_recon_errors(x,topmodel)[-1]
			O_recons.append(model_inspector.reconstruct(topmodel,x))
		print "Data extraction complete"
		#F_by_F = np.corrcoef(F_by_O)
		#O_by_O = np.corrcoef(F_by_O.T)
		#scipy.misc.imsave("O_by_O.png",O_by_O)

		#print "Corrcoefs complete"
		#SVD Calcs for O by O, comments taken from Som's community detection code.
		#k = 13
		for k in [3,6,9,12]:
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

				###Reorder the clusters using RCM to reveal between-cluster structure.
				clustermeans = []
				clusterstdevs = []
				clustercoefs = np.zeros((k,k))
				for cl1 in range(k):
					cl1_indices = np.nonzero(Z_u_labels==cl1)[0]
					members = F_by_O[:,cl1_indices]
					clustermeans.append(np.mean(members,axis=0))
					clusterstdevs.append(np.std(members,axis=0))
					for cl2 in range(k):
						if cl1==cl2:
							clustercoefs[cl1,cl2] = 1
						else:
							cl2_indices = np.nonzero(Z_u_labels==cl2)[0]
							coefs = Z_u[zip(*itertools.product(cl1_indices,cl2_indices))]
							clustercoefs[cl1,cl2] = np.mean(coefs)
				print clustermeans
				print clusterstdevs
				#RCM that matrix to get the new cluster ordering.
				if k in cluster_orders.keys():
					clustercoefs_rcm = cluster_orders[k]
				else:
					clustercoefs_rcm = qutip.reverse_cuthill_mckee(scipy.sparse.csr_matrix(sklearn.preprocessing.binarize(clustercoefs,t)))
				Z_u_labels_rcm = np.zeros(Z_u_labels.shape)
				for key,index in enumerate(clustercoefs_rcm):
					Z_u_labels_rcm[np.nonzero(Z_u_labels==index)] += key
				scipy.misc.imsave(save_path+"Z_u_clustercoefs_"+str(k)+".png",clustercoefs)
				Z_u_labels = Z_u_labels_rcm
				#Replace all the labels in Z_u_labels with those from the new ordering

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

				#Z_u_labels.shape = (Z_u_labels.shape[0],1)
				#Z_u_overlaid = cm(Z_u_labels + Z_u)
				#scipy.misc.imsave(save_path+"Z_u_clusters"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters, bytes=True))
				#pprint.pprint(Z_u_overlaid)
				#scipy.misc.imsave("Z_u_clusters_"+str(k)+".png",cm(Z_u_clusters, bytes=True))
				#scipy.misc.imsave("Z_u_nocorr_"+str(k)+".png",Z_u)
				scipy.misc.imsave(save_path+"Z_u_"+str(k)+".png",Z_u)
				#Z_u_bin = sklearn.preprocessing.binarize(Z_u,t)
				#scipy.misc.imsave("Z_u_bin_nocorr_"+str(k)+"_"+str(t)+".png",Z_u_bin)
				#scipy.misc.imsave(save_path+"Z_u_bin_"+str(k)+"_"+str(t)+".png",Z_u_bin)

				#rcm_order = qutip.reverse_cuthill_mckee(scipy.sparse.csr_matrix(1 - Z_u_bin))

				#Z_u_rcm = Z_u[rcm_order,:]
				#Z_u_rcm = Z_u_rcm[:,rcm_order]
				Z_u_clst = Z_u[clustersort,:]
				Z_u_clst = Z_u_clst[:,clustersort]
				#Z_u_clusters_rcm = Z_u_clusters[rcm_order,:]
				#Z_u_clusters_rcm = Z_u_clusters_rcm[:,rcm_order]
				Z_u_clusters_clst = Z_u_clusters[clustersort,:]
				Z_u_clusters_clst = Z_u_clusters_clst[:,clustersort]
				#scipy.misc.imsave(save_path+"Z_u_rcm_"+str(k)+"_"+str(t)+".png",Z_u_rcm)
				scipy.misc.imsave(save_path+"Z_u_clst_"+str(k)+"_"+str(t)+".png",Z_u_clst)
				#scipy.misc.imsave(save_path+"Z_u_clusters_rcm_"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters_rcm, bytes=True))
				scipy.misc.imsave(save_path+"Z_u_clusters_clst_"+str(k)+"_"+str(t)+".png",cm(Z_u_clusters_clst, bytes=True))
				print "Reorder and image saving complete"


	def run(self,data, hypers, logging=False, pretrained = [], cv = True):
		from pylearn2.config import yaml_parse
		print hypers

		objectives = []
		objectives_by_layer = []
		model_files = []
		models = []
		yaml_ext = ".yaml"
		if not cv:
			yaml_ext = "_nocv"+yaml_ext
		for layer_num in range(1,hypers['num_layers']+1):
			layer_yaml = open(hypers["yaml_path"]+hypers["layer_fn"]+"_l"+str(layer_num)+yaml_ext, 'r').read()
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
				train.model.monitor = Monitor(train.model)
			train.main_loop()

			if 'model' in dir(train):
				obj = train.model.monitor.channels['objective'].val_record[-1] #This is a single Train object with no k-fold CV happening.
				objectives_by_layer.append([float(i) for i in train.model.monitor.channels['objective'].val_record])
			else:
				obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers]) #This is a TrainCV object that's doing k-fold CV.
				objectives_by_layer.append([float(j) for j in np.mean([i.model.monitor.channels['test_objective'].val_record for i in train.trainers])])
			print obj
			objectives.append(obj)
			models.append(train)
			model_files.append(layer_hypers['save_path']+hypers["layer_fn"]+"_l"+str(layer_num)+".pkl")

		objective = float(np.sum(objectives))
		print objectives_by_layer
		if logging:
			return {"objective": objective,
					"training_objectives": objectives_by_layer,
					"model_files": model_files}
					#"models": models,
					#"training_data": data}
		else:
			return objective


	def __init__(self, domain_name, scratch_path = "", selected_hypers={}):
		from pylearn2.config import yaml_parse
		self.fixed_hypers["costs"] = yaml_parse.load('cost : !obj:pylearn2.costs.cost.SumOfCosts {costs: [!obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {}, !obj:pylearn2.costs.autoencoder.SparseActivation {coeff: 1,p: 0.15}]}')['cost']
		ConceptualSpace.__init__(self, domain_name, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)

def LSTMConceptualSpace(ConceptualSpace):
	fixed_hypers = {}
	hyper_space = {}

	def __init__(self, domain_name, scratch_path = "", selected_hypers={}):
		ConceptualSpace.__init__(self, domain_name, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)


if __name__ == "__main__":
	cs = SDAConceptualSpace("data/")
	cs.train({}, threshold = 0.01, look_back = 5)
	#from sdae_l3_reconstruct import reconstruct
	#reconstruct("data/"+cs.exp_path+"/dae")
	#from sdae_l3_show_weights_decoder import show_weights
	#show_weights("data/"+cs.exp_path+"/dae")
