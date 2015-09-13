try: import simplejson as json
except ImportError: import json
from sklearn.preprocessing import StandardScaler,Imputer
#from sklearn.manifold import TSNE
import AnimTSNE_experiment
import monary
from copy import deepcopy

import os, os.path, pprint, textwrap, csv, importlib, sys, optparse , time, inspect, DUconfig, model_inspector, scipy.misc, scipy.spatial, sklearn.mixture, matplotlib, qutip, itertools, copy, sklearn.preprocessing, pymc

from collections import OrderedDict, deque, namedtuple

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
import scipy.stats
import theano.tensor as T
import theano
import heapq
import pymongo

import pandas as pd
from pandas.tools.plotting import radviz
import matplotlib.pyplot as plt

import bh_tsne.bhtsne as bh_tsne

plt.style.use('ggplot')
np.set_printoptions(linewidth=200)


from networkx.utils import reverse_cuthill_mckee_ordering

from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import squareform
from sklearn.manifold import _utils
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import validation


MACHINE_EPSILON = np.finfo(np.double).eps

iterables = (types.DictType, types.ListType, types.TupleType, types.GeneratorType)
keepables = (types.TypeType, types.BooleanType, types.IntType, types.LongType, types.FloatType, types.ComplexType, types.StringType, types.UnicodeType, types.NoneType)
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

# Function for calculating the deep reconstruction error -- AE cost only does per-layer error.  This should probably go somewhere else.
def deep_recon(data, encoders, decoders, cost, batch_size, test_indices=[]):
	results = []
	if len(test_indices):
		for fold in range(len(encoders)):
			test_data = data[test_indices[fold]]
			batch_results = []
			for batch in np.array_split(test_data,max(1,test_data.shape[0]/batch_size)):
				d = np.atleast_2d(batch)
				d_prime = d
				for l in range(fold):
					d_prime = encoders[l][fold](d_prime)
				for l in reversed(range(fold)):
					d_prime = decoders[l][fold](d_prime)
				batch_results.append(cost[fold](d,d_prime))
			results.append(np.mean(np.vstack(batch_results)))
		results = np.mean(np.vstack(results))
	else:
		for batch in np.array_split(data,max(1,data.shape[0]/batch_size)):
			d = np.atleast_2d(batch)
			d_prime = d
			for l in range(len(encoders)):
				d_prime = encoders[l](d_prime)
			for l in reversed(range(len(encoders))):
				d_prime = decoders[l](d_prime)
			results.append(cost(d,d_prime))
		results = np.mean(np.vstack(results))
	return float(results) # We have to cast this to a non-numpy float for some reason to do with mongodb -- I suspect BSON is at fault, but no matter.

def monary_load(collection, fields_x, fields_y, start=0, stop=0, find_args={}, return_timefield=False,timefield="timefield", type="float32", split = None, shuffle_split=True):
	print "Loading from MongoDB via monary with:"
	print "  - collection:",collection
	print "  - fields_x:",fields_x
	print "  - fields_y:",fields_y,
	print "  - start:",start
	print "  - stop:",stop
	print "  - find_args:",find_args

	monary.monary.MAX_COLUMNS = 4096
	if return_timefield:
		fields_x = deepcopy(fields_x)
		fields_x.append(timefield)
	numfields = len(fields_x)+len(fields_y)
	monary_type = type
	if type == "exists":
		monary_type = "type"
	with monary.Monary("127.0.0.1") as monaryclient:
		out = monaryclient.query(
			"creeval",
			collection,
			find_args,
			fields_x+fields_y,
			[monary_type] * numfields,
			limit=(stop-start),
			offset=start
		)
	if type=="exists":
		for i,col in enumerate(out[0:len(fields_x)]):
			out[i] = np.ma.filled(col,0)
			#if any(np.isnan(col)):
		out = np.ma.row_stack(out).T
		X = out[:,0:len(fields_x)]
		X = (X>0).astype(int)
		if return_timefield:
			timefields = X[:,-1]
			X = X.astype(bool).astype(int)
		y = out[:,len(fields_x):]
		y = (y > 0).astype(int)
		y = np.asarray(y)
	else:
		for i,col in enumerate(out[0:len(fields_x)]):
			out[i] = np.ma.filled(col,np.ma.mean(col))
			#if any(np.isnan(col)):
		out = np.ma.row_stack(out).T
		X = out[:,0:len(fields_x)]
		if return_timefield:
			timefields = X[:,-1]
			X = X[:,0:-1]
		y = out[:,len(fields_x):]
		y = (y > 0).astype(int)
		y = np.asarray(y)

		if X.shape[0]:
			scaler = StandardScaler().fit(X)
			X = scaler.transform(X)
			pickle.dump(scaler,open(collection+"_scaler.pkl","wb"))

	print "Retrieved and scaled",X.shape[0],"datapoints."

	if split is not None:
		if shuffle_split:
			p = np.random.permutation(len(X))
			X = X[p]
			y = y[p]
		split_ind = int(len(X) * split)
		X_train = X[:split_ind,:]
		X_test = X[split_ind:,:]
		y_train = y[:split_ind,:]
		y_test = y[split_ind:,:]
		if return_timefield:
			return DenseDesignMatrix(X=X_train,y=y_train),DenseDesignMatrix(X=X_test,y=y_test), timefields
		return DenseDesignMatrix(X=X_train,y=y_train),DenseDesignMatrix(X=X_test,y=y_test)
	else:
		if return_timefield:
			return DenseDesignMatrix(X=X,y=y), timefields
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
		self.short_domain_name = domain_name.split("_")[0]
		self.hyper_space = hyper_space
		self.fixed_hypers = fixed_hypers
		self.scratch_path = scratch_path
		self.hypers = copy.deepcopy(selected_hypers)
		self.fixed_hypers["layer_fn"] = self.short_domain_name+"_"+self.fixed_hypers["layer_fn"]

	def pretrain(self,metadata, override_query,sample_limit=0):
		self.metadata = metadata
		q = metadata["query"]
		if len(override_query.keys()):
			q = override_query
		pretrain_query = {'$and': [deepcopy(q)]}
		if "pretrain_start" in metadata.keys():
			pretrain_query['$and'].append({metadata["timefield"]: {"$gte": metadata["pretrain_start"]}})
			pretrain_query['$and'].append({metadata["timefield"]: {"$lt": metadata["pretrain_stop"]}})
		else:
			metadata["pretrain_start"] = None
			metadata["pretrain_stop"] = None
		train_ddm = None
		test_ddm = None
		if q is not None:
			train_ddm, test_ddm = monary_load(self.domain_name, metadata["fields_x"],metadata["fields_y"],find_args=pretrain_query,stop=sample_limit, split = 0.9, shuffle_split=True, type=self.fixed_hypers["monary_type"])
			if not train_ddm.X.shape[0]:
				sys.exit("Pretrain failed as no examples were found within the pretraining time window of "+str(metadata["pretrain_start"])+" to "+str(metadata["pretrain_stop"]))
			print "---------SAMPLES FROM DATA:---------"
			self.data_samples = []
			for i in range(100):
				samp = train_ddm.X[np.random.randint(train_ddm.X.shape[0]),:]
				self.data_samples.append(samp)
				print samp
		#scaler = pickle.load(open(domain_name+"_scaler.pkl","rb"))
		DUconfig.dataset = train_ddm
		DUconfig.test_dataset = test_ddm
		params = self.hypers
		params.update(self.fixed_hypers)
		if q is not None:
			params["train_stop"] = train_ddm.X.shape[0]
		params['save_path'] = os.path.join(self.scratch_path,"step_0")
		params['yaml_path'] = params['yaml_path'].lstrip("./") #The existing yaml_path starts with "../../" in order to get out of the spearmint dir.
		result = self.run(test_ddm,params,logging=True, cv=False)
		if metadata["pretrain_start"] is not None:
			result["start"] = metadata["pretrain_start"]
			result["stop"] =  metadata["pretrain_stop"]
		#pprint(result)
		return result

	def stepwise_train(self,metadata, override_query,sample_limit=0):
		#metadata["time_slice"] = 1 # Debug measure to speed shit up.
		start_time = metadata["pretrain_stop"]
		stop_time = metadata["pretrain_stop"] + metadata["time_slice"]

		if len(override_query.keys()):
			q = override_query
		else:
			q = deepcopy(metadata["query"])
		while stop_time < metadata["train_stop"]:
			#query
			step_query = {'$and': [q]}
			step_query['$and'].append({metadata["timefield"]: {"$gte": start_time}})
			step_query['$and'].append({metadata["timefield"]: {"$lt": stop_time}})

			#train
			ddm = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=step_query,stop=sample_limit)
			if ddm.X.shape[0]:
				DUconfig.dataset = ddm
				params = self.hypers
				params.update(self.fixed_hypers)
				params['save_path'] = os.path.join(self.scratch_path,"step_"+str(len(metadata["steps"])))
				params['yaml_path'] = params['yaml_path'].lstrip("./") #The existing yaml_path starts with "../../" in order to get out of the spearmint dir.
				params["train_stop"] = ddm.X.shape[0]
				pretrain_prefix = os.path.join(self.scratch_path,"step_"+str(len(metadata["steps"])-1))
				pretrain_paths = [os.path.join(pretrain_prefix,params["layer_fn"]+"_l"+str(i+1)+".pkl") for i in range(params["num_layers"])]
				result = self.run(ddm, params,logging=True, pretrained = pretrain_paths, cv=False)
			else:
				result = {"error": "NO DATA FOUND FOR THIS TIMESTEP."}
			result["start"] = start_time
			result["stop"] =  stop_time
			metadata["steps"].append(result)
			print metadata["steps"]
			start_time = stop_time
			stop_time += metadata["time_slice"]

	def stepwise_inspect(self,metadata,override_query, sample_size=50000,save_path="", resample=1, start_step=0):
		start_time = metadata["pretrain_start"]
		if len(override_query.keys()):
			q = override_query
		else:
			q = deepcopy(metadata["query"])
		print "Calculating unexpectedness of each step."
		for i,step in enumerate(metadata["steps"][start_step+1:]):
			#start_time = step["start"]
			stop_time = step["stop"]
			print "Inspecting step "+str(i+start_step)+". "+str(start_time)+"--"+str(stop_time)+"."
			if "error" not in step.keys():
				#query
				step_query = {'$and': [q]}
				step_query['$and'].append({metadata["timefield"]: {"$gte": start_time}})
				#step_query['$and'].append({metadata["timefield"]: {"$gte": metadata["pretrain_start"]}})
				step_query['$and'].append({metadata["timefield"]: {"$lt": stop_time}})
				print "   --- Query built."

				#sample
				data, times = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=step_query, return_timefield=True,timefield=metadata["timefield"])
				for k in range(resample): #Testing multiple clusterings of each step with different samplings.
					if data.X.shape[0] > sample_size:
						indices = np.random.choice(data.X.shape[0],size=sample_size)
						times_sample = times[indices]
						data_sample = data.X[indices,:]
					else:
						data_sample = data.X
						times_sample = times
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
					model1["train_start"] = step["start"]
					model1["train_stop"] = step["stop"]
					model1["name"] = "current_"+str(k)
					print "   --- Model 1 load complete."

					prev = i+start_step # i is actually already the index of the previous model, since we're iterating from element 1 onwards.
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
					model2["train_start"] = metadata['steps'][prev]["start"]
					model2["train_stop"] = metadata['steps'][prev]["stop"]
					model2["name"] = "previous_"+str(k)
					print "   --- Model 2 load complete"

					self.compare_models(data_sample, model1, model2, times_sample, save_path=os.path.dirname(step["model_files"][0]))
					del model1
					del model2
					del data_sample
					del times_sample
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
	def compare_models(self, O_by_A, model1, model2, times, save_path=""):
		raise NotImplementedError

	def train_mongo(self, fields_x, fields_y, query, threshold = 0.01, look_back = 3, sample_limit = 0):
		name = self.__class__.__name__+str(id(self))
		expdir = os.path.abspath(os.path.join(self.scratch_path,name))+"/"
		if not os.path.exists(expdir):

			os.makedirs(expdir)
		self.fixed_hypers["_mongo"] = {"collection": self.domain_name, "fields_x": fields_x, "fields_y": fields_y, "query": query}
		self._train(threshold,look_back, sample_limit=sample_limit)
		return self.hypers


	def train_csv(self, data, threshold = 0.01, look_back = 3):
		name = self.__class__.__name__+str(id(self))
		expdir = os.path.abspath(os.path.join(self.scratch_path,name))+"/"
		if not os.path.exists(expdir):
			os.makedirs(expdir)
		self.write_data(data, expdir)
		self._train(threshold,look_back)

	def _train(self, threshold, look_back, sample_limit = 0):
		name = self.__class__.__name__+str(id(self))
		self.gen_spearmint_template(self.hyper_space, name)
		#self.gen_model_script(textwrap.dedent(self.spearmint_imports), textwrap.dedent(self.spearmint_run), name)
		run_source = inspect.getsourcelines(self.run)[0]
		run_source[0] = run_source[0].replace("self,","") #The written copy of the method is in a class, but the spearmint version is static.
		self.gen_model_script(textwrap.dedent("".join(run_source)), name, spearmintImports=self.spearmint_imports, sample_limit = sample_limit)
		self.run_spearmint(name, threshold=threshold, look_back=look_back)

	def write_data(self,data, name):
		with open(name+"/data.csv", "wb") as csvf:
			writer = csv.writer(csvf)
			writer.writerows(data)

		# Tuples indicate min and max values for real-valued params, lists indicate possible values -- lists of strings for categorical, lists of ints for ordinal
	def gen_spearmint_template(self, params, fname):
		data = {"language": "PYTHON", "main-file":fname+".py", "experiment-name": fname, "likelihood": "GAUSSIAN", "resources": {"my-machine": {"scheduler":"local","max-concurrent":1}}}
		paramdict = {}
		for k,v in params.iteritems():
			var = {}
			if type(v) is list:
				if type(v[0]) is int:
					if len(v) > 2:
						var["type"] = "ENUM"
						var["size"] = 1
						var["options"] = v
					else:
						var["type"] = "INT"
						var["size"] = 1
						var["min"] = v[0]
						var["max"] = v[1]
				else:
					var["type"] = "ENUM"
					var["size"] = 1
					var["options"] = v
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

	def gen_model_script(self, spearmintRun, fname, spearmintImports = "", sample_limit = 0):
		experiment_dir = os.path.abspath(os.path.join(self.scratch_path,fname))
		with open(experiment_dir+"/"+fname+".py", "w") as f:
			sanitised_fixed_hypers = deepcopy(self.fixed_hypers) # This gets rid of objects that seem to get left in the hypers -- particularly a Costs object.
			sanitise_for_str_out(sanitised_fixed_hypers)
			f.write(spearmintImports+"\nimport numpy as np\nimport sys\nsys.path.append('..')\nfrom conceptual_space import monary_load,deep_recon\nimport DUconfig\nfrom monary import Monary\n\n")
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
					elif fixed_params['_mongo']['query'] == None:
						data = None
						print "Bypassing mongo -- expecting dataset in YAML."
					else:
						if 'monary_type' in fixed_params.keys():
							data = monary_load(fixed_params['_mongo']['collection'], fixed_params['_mongo']['fields_x'], fixed_params['_mongo']['fields_y'], find_args=fixed_params['_mongo']['query'], stop={2}, type=fixed_params['monary_type'])
						else:
							data = monary_load(fixed_params['_mongo']['collection'], fixed_params['_mongo']['fields_x'], fixed_params['_mongo']['fields_y'], find_args=fixed_params['_mongo']['query'], stop={2})
						DUconfig.dataset = data
						print "---------SAMPLES FROM DATA:---------"
						for i in range(100):
							print data.X[np.random.randint(data.X.shape[0]),:]
						fixed_params["train_stop"] = data.X.shape[0]
						#fixed_params["yaml_path"] = fixed_params["yaml_path"] + fixed_params['_mongo']['collection'] + "_"
					hypers = fixed_params
					hypers.update(asciiparams)
					return run(data, hypers)
			""".format(str(sanitised_fixed_hypers), experiment_dir+"/data.csv",sample_limit)))

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
						print performance,'(',look_back,')'
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

	def compare_models(self, O_by_A, model1, model2, times, save_path=""):
		ObyA_tsne = np.array(list(bh_tsne.bh_tsne(O_by_A, no_dims=2, perplexity=30)))
		for alpha in [1000]:
			for m,model in enumerate([model1, model2]):
				model["tsne_joint_probs"] = squareform(_joint_probabilities(pairwise_distances(validation.check_array(model["F_by_O_normed"].T, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64), metric='euclidean', n_jobs=1, squared=True),30,2))
				#model["tsne"] = np.array(list(bh_tsne.bh_tsne(model["F_by_O_normed"].T, no_dims=2, perplexity=30)))
				model["tsne"] = AnimTSNE_experiment.TSNE(n_components=2, perplexity=30,n_iter=200, save_path=os.path.join(save_path,"AnimTSNE",model["name"]+"/"))
				model["tsne"].fit_transform(model["F_by_O_normed"].T, c=[0]*O_by_A.shape[0])
				'''
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
			#probdiffs = [cosine(u,v) for (u,v) in zip(model1["tsne_joint_probs"], model2["tsne_joint_probs"])]
			probdiffs = [scipy.stats.entropy(u,v) for (u,v) in zip(model1["tsne_joint_probs"], model2["tsne_joint_probs"])]
			probdiffs = sklearn.preprocessing.MinMaxScaler().fit_transform(np.log(probdiffs))
			sq_probdiffs = np.sqrt(probdiffs)
			m1_cosmat = sklearn.metrics.pairwise.pairwise_distances(model1["F_by_O"].T,metric="cosine")
			m2_cosmat = sklearn.metrics.pairwise.pairwise_distances(model2["F_by_O"].T,metric="cosine")
			cosdiffs = [cosine(u,v) for (u,v) in zip(m1_cosmat, m2_cosmat)]
			cosdiffs = sklearn.preprocessing.MinMaxScaler().fit_transform(cosdiffs)
			sq_cosdiffs = np.sqrt(cosdiffs)
			euc_cosdiffs = [euclidean(u,v) for (u,v) in zip(m1_cosmat, m2_cosmat)]
			euc_cosdiffs = sklearn.preprocessing.MinMaxScaler().fit_transform(euc_cosdiffs)
			m1_eucmat = sklearn.metrics.pairwise.pairwise_distances(model1["F_by_O"].T,metric="euclidean")
			m2_eucmat = sklearn.metrics.pairwise.pairwise_distances(model2["F_by_O"].T,metric="euclidean")
			eucdiffs = [euclidean(u,v) for (u,v) in zip(m1_eucmat, m2_eucmat)]
			eucdiffs = sklearn.preprocessing.MinMaxScaler().fit_transform(eucdiffs)
			#cosdiffs = [cosine(u,v) for (u,v) in zip(model1["F_by_O"].T, model2["F_by_O"].T)]
			m1_errors = [deep_recon(d,model1["model"]["encoders"],model1["model"]["decoders"],model1["model"]["comparative_costs"][0],self.fixed_hypers["batch_size"]) for d in O_by_A]
			m2_errors = [deep_recon(d,model2["model"]["encoders"],model2["model"]["decoders"],model2["model"]["comparative_costs"][0],self.fixed_hypers["batch_size"]) for d in O_by_A]
			dErrors = np.array(m2_errors) - np.array(m1_errors) #previous - current gives improvement (reduction) in cost. Higher=better
			#dErrors_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(dErrors)
			dErrors_scaled = sklearn.preprocessing.StandardScaler().fit_transform(dErrors)
			times_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(times)

			dErrors_positive = sklearn.preprocessing.MinMaxScaler().fit_transform(dErrors)
			dbscan = sklearn.cluster.DBSCAN(eps=0.5,min_samples=5,metric="precomputed")
			dbscan.fit(m1_eucmat,dErrors_positive)

			abs_dErrors = np.absolute(dErrors)
			binned_times = (times>model2["train_stop"]).astype(int)
			old = binned_times.astype(bool)<1
			new = binned_times.astype(bool)

			dErrors_stdev = np.std(dErrors)
			dErrors_mean = np.mean(dErrors)
			dErrors_colourcap = 5
			dErrors_colournorm = matplotlib.colors.Normalize(vmin=-dErrors_colourcap * dErrors_stdev,vmax=dErrors_colourcap * dErrors_stdev)

			model2["tsne"].tsne_update(model1["tsne"].P, c=eucdiffs)

			'''***
			print "scipy.stats.ks_2samp(dErrors[old],dErrors[new]):",scipy.stats.ks_2samp(dErrors[old],dErrors[new])
			print "scipy.stats.ks_2samp(abs_dErrors[old],abs_dErrors[new]):",scipy.stats.ks_2samp(abs_dErrors[old],abs_dErrors[new])
			print "scipy.stats.ks_2samp(probdiffs[old],probdiffs[new]):",scipy.stats.ks_2samp(probdiffs[old],probdiffs[new])
			print "scipy.stats.ks_2samp(cosdiffs[old],cosdiffs[new]):",scipy.stats.ks_2samp(cosdiffs[old],cosdiffs[new])
			print "scipy.stats.ks_2samp(eucdiffs[old],eucdiffs[new]):",scipy.stats.ks_2samp(eucdiffs[old],eucdiffs[new])

			print "----------"

			print "scipy.stats.ttest_ind(dErrors[old],dErrors[new], equal_var=False):",scipy.stats.ttest_ind(dErrors[old],dErrors[new], equal_var=False)
			print "scipy.stats.ttest_ind(abs_dErrors[old],abs_dErrors[new], equal_var=False)):",scipy.stats.ttest_ind(abs_dErrors[old],abs_dErrors[new], equal_var=False)
			print "scipy.stats.ttest_ind(probdiffs[old],probdiffs[new], equal_var=False)):",scipy.stats.ttest_ind(probdiffs[old],probdiffs[new], equal_var=False)
			print "scipy.stats.ttest_ind(cosdiffs[old],cosdiffs[new], equal_var=False)):",scipy.stats.ttest_ind(cosdiffs[old],cosdiffs[new], equal_var=False)
			print "scipy.stats.ttest_ind(eucdiffs[old],eucdiffs[new], equal_var=False)):",scipy.stats.ttest_ind(eucdiffs[old],eucdiffs[new], equal_var=False)

			print "----------"
			print 'scipy.stats.pearsonr(times,abs_dErrors):',scipy.stats.pearsonr(times,abs_dErrors)
			print 'scipy.stats.pearsonr(binned_times,abs_dErrors):',scipy.stats.pearsonr(binned_times,abs_dErrors)
			print 'scipy.stats.pearsonr(probdiffs,abs_dErrors):',scipy.stats.pearsonr(probdiffs,abs_dErrors)
			print 'scipy.stats.pearsonr(cosdiffs,abs_dErrors):',scipy.stats.pearsonr(cosdiffs,abs_dErrors)
			print 'scipy.stats.pearsonr(eucdiffs,abs_dErrors):',scipy.stats.pearsonr(eucdiffs,abs_dErrors)
			print 'scipy.stats.pearsonr(times,probdiffs):',scipy.stats.pearsonr(times,probdiffs)
			print 'scipy.stats.pearsonr(times,cosdiffs):',scipy.stats.pearsonr(times,cosdiffs)
			print 'scipy.stats.pearsonr(times,eucdiffs):',scipy.stats.pearsonr(times,eucdiffs)
			print 'scipy.stats.pearsonr(binned_times,probdiffs):',scipy.stats.pearsonr(binned_times,probdiffs)
			print 'scipy.stats.pearsonr(binned_times,cosdiffs):',scipy.stats.pearsonr(binned_times,cosdiffs)
			print 'scipy.stats.pearsonr(binned_times,eucdiffs):',scipy.stats.pearsonr(binned_times,eucdiffs)
			print 'scipy.stats.pearsonr(cosdiffs,eucdiffs):',scipy.stats.pearsonr(cosdiffs,eucdiffs)

			print "----------"
			print 'scipy.stats.spearmanr(times,abs_dErrors):',scipy.stats.spearmanr(times,abs_dErrors)
			print 'scipy.stats.spearmanr(binned_times,abs_dErrors):',scipy.stats.spearmanr(binned_times,abs_dErrors)
			print 'scipy.stats.spearmanr(probdiffs,abs_dErrors):',scipy.stats.spearmanr(probdiffs,abs_dErrors)
			print 'scipy.stats.spearmanr(cosdiffs,abs_dErrors):',scipy.stats.spearmanr(cosdiffs,abs_dErrors)
			print 'scipy.stats.spearmanr(eucdiffs,abs_dErrors):',scipy.stats.spearmanr(eucdiffs,abs_dErrors)
			print 'scipy.stats.spearmanr(times,probdiffs):',scipy.stats.spearmanr(times,probdiffs)
			print 'scipy.stats.spearmanr(times,cosdiffs):',scipy.stats.spearmanr(times,cosdiffs)
			print 'scipy.stats.spearmanr(times,eucdiffs):',scipy.stats.spearmanr(times,eucdiffs)
			print 'scipy.stats.spearmanr(binned_times,probdiffs):',scipy.stats.spearmanr(binned_times,probdiffs)
			print 'scipy.stats.spearmanr(binned_times,cosdiffs):',scipy.stats.spearmanr(binned_times,cosdiffs)
			print 'scipy.stats.spearmanr(binned_times,eucdiffs):',scipy.stats.spearmanr(binned_times,eucdiffs)
			print 'scipy.stats.spearmanr(cosdiffs,eucdiffs):',scipy.stats.spearmanr(cosdiffs,eucdiffs)

			print "----------"
			from minepy import minestats
			print 'minestats(times,abs_dErrors)["mic"]:',minestats(times,abs_dErrors)["mic"]
			print 'minestats(binned_times,abs_dErrors)["mic"]:',minestats(binned_times,abs_dErrors)["mic"]
			print 'minestats(probdiffs,abs_dErrors)["mic"]:',minestats(probdiffs,abs_dErrors)["mic"]
			print 'minestats(cosdiffs,abs_dErrors)["mic"]:',minestats(cosdiffs,abs_dErrors)["mic"]
			print 'minestats(eucdiffs,abs_dErrors)["mic"]:',minestats(eucdiffs,abs_dErrors)["mic"]
			print 'minestats(times,probdiffs)["mic"]:',minestats(times,probdiffs)["mic"]
			print 'minestats(times,cosdiffs)["mic"]:',minestats(times,cosdiffs)["mic"]
			print 'minestats(times,eucdiffs)["mic"]:',minestats(times,eucdiffs)["mic"]
			print 'minestats(binned_times,probdiffs)["mic"]:',minestats(binned_times,probdiffs)["mic"]
			print 'minestats(binned_times,cosdiffs)["mic"]:',minestats(binned_times,cosdiffs)["mic"]
			print 'minestats(binned_times,eucdiffs)["mic"]:',minestats(binned_times,eucdiffs)["mic"]
			print 'minestats(eucdiffs,cosdiffs)["mic"]:',minestats(eucdiffs,cosdiffs)["mic"]
			***'''
			#Comparisons against time and error
			plt.figure(figsize=(10,10))
			plt.scatter(times,probdiffs, c=dErrors_scaled, cmap="coolwarm_r", s=15)  # Red is for low dError (got worse in the new model), blue is for high dError (got better)
			plt.savefig(os.path.join(save_path,"probdiffs_v_times_c=dErrors_scaled.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,eucdiffs, c=dErrors_scaled, cmap="coolwarm_r", s=15)
			plt.savefig(os.path.join(save_path,"eucdiffs_v_times_c=dErrors_scaled.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,cosdiffs, c=dErrors_scaled, cmap="coolwarm_r", s=15)
			plt.savefig(os.path.join(save_path,"cosdiffs_v_times_c=dErrors_scaled.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,euc_cosdiffs, c=dErrors_scaled, cmap="coolwarm_r", s=15)
			plt.savefig(os.path.join(save_path,"euc_cosdiffs_v_times_c=dErrors_scaled.png"), bbox_inches='tight')
			plt.close("all")

			plt.figure(figsize=(10,10))
			plt.scatter(times,dErrors, c=sq_probdiffs, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"dErrors_v_times_c=probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,dErrors, c=eucdiffs, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"dErrors_v_times_c=eucdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,dErrors, c=cosdiffs, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"dErrors_v_times_c=cosdiffs.png"), bbox_inches='tight')
			plt.close("all")

			plt.figure(figsize=(10,10))
			plt.scatter(cosdiffs,dErrors, c=times_scaled, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"cosdiffs_v_dErrors_c=time.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(probdiffs,dErrors, c=times_scaled, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"probdiffs_v_dErrors_c=time.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(eucdiffs,dErrors, c=times_scaled, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"eucdiffs_v_dErrors_c=time.png"), bbox_inches='tight')
			plt.close("all")

			plt.figure(figsize=(10,10))
			plt.scatter(cosdiffs,dErrors, c=binned_times, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"cosdiffs_v_dErrors_c=binned_times.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(probdiffs,dErrors, c=binned_times, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"probdiffs_v_dErrors_c=binned_times.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(eucdiffs,dErrors, c=binned_times, cmap="copper", s=15)
			plt.savefig(os.path.join(save_path,"eucdiffs_v_dErrors_c=binned_times.png"), bbox_inches='tight')
			plt.close("all")

			#Comparisons against time and absolute error
			plt.figure(figsize=(10,10))
			plt.scatter(times,probdiffs, c=dErrors, cmap="coolwarm_r", s=15, norm=dErrors_colournorm)  # Red is for low dError (got worse in the new model), blue is for high dError (got better)
			plt.savefig(os.path.join(save_path,"probdiffs_v_times_c=dErrors.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,eucdiffs, c=dErrors, cmap="coolwarm_r", s=15, norm=dErrors_colournorm)
			plt.savefig(os.path.join(save_path,"eucdiffs_v_times_c=dErrors.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,cosdiffs, c=dErrors, cmap="coolwarm_r", s=15, norm=dErrors_colournorm)
			plt.savefig(os.path.join(save_path,"cosdiffs_v_times_c=dErrors.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(10,10))
			plt.scatter(times,euc_cosdiffs, c=dErrors, cmap="coolwarm_r", s=15, norm=dErrors_colournorm)
			plt.savefig(os.path.join(save_path,"euc_cosdiffs_v_times_c=dErrors.png"), bbox_inches='tight')
			plt.close("all")


			'''
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
			'''

			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=sq_probdiffs, cmap="copper", s=[basesize+(modsize*p) for p in sq_probdiffs])
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=eucdiffs, cmap="copper", s=[basesize+(modsize*p) for p in eucdiffs])
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_eucdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=times_scaled, cmap="copper", s=[basesize+(modsize*p) for p in times_scaled])
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_times.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=binned_times, cmap="copper", s=[basesize+(modsize*p) for p in binned_times])
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_times_binned.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=dErrors_scaled, cmap="coolwarm_r", s=30)
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_dErrors_scaled.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=dErrors, cmap="coolwarm_r", s=30, norm=dErrors_colournorm)
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_dErrors.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(ObyA_tsne[:,0],ObyA_tsne[:,1], c=cosdiffs, cmap="copper", s=[basesize+(modsize*p) for p in cosdiffs])
			plt.savefig(os.path.join(save_path,"ObyA_TSNE_cosdiffs.png"), bbox_inches='tight')
			plt.close("all")

			'''
			plt.figure(figsize=(20,20))
			plt.scatter(model1["tsne"][:,0],model1["tsne"][:,1], c=probdiffs, cmap="copper", s=[basesize+(modsize*p) for p in sq_probdiffs])
			plt.savefig(os.path.join(save_path,model1["name"]+"_FbyO_TSNE_probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(model2["tsne"][:,0],model2["tsne"][:,1], c=probdiffs, cmap="copper", s=[basesize+(modsize*p) for p in sq_probdiffs])
			plt.savefig(os.path.join(save_path,model2["name"]+"_FbyO_TSNE_probdiffs.png"), bbox_inches='tight')
			plt.close("all")
			'''


			'''
			plt.figure(figsize=(20,20))
			plt.scatter(model1["tsne"][:,0],model1["tsne"][:,1], c=cosdiffs, cmap="copper", s=[basesize+(modsize*p) for p in sq_cosdiffs])
			plt.savefig(os.path.join(save_path,model1["name"]+"_FbyO_TSNE_cosdiffs.png"), bbox_inches='tight')
			plt.close("all")
			plt.figure(figsize=(20,20))
			plt.scatter(model2["tsne"][:,0],model2["tsne"][:,1], c=cosdiffs, cmap="copper", s=[basesize+(modsize*p) for p in sq_cosdiffs])
			plt.savefig(os.path.join(save_path,model2["name"]+"_FbyO_TSNE_cosdiffs.png"), bbox_inches='tight')
			plt.close("all")
			'''

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
		import theano
		import os.path
		from pylearn2.utils import safe_zip
		from pylearn2.utils.data_specs import DataSpecsMapping

		print hypers

		objectives = []
		objectives_by_layer = []
		model_files = []
		models = []
		encoders = []
		decoders = []
		costs = []
		test_indices = []
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
			else:
				layer_hypers["nvis"] = data.X.shape[1]
			layer_yaml = layer_yaml % (layer_hypers)
			print "-----LAYER_"+str(layer_num)+"-----"
			print layer_yaml
			print "-----------------"
			train = yaml_parse.load(layer_yaml)
			if len(pretrained):
				train.model = pickle.load(open(pretrained[layer_num-1]))
				print "Replaced model with pretrained one from",pretrained[layer_num-1]
				train.model.monitor = Monitor(train.model)
				train.model.corruptor.__init__(train.model.corruptor.corruption_level)
			train.main_loop()


			if 'model' in dir(train):
				obj = train.model.monitor.channels['objective'].val_record[-1] #This is a single Train object with no k-fold CV happening.
				objectives_by_layer.append([float(i) for i in train.model.monitor.channels['objective'].val_record])
				I = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
				E = train.model.encode(I)
				encoders.append(theano.function( [I], E ))

				H = train.model.get_output_space().make_theano_batch(batch_size=hypers["batch_size"])
				D = train.model.decode(H)
				decoders.append(theano.function( [H], D ))

				I2 = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
				costs.append(theano.function([I,I2], train.algorithm.cost.costs[0].cost(I,I2)))
			else:

				obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers]) #This is a TrainCV object that's doing k-fold CV.
				objectives_by_layer.append([float(j) for j in np.mean([i.model.monitor.channels['test_objective'].val_record for i in train.trainers],axis=0)])
				enc = []
				dec = []
				cst = []

				for k,fold in enumerate(train.trainers):
					I = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
					E = fold.model.encode(I)
					enc.append(theano.function( [I], E ))

					H = fold.model.get_output_space().make_theano_batch(batch_size=hypers["batch_size"])
					D = fold.model.decode(H)
					dec.append(theano.function( [H], D ))

					I2 = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
					cst.append(theano.function([I,I2], fold.algorithm.cost.costs[0].cost(I,I2)))

					if layer_num == 1:
						test_indices.append(train.dataset_iterator.subset_iterator[k][1]) # subset_iterator is a list of (train,test) tuples of indices.
				encoders.append(enc)
				decoders.append(dec)
				costs.append(cst)
			print "obj:",obj
			print "objectives_by_layer:",objectives_by_layer[-1]
			objectives.append(obj)
			models.append(train)
			model_files.append(os.path.join(layer_hypers['save_path'],hypers["layer_fn"]+"_l"+str(layer_num)+".pkl"))

		deep_recon_error = deep_recon(data.X,encoders,decoders,costs[0],hypers["batch_size"], test_indices)

		print "deep_recon_error:",deep_recon_error
		#objective = float(np.sum(objectives))
		print "errors_by_layer:",objectives_by_layer
		if logging:
			return {"objective": deep_recon_error,
					"training_objectives": objectives_by_layer,
					"model_files": model_files}
					#"models": models,
					#"training_data": data}
		else:
			return deep_recon_error



	def __init__(self, domain_name, scratch_path = "", selected_hypers={}, max_epochs = 10):
		from pylearn2.config import yaml_parse
		self.fixed_hypers["costs"] = yaml_parse.load('cost : !obj:pylearn2.costs.cost.SumOfCosts {costs: [!obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {}, !obj:pylearn2.costs.autoencoder.SparseActivation {coeff: 1,p: 0.15}]}')['cost']
		self.fixed_hypers["max_epochs"] = max_epochs
		ConceptualSpace.__init__(self, domain_name, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)

class LSTMConceptualSpace(ConceptualSpace):
	fixed_hypers = {}
	hyper_space = {}

	def __init__(self, domain_name, scratch_path = "", selected_hypers={}, max_epochs = 10):
		ConceptualSpace.__init__(self, domain_name, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)


class VAEConceptualSpace(ConceptualSpace):
	fixed_hypers = {"batch_size": 1000,
					"monitoring_batch_size": 100,
					"save_path": ".",
					"yaml_path": "../../../../model_yamls/",
					"layer_fn": "vae",
					"num_layers": 1,
					"n_folds": 5,
	                "nhid_mlp1": 1000,
	                "nhid_mlp2": 1000,
	                "mom_max": 0.95,
					"monary_type": "exists"
	}
	hyper_space = { "nhid": range(100,301,100),
	                "mom_init": [.0, .3, .6],#list(np.linspace(0,0.5, 6)),
	                "mom_fin": [.3, .6, .9],#list(np.linspace(0,1, 6)),
					#"nhid_mlp1": range(1000,1001,50),
					#"nhid_mlp2": range(1000,1001,50),
					"learn_rate": [5e-3, 1e-2, 2e-2],
					"max_epochs": [50, 100, 150]
	}
	def load(self, f):
		self.model = pickle.load(open(f))

	def init_model_functions(self):
		hypers = {}
		hypers.update(self.fixed_hypers)
		hypers.update(self.hyper_space)
		I = self.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
		recon = self.model.reconstruct(I, False)
		self._recon = theano.function([I], recon,  allow_input_downcast=True)

		I2 = self.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
		recon_noisy = self.model.reconstruct(I2, True)
		self._recon_noisy = theano.function([I2], recon_noisy, allow_input_downcast=True)

		I3 = self.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
		recon_return = self.reconstruct_and_return(I3, False)
		self._recon_return = theano.function([I3], recon_return,  allow_input_downcast=True)

		I4 = self.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
		recon_return_noisy = self.reconstruct_and_return(I4, True)
		self._recon_return_noisy = theano.function([I4], recon_return_noisy, allow_input_downcast=True)

		I5 = self.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
		R = self.model.log_likelihood_approximation(I5, 20)
		self._recon_cost = theano.function([I5],R, allow_input_downcast=True)

	def reconstruct_and_return(self, X, noisy_encoding=False, return_sample_means=True):
		epsilon = self.model.sample_from_epsilon((X.shape[0], self.model.nhid))
		if not noisy_encoding:
			epsilon *= 0
		# Encode q(z | x) parameters
		phi = self.model.encode_phi(X)
		# Compute z
		z = self.model.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
		# Compute expectation term
		theta = self.model.decode_theta(z)
		reconstructed_X = self.model.sample_from_p_x_given_z(num_samples=X.shape[0], theta=theta)
		return (reconstructed_X, self.model.means_from_theta(theta), epsilon, phi[0], phi[1], z, theta[0])


	def featurewise_inspect(self,metadata,override_query, sample_sizes = 10000, n_iter=100):
		self.metadata = metadata
		threshold = 2
		depth_limit = 3
		n_reform_samples = 50
		n_reform_iter = 10
		reform_drop = False
		prefix = "i_"
		Surprise = namedtuple("Surprise",("discovery","context"))

		bacon_cupcake = ["sugar", "butter", "flour", "bacon", "cocoa", "salt", "eggs", "baking powder"]
		bacon_cupcake = [prefix+i for i in bacon_cupcake]
		southwest_riblets = ['i_chocolate', 'i_apple cider vinegar', 'i_onions', 'i_sugar', 'i_water', 'i_garlic', 'i_chillies', 'i_salt', 'i_vegetable oil', "i_pork"]
		bacon_pasta = ['i_pecans', 'i_dill', 'i_cheese', 'i_salt', 'i_eggs', 'i_black pepper', 'i_bacon', 'i_parmesan', 'i_butter', 'i_vodka', 'i_pasta', 'i_white wine']
		velvet_salad = ['i_mixed nuts', 'i_water', 'i_cream cheese', 'i_whipped cream', 'i_cherries', 'i_celery', "i_gelatin"]
		meatlessloaf = ['i_cottage cheese', 'i_onions', 'i_eggs', 'i_rice', 'i_peanuts', 'i_black pepper', 'i_salt', 'i_olive oil']
		dilly_muffins = ['i_sugar', 'i_ricotta', 'i_baking powder', 'i_margarine', 'i_eggs', 'i_flour', 'i_zucchini', 'i_dill', 'i_salt', 'i_milk']
		bloody_mary = ['i_worcestershire sauce', 'i_lemon', 'i_black pepper', 'i_salt', 'i_vodka', 'i_tomatoes']



		triggers = [bacon_cupcake, dilly_muffins, velvet_salad]
#		surps = [{Surprise(discovery='i_cottage cheese', context=frozenset(['i_peanuts', 'i_eggs', 'i_onions'])):9.8},
#				 {Surprise(discovery='i_worcestershire sauce', context=frozenset(["i_tomatoes", "i_vodka"])):13.},
#				 {Surprise(discovery='i_pork', context=frozenset(["i_chocolate", "i_salt", "i_sugar"])):10.8},
#				 {Surprise(discovery='i_celery', context=frozenset(['i_whipped cream', 'i_cherries', 'i_cream cheese'])):11.4},
#				 {Surprise(discovery='i_dill', context=frozenset(['i_margarine', 'i_ricotta', 'i_zucchini'])):15.5}]

		#triggers = triggers[:-2]
		#surps = surps[:-2]

	#	for trigger,surp in zip(triggers,surps):
		for trigger in triggers:
			print "Surprising combinations in",trigger," (from prompt):"
			s = self.surprising_sets(trigger, threshold=threshold, n_samples=100000, depth_limit = depth_limit, beam_width=None)
			self.print_surprise(s)
			#sys.exit()
			#s = {Surprise(discovery='i_bacon', context=frozenset(['i_cocoa', 'i_butter', "i_sugar"])):10.8}
			#s = {Surprise(discovery='i_celery', context=frozenset(['i_whipped cream', 'i_cherries', 'i_cream cheese'])):11.4}
			#s = {Surprise(discovery='i_black pepper', context=frozenset(['i_peanuts', 'i_eggs', 'i_olive oil'])):15.5}
			#s = {Surprise(discovery='i_dill', context=frozenset(['i_margarine', 'i_ricotta', 'i_zucchini'])):15.5}
			print "using saved surprise sets:"
#			self.print_surprise(surp)
#			self.reformulation_test(surp, n_reform_samples,n_reform_iter,reform_drop,depth_limit,threshold)

		sys.exit()
		if len(override_query.keys()):
			q = override_query
		else:
			q = deepcopy(metadata["query"])
		train_data, test_data = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=q, split = 0.9, shuffle_split=True, type=self.fixed_hypers["monary_type"])
		all_data = np.vstack((train_data.X,test_data.X))
		data_means = np.array(all_data).mean(axis=0)

		print "   --- Data loaded."

		samples_to_print = 3

		print "---------SAMPLES FROM DATA:---------"
		self.data_samples = []
		for i in range(samples_to_print):
			samp = all_data[np.random.randint(all_data.shape[0]),:]
			self.data_samples.append(samp)
			print self.binarise_features(self.features_from_design_vector(samp), 0)
		print "------------------------------------"

		R = self.model.sample(sample_sizes, return_sample_means=False)
		_sample = theano.function([],R)
		sampled_designs = _sample()
		sample_means = np.array(sampled_designs).mean(axis=0)

		print "---------SAMPLES FROM MODEL:---------"
		for i in range(samples_to_print):
			samp = sampled_designs[np.random.randint(sampled_designs.shape[0]),:]
			print self.binarise_features(self.features_from_design_vector(samp), 0)
		print "------------------------------------"

		#self.co_occurence_matrix(outpath="co_occurence.csv")



		max_data_surps = []
		for s in self.data_samples:
			d = self.binarise_features(self.features_from_design_vector(s))
			print "Surprising combinations in",d," (from data):"
			s = self.surprising_sets(d, threshold=threshold, n_samples=1000, depth_limit = depth_limit, beam_width=len(d))
			self.print_surprise(s)
			max, best = self.max_surprise(s, return_best=True)
			max_data_surps.append(max)
			self.reformulation_test(s, n_reform_samples,n_reform_iter,reform_drop,depth_limit,threshold)

		print max_data_surps

		max_model_surps = []
		for i in range(len(self.data_samples)):
			s = sampled_designs[np.random.randint(sampled_designs.shape[0]),:]
			d = self.binarise_features(self.features_from_design_vector(s))
			print "Surprising combinations in",d," (from model):"
			s = self.surprising_sets(d, threshold=threshold, n_samples=1000, depth_limit = depth_limit, beam_width=len(d))
			self.print_surprise(s)
			max, best = self.max_surprise(s, return_best=True)
			max_model_surps.append(max)
			self.reformulation_test(s, n_reform_samples,n_reform_iter,reform_drop,depth_limit,threshold)

		print max_model_surps

		print "Mean surprise from data was {0} w/ sigma {1}. Mean surprise from model was {2} w/ sigma {3}.".format(np.mean(max_data_surps,axis=0),np.std(max_data_surps,axis=0),np.mean(max_model_surps,axis=0),np.std(max_model_surps,axis=0))

	def reformulation_test(self,s, n_reform_samples,n_reform_iter,reform_drop,depth_limit,threshold, replication=True,same_context=True,same_discovery=True):
		max, best = self.max_surprise(s, return_best=True)
		if best is not None:
			if replication:
				print "  Generating with replication reformulation using",best
				count = 0
				cond = [best.discovery]+list(best.context)
				while count < n_reform_samples:
					designs= self.synthesise_with_reformulation(best,reform_type="replicate", n_samples=1000,n_iter=n_reform_iter, drop_bad=reform_drop)
					for des in designs:
						d = self.binarise_features(self.features_from_design_vector(des))
						if all([c in d for c in cond]):
							print
							print "   ",d
							count +=1
							if count == n_reform_samples:
								break
					print ".",
			if same_context:
				print "  Generating with same_context reformulation using",best
				if len(best.context):
					count = 0
					cond = list(best.context)
					while count < n_reform_samples:
						designs= self.synthesise_with_reformulation(best,reform_type="same_context", n_samples=1000,n_iter=n_reform_iter, drop_bad=reform_drop)
						for des in designs:
							d = self.binarise_features(self.features_from_design_vector(des))
							if all([c in d for c in cond]):
								print
								print "   ",d
								count +=1
								s = self.surprising_sets(d, fixed_context=cond, threshold=threshold, n_samples=1000, depth_limit = depth_limit, beam_width=len(d)*2)
								self.print_surprise(s, prefix="      ")
								if count == n_reform_samples:
									break
						print ".",
				else:
					print "  -- skipping same_context as context is empty."
			if same_discovery:
				print "  Generating with same_discovery reformulation using",best
				R = self.model.sample(1000, return_sample_means=False)
				_sample = theano.function([],R)
				count = 0
				cond = best.discovery
				while count < n_reform_samples:
					designs = _sample()
					#designs= self.synthesise_with_reformulation(best,reform_type="same_discovery", n_samples=1000,n_iter=n_reform_iter, drop_bad=reform_drop)
					for des in designs:
						d = self.binarise_features(self.features_from_design_vector(des))
						if cond in d:
							print "   ",d
							count +=1
							s = self.surprising_sets(d, fixed_discovery=cond, threshold=threshold, n_samples=1000, depth_limit = depth_limit, beam_width=len(d)*2)
							self.print_surprise(s, prefix="      ")
							if count == n_reform_samples:
								break
					print ".",
				print


	def inspection_test(self,metadata,override_query, sample_sizes = 10000, n_iter=1000):

		if len(override_query.keys()):
			q = override_query
		else:
			q = deepcopy(metadata["query"])
		self.metadata = metadata
		train_data, test_data = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=q, split = 0.9, shuffle_split=True, type=self.fixed_hypers["monary_type"])
		all_data = np.vstack((train_data.X,test_data.X))
		data_means = np.array(all_data).mean(axis=0)

		print "   --- Data loaded."

		samples_to_print = 3

		print "---------SAMPLES FROM DATA:---------"
		self.data_samples = []
		for i in range(samples_to_print):
			samp = all_data[np.random.randint(all_data.shape[0]),:]
			self.data_samples.append(samp)
			print self.binarise_features(self.features_from_design_vector(samp), 0)
		print "------------------------------------"

		R = self.model.sample(sample_sizes, return_sample_means=False)
		_sample = theano.function([],R)
		sampled_designs = _sample()
		sample_means = np.array(sampled_designs).mean(axis=0)

		print "---------SAMPLES FROM MODEL:---------"
		for i in range(samples_to_print):
			samp = sampled_designs[np.random.randint(sampled_designs.shape[0]),:]
			print self.binarise_features(self.features_from_design_vector(samp), 0)
		print "------------------------------------"

		condition = {"i_beef": 1}
		cond_indices = [metadata["fields_x"].index(i) for i in condition.keys() if condition[i]==1]
		actual_cond_probs = []
		count = 0.0
		for s in all_data:
			if all(s[cond_indices]):
				count+= 1.0
		condition_only_prob = count/all_data.shape[0]
		for f in range(len(metadata["fields_x"])):
			count = 0.0
			for s in all_data:
				if all(s[cond_indices+[f]]):
					count+= 1.0
			actual_cond_probs.append(count/all_data.shape[0]/condition_only_prob)

		print "------------------",condition
		it_cond = self.estimate_conditional_dists(condition, samples=sample_sizes, n_iter=n_iter)
		it_cond_drop = self.estimate_conditional_dists(condition, samples=sample_sizes, n_iter=n_iter, drop_bad_recons=True)
		vals = zip(metadata['fields_x'],data_means, sample_means, actual_cond_probs, it_cond, it_cond_drop, actual_cond_probs/data_means, it_cond/sample_means, it_cond_drop/sample_means, -np.log2(actual_cond_probs) + np.log2(data_means), -np.log2(it_cond) + np.log2(sample_means), -np.log2(it_cond_drop) + np.log2(sample_means))
		for f in vals:
				print "{0} -- d_mean: {1:.3f}, s_mean: {2:.3f}, data_prob: {3:.3f}, it_cond: {4:.3f}, it_cond_dr: {5:.3f}, data_ratio {6:.3f}, it_cond_ratio {7:.3f}, it_cond_dr_ratio {8:.3f}, data_surp {9:.3f}, it_cond_surp {10:.3f}, it_cond_dr_surp {11:.3f}".format(*f)

		condition = {"i_beef": 1, "i_garlic": 1}
		cond_indices = [metadata["fields_x"].index(i) for i in condition.keys() if condition[i]==1]
		actual_cond_probs = []
		count = 0.0
		for s in all_data:
			if all(s[cond_indices]):
				count+= 1.0
		condition_only_prob = count/all_data.shape[0]
		for f in range(len(metadata["fields_x"])):
			count = 0.0
			for s in all_data:
				if all(s[cond_indices+[f]]):
					count+= 1.0
			actual_cond_probs.append(count/all_data.shape[0]/condition_only_prob)

		print "------------------",condition
		it_cond = self.estimate_conditional_dists(condition, samples=sample_sizes, n_iter=n_iter)
		it_cond_drop = self.estimate_conditional_dists(condition, samples=sample_sizes, n_iter=n_iter, drop_bad_recons=True)
		vals = zip(metadata['fields_x'],data_means, sample_means, actual_cond_probs, it_cond, it_cond_drop, actual_cond_probs/data_means, it_cond/sample_means, it_cond_drop/sample_means, -np.log2(actual_cond_probs) + np.log2(data_means), -np.log2(it_cond) + np.log2(sample_means), -np.log2(it_cond_drop) + np.log2(sample_means))
		for f in vals:
				print "{0} -- d_mean: {1:.3f}, s_mean: {2:.3f}, data_prob: {3:.3f}, it_cond: {4:.3f}, it_cond_dr: {5:.3f}, data_ratio {6:.3f}, it_cond_ratio {7:.3f}, it_cond_dr_ratio {8:.3f}, data_surp {9:.3f}, it_cond_surp {10:.3f}, it_cond_dr_surp {11:.3f}".format(*f)

		condition = {"i_beef": 1, "i_garlic": 1, "i_chili powder": 1}
		cond_indices = [metadata["fields_x"].index(i) for i in condition.keys() if condition[i]==1]
		actual_cond_probs = []
		count = 0.0
		for s in all_data:
			if all(s[cond_indices]):
				count+= 1.0
		condition_only_prob = count/all_data.shape[0]
		for f in range(len(metadata["fields_x"])):
			count = 0.0
			for s in all_data:
				if all(s[cond_indices+[f]]):
					count+= 1.0
			actual_cond_probs.append(count/all_data.shape[0]/condition_only_prob)

		print "------------------",condition
		it_cond = self.estimate_conditional_dists(condition, samples=sample_sizes, n_iter=n_iter)
		it_cond_drop = self.estimate_conditional_dists(condition, samples=sample_sizes, n_iter=n_iter, drop_bad_recons=True)
		vals = zip(metadata['fields_x'],data_means, sample_means, actual_cond_probs, it_cond, it_cond_drop, actual_cond_probs/data_means, it_cond/sample_means, it_cond_drop/sample_means, -np.log2(actual_cond_probs) + np.log2(data_means), -np.log2(it_cond) + np.log2(sample_means), -np.log2(it_cond_drop) + np.log2(sample_means))
		for f in vals:
				print "{0} -- d_mean: {1:.3f}, s_mean: {2:.3f}, data_prob: {3:.3f}, it_cond: {4:.3f}, it_cond_dr: {5:.3f}, data_ratio {6:.3f}, it_cond_ratio {7:.3f}, it_cond_dr_ratio {8:.3f}, data_surp {9:.3f}, it_cond_surp {10:.3f}, it_cond_dr_surp {11:.3f}".format(*f)


		#print "cs.reconstruct(cs.partial_design_vector_from_features({}))"
		#for i in range(10):
		#	print i,":", cs.reconstruct(cs.partial_design_vector_from_features({}))[0]

		#print "cs.binarise_features(cs.features_from_design_vector(cs.reconstruct(cs.design_vector_from_features({}), noisy_encoding=False)[0]), 0)"
		#for i in range(10):
		#	print i,":", cs.binarise_features(cs.features_from_design_vector(cs.reconstruct(cs.design_vector_from_features({}), noisy_encoding=False)[0]), 0)


		#print 'cs.binarise_features(cs.features_from_design_vector(cs.reconstruct(cs.design_vector_from_features({"i_almonds": 1}), noisy_encoding=False)[0]), 0)'
		#for i in range(10):
		#	print i,":", cs.binarise_features(cs.features_from_design_vector(cs.reconstruct(cs.design_vector_from_features({"i_almonds": 1}), noisy_encoding=False)[0]), 0)


		#print 'cs.reconstruct(cs.design_vector_from_features({"i_walnuts": 1, "i_all-purpose flour": 1, "i_brown sugar": 1, "i_oats": 1, "i_vanilla": 1, "i_eggs": 1, "i_white sugar": 1, "i_chocolate": 1, "i_baking soda": 1, "i_salt": 1, "i_semi-sweet chocolate": 1, "i_butter": 1, "i_baking powder": 1}), noisy_encoding=False)[0]'
		#for i in range(10):
		#	d = {"i_walnuts": 1, "i_all-purpose flour": 1, "i_brown sugar": 1, "i_oats": 1, "i_vanilla": 1, "i_eggs": 1, "i_white sugar": 1, "i_chocolate": 1, "i_baking soda": 1, "i_salt": 1, "i_semi-sweet chocolate": 1, "i_butter": 1, "i_baking powder": 1}
		#	d_v = cs.design_vector_from_features(d)
		#	d_r = cs.reconstruct(d_v, noisy_encoding=True)[0]
		#	print i,":",cs.binarise_features(cs.features_from_design_vector(d_r), 0)
		#	print "    d_v.cost="+str(cs.recon_cost(d_v)), "d_r.cost="+str(cs.recon_cost(d_r))
		'''
		print "Reconstructed training samples:"
		i = 0
		num_samples_each = 1
		sample_expectations = []
		for d in self.data_samples:
			print i," d:",self.binarise_features(self.features_from_design_vector(d), 0),"d.cost="+str(self.recon_cost(d))
			for s in range(num_samples_each):
				d_r = self.reconstruct(d, noisy_encoding=False, return_all=True)
				print "    r:",self.binarise_features(self.features_from_design_vector(d_r[0]), 0),"r.cost="+str(self.recon_cost(d_r[0]))
				#print "    means:",d_r[1]
				#print "    epsilon:",d_r[2]
				#print "    phi[0]:",d_r[3]
				#print "    phi[1]:",d_r[4]
				#print "    z:",d_r[5]
				#print "    theta:",d_r[6]
				sample_expectations.append(d_r[1])
			#for s in range(num_samples_each):
			#	d_r = self.reconstruct(d, noisy_encoding=True)[0]
			#	print "    n:",self.binarise_features(self.features_from_design_vector(d_r), 0),"n.cost="+str(self.recon_cost(d_r))
			i+=1
			print
		sample_expectations = np.vstack(sample_expectations)
		print sample_expectations
		print np.mean(sample_expectations,axis=0)
		'''

	def run(self,data, hypers, logging=False, pretrained = [], cv = True):
		from pylearn2.config import yaml_parse
		import theano, pprint
		import os.path
		import theano.tensor as T

		hypers["mom_fin"] += hypers["mom_init"]
		hypers["mom_fin"] = min(hypers["mom_fin"],hypers["mom_max"])

		print hypers

		objectives = []
		objectives_by_layer = []
		model_files = []
		models = []
		encoders = []
		decoders = []
		costs = []
		test_indices = []
		yaml_ext = ".yaml"
		if not cv:
			yaml_ext = "_nocv"+yaml_ext
		yaml = open(hypers["yaml_path"]+hypers["layer_fn"]+yaml_ext, 'r').read()
		###################### Rewriting from here to work with VAE not SDAE
		if "_mongo" in hypers.keys():
			hypers["nvis"] = len(hypers["_mongo"]["fields_x"])
		else:
			hypers["nvis"] = len(self.metadata["fields_x"])
		yaml = yaml % (hypers)
		print "-----YAML-----"
		print yaml
		print "-----------------"
		train = yaml_parse.load(yaml)
		if len(pretrained):
			train.model = pickle.load(open(pretrained[0]))
			print "Replaced model with pretrained one from",pretrained[0]
			train.model.monitor = Monitor(train.model)
		train.main_loop()


		if 'model' in dir(train):
			self.model = train.model
			if data==None:
				data = train.dataset
			obj = train.model.monitor.channels['test_objective'].val_record[-1] #This is a single Train object with no k-fold CV happening.
			objectives_by_layer.append([float(i) for i in train.model.monitor.channels['test_objective'].val_record])

			I = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
			recon = train.model.reconstruct(I, False)
			recon_noisy = train.model.reconstruct(I, True)
			self._recon = theano.function([I], recon,  allow_input_downcast=True)
			self._recon_noisy = theano.function([I], recon_noisy, allow_input_downcast=True)
			R = train.model.log_likelihood_approximation(I, 20)
			self._recon_cost = theano.function([I],R, allow_input_downcast=True)

			batch_result_list = []
			for i in xrange(data.X.shape[0] / hypers["batch_size"]):
				X_batch = data.X[hypers["batch_size"] * i: hypers["batch_size"] * (i + 1)]
				batch_result_list.append(np.mean(self._recon_cost(X_batch)))

			sample_func, _ = train.model.sample(20)
			_sample = theano.function([], sample_func)
			sample = _sample()
			zsample = zip(sample, self._recon_cost(sample))
			for s in zsample:
				print "Sample: {0}, cost: {1}".format(list(s[0].astype(int)), s[1])
				print self.binarise_features(self.features_from_design_vector(s[0]), 0)

			print "All 1s:",self._recon_cost(np.ones((1,sample.shape[1])))
			print "All 0s:",self._recon_cost(np.zeros((1,sample.shape[1])))
			deep_recon_error = float(-1 * np.mean(batch_result_list))

			''' #For now, guessing that we don't need this stuff - VAE has a proper deep recon function built in.
			I = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
			E = train.model.encode_phi(I)
			encoders.append(theano.function( [I], E ))

			H = train.model.get_output_space().make_theano_batch(batch_size=hypers["batch_size"])
			D = train.model.decode(H)
			decoders.append(theano.function( [H,train.model], D ))

			I2 = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
			costs.append(theano.function([I,I2, train.model], self.VAEComparitiveCost(I,I2,train.model)))
			#'''
		else:
			if data==None:
				data = train.dataset_iterator.dataset

			obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers]) #This is a TrainCV object that's doing k-fold CV.
			objectives_by_layer.append([float(j) for j in np.mean([i.model.monitor.channels['test_objective'].val_record for i in train.trainers],axis=0)])

			recon_costs = []
			recon_errors = []
			for k,fold in enumerate(train.trainers):
				I = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])


				R = fold.model.log_likelihood_approximation(I, 20)
				recon_cost = theano.function([I], R, allow_input_downcast=True)

				batch_result_list = []
				for i in xrange(data.X.shape[0] / hypers["batch_size"]):
					X_batch = data.X[hypers["batch_size"] * i: hypers["batch_size"] * (i + 1)]
					batch_result_list.append(np.mean(recon_cost(X_batch)))
				recon_errors.append(np.mean(batch_result_list))

				sample_func, _ = fold.model.sample(20)
				_sample = theano.function([], sample_func)
				sample = _sample()
				zsample = zip(sample, recon_cost(sample))
				for s in zsample:
					print "Sample: {0}, cost: {1}".format(list(s[0].astype(int)), s[1])

				print "All 1s:",recon_cost(np.ones((1,sample.shape[1])))
				print "All 0s:",recon_cost(np.zeros((1,sample.shape[1])))

			deep_recon_error = float(-1 * np.mean(np.vstack(recon_errors)))

			''' #For now, guessing that we don't need this stuff - VAE has a proper recon function built in.
			enc = []
			dec = []
			cst = []

			for k,fold in enumerate(train.trainers):
				I = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
				E = fold.model.encode_phi(I)
				enc.append(theano.function( [I], E ))

				H = fold.model.get_output_space().make_theano_batch(batch_size=hypers["batch_size"])
				D = fold.model.decode(H)
				dec.append(theano.function( [H], D ))

				I2 = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
				cst.append(theano.function([I,I2, fold.model], self.VAEComparitiveCost(I,I2,fold.model)))

				test_indices.append(train.dataset_iterator.subset_iterator[k][1]) # subset_iterator is a list of (train,test) tuples of indices.
			encoders.append(enc)
			decoders.append(dec)
			costs.append(cst)
			#'''
		print "obj:",obj
		print "objectives_by_layer:",objectives_by_layer[-1]
		objectives.append(obj)
		models.append(train)
		model_files.append(os.path.join(hypers['save_path'],hypers["layer_fn"]+".pkl"))


		print "deep_recon_error:",deep_recon_error
		#objective = float(np.sum(objectives))
		print "errors_by_layer:",objectives_by_layer
		if logging:
			return {"objective": deep_recon_error,
					"training_objectives": objectives_by_layer,
					"model_files": model_files}
					#"models": models,
					#"training_data": data}
		else:
			return deep_recon_error

	def estimate_conditional_dists(self, observed, samples=10000, probs="random", n_iter=1, verbose=False, return_samples = False, drop_bad_recons=False):
		if type(observed) == list:
			observed = {i:1 for i in observed}
		designs = self.partial_design_vector_from_features(observed, n=samples, fill=probs)
		if verbose:
			print "observed:", observed
			print "probs:",probs
			print designs
		rec = self.reconstruct(designs,True)
		recs = [rec[1].mean(axis=0)]
		for i in range(n_iter-1):
			for d in observed.keys():
				rec[0][:,self.metadata["fields_x"].index(d)] = observed[d]
			rec = self.reconstruct(rec[0],True)
			recs.append(rec[1].mean(axis=0))
			if verbose:
				print rec[0]
		sums = np.zeros(len(self.metadata["fields_x"]))
		count = 0.0
		for samp,means in zip(rec[0],rec[1]):
			if all([samp[self.metadata["fields_x"].index(d)] == observed[d] for d in observed.keys()]):
				sums += means
				count += 1.0
		sums /= count
		#if observed.keys():
			#print "  For condition {0}, succeeded with {1} after {2} iterations.".format(observed,count,n_iter)
			#print "newmeans",sums
			#print "oldmeans",rec[1].mean(axis=0)
		if verbose:
			print rec[1].mean(axis=0)
		if drop_bad_recons:
			if return_samples:
				return sums,rec[0]
			return sums
		if return_samples:
			return rec[1].mean(axis=0), rec[0]
		return rec[1].mean(axis=0)
		#print "Reconstruction means:"
		#print recs
		#print


	def reconstruct(self, design, noisy_encoding=False, return_all=False):
		if type(design) is list:
			design = self.design_vector_from_features(design)
		recon = np.zeros(design.shape)
		means = np.zeros(design.shape)
		for i in range(design.shape[0]/self.fixed_hypers["batch_size"]):
			d = design[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:]
			if return_all:
				if noisy_encoding:
					recon[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:], means[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:] = self._recon_return_noisy(np.atleast_2d(d))
				recon[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:], means[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:] =  self._recon_return(np.atleast_2d(d))
			if noisy_encoding:
				recon[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:], means[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:] =  self._recon_noisy(np.atleast_2d(d))
			recon[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:], means[i*self.fixed_hypers["batch_size"]:(i+1)*self.fixed_hypers["batch_size"],:] =  self._recon(np.atleast_2d(d))
		return recon,means

	def recon_cost(self, design):
		if type(design) is list:
			design = self.design_vector_from_features(design)
		return self._recon_cost(np.atleast_2d(design))[0]

	# Takes dict of features with 0/1 values, sets them, makes everything else 0.
	def design_vector_from_features(self,features):
		feature_list = self.metadata["fields_x"]
		design = np.zeros(len(feature_list))
		for k,v in features.iteritems():
			if k in feature_list:
				design[feature_list.index(k)] = v
		return design


	# Takes dict of features with 0/1 values, sets them, makes everything else random -- either uniformly or normally given provided vectors of mu and sigma.
	def partial_design_vector_from_features(self, features, n=1, fill="random"):
		feature_list = self.metadata["fields_x"]
		design = np.zeros((n,len(feature_list)))
		if fill == "random":
			design = np.random.random_integers(0,high=1,size=(n,len(feature_list)))
		elif type(fill) in [list,np.ndarray]: # Indicates proportional randomness
			for i in range(n):
				for k in range(len(design)):
					if np.random.random() < fill[k]:
						design[i,k] = 1
		else:
			design += 0.5
		for k,v in features.iteritems():
			if k in feature_list:
				design[:,feature_list.index(k)] = v
			else:
				print k,"doesn't appear to be in a variable in the design space!"
		return design


	def features_from_design_vector(self, design):
		feature_list = self.metadata["fields_x"]
		return dict(zip(feature_list,np.ravel(design)))


	def binarise_features(self, features, threshold=0):
		new = []
		for k,v in features.iteritems():
			if v > threshold:
				new.append(k.encode("ascii", "ignore"))
		return new

	def estimate_surprise_given_context(self, design, context, samples, probs, n_iter, drop_bad, sample_means=None, candidates=None):
		surprisals = {}
		if sample_means is None:
			R = self.model.sample(samples, return_sample_means=False)
			_sample = theano.function([],R)
			sampled_designs = _sample()
			sample_means = np.array(sampled_designs).mean(axis=0)

		if len(context):
			if candidates is None:
				surprise_candidates = [d for d in design if d not in context]
			else:
				surprise_candidates = candidates
			conditional_dists = self.estimate_conditional_dists(context, samples=samples, probs=probs, n_iter=n_iter, drop_bad_recons=drop_bad)

			for c in surprise_candidates:
				index = self.metadata["fields_x"].index(c)
				surprisals[c] = -np.log2(conditional_dists[index]) + np.log2(sample_means[index])
		else:
			mean_ing_likelihood = -np.log2(np.mean(sample_means))
			surprisals = {d:-np.log2(sample_means[self.metadata["fields_x"].index(d)]) - mean_ing_likelihood for d in design}
		return surprisals

	def print_surprise(self, surps, prefix=""):
		for s in surps:
			print str(prefix)+"  {0} given {1}: {2:.3f} wows.".format(s.discovery,list(s.context),surps[s])

	def sorted_surprise(self, surps):
		l = [(k,v) for k,v in surps.iteritems()]
		return sorted(l,reverse=True,key=lambda x:x[1])

	def max_surprise(self, surps, return_best=False):
		if type(surps) is dict:
			surps = self.sorted_surprise(surps)
		if len(surps):
			if return_best:
				return surps[0][1],surps[0][0]
			return surps[0][1]
		elif return_best:
			return 0,None
		return 0

	def surprising_sets(self, design, fixed_context=None, fixed_discovery=None, threshold=2, drop_threshold= None, depth_limit = None, beam_width=-1, n_samples=10000, probs="random", n_iter=25, drop_bad=True):
		openlist = deque([frozenset([])]) # Initialise the queue to just the empty set
		closedlist = set()
		foundlist = {}
		Surprise = namedtuple("Surprise",("discovery","context"))

		batch_size = 1000
		R = self.model.sample(min(batch_size,n_samples), return_sample_means=False)
		_sample = theano.function([],R)

		sampled_designs = np.zeros((n_samples,len(self.metadata["fields_x"])))

		print "   --Generating samples:"
		for i in range(n_samples/batch_size):
			sampled_designs[i*batch_size:(i+1)*batch_size,:] = _sample()
		sample_means = np.array(sampled_designs).mean(axis=0)
		'''
		if mode == "exhaustive":
			while len(openlist):
				context = openlist.popleft()
				candidate_surprisals = self.estimate_surprise_given_context(design, list(context), n_samples, probs, n_iter, drop_bad, sample_means = sample_means)
				foundlist.update({Surprise(discovery=c,context=context): candidate_surprisals[c] for c in candidate_surprisals.keys() if candidate_surprisals[c] > threshold})
				if drop_threshold is not None:
					if depth_limit is not None:
						additions = [frozenset(list(context)+[c]) for c in candidate_surprisals.keys() if candidate_surprisals[c] > drop_threshold and frozenset(list(context)+[c]) not in closedlist and len(c) <= depth_limit]
					additions = [frozenset(list(context)+[c]) for c in candidate_surprisals.keys() if candidate_surprisals[c] > drop_threshold and frozenset(list(context)+[c]) not in closedlist]
				else:
					if depth_limit is not None:
						additions = [frozenset(list(context)+[c]) for c in candidate_surprisals.keys() if frozenset(list(context)+[c]) not in closedlist and len(c) <= depth_limit]
					additions = [frozenset(list(context)+[c]) for c in candidate_surprisals.keys() if frozenset(list(context)+[c]) not in closedlist]
				openlist.extend(additions)
				closedlist.update(additions)
		elif type(mode) is tuple and mode[0] == "beam":
			'''
		print "   --Evaluating surprise."
		openlist = [(0,frozenset([]))]
		if fixed_context is not None:
			context_list = itertools.chain.from_iterable(itertools.combinations(fixed_context, r) for r in range(1,len(fixed_context)+1))
			openlist = [(0,frozenset(c)) for c in context_list]
		while len(openlist):
			context = heapq.heappop(openlist)[1]#openlist.popleft()[1]
			candidate_surprisals = self.estimate_surprise_given_context(design, list(context), n_samples, probs, n_iter, drop_bad, sample_means = sample_means)
			if fixed_discovery is None:
				foundlist.update({Surprise(discovery=c,context=context): candidate_surprisals[c] for c in candidate_surprisals.keys() if candidate_surprisals[c] > threshold})
			else:
				foundlist.update({Surprise(discovery=c,context=context): candidate_surprisals[c] for c in candidate_surprisals.keys() if candidate_surprisals[c] > threshold and c == fixed_discovery})
			if fixed_context is None:
				additions = []
				if depth_limit is None or len(context) < depth_limit:
					if drop_threshold is not None:
						if fixed_discovery is not None:
							additions = [(-abs(candidate_surprisals[c]),frozenset(list(context)+[c])) for c in candidate_surprisals.keys() if candidate_surprisals[c] > drop_threshold and frozenset(list(context)+[c]) not in closedlist and fixed_discovery not in list(context)+[c]]
						else:
							additions = [(-abs(candidate_surprisals[c]),frozenset(list(context)+[c])) for c in candidate_surprisals.keys() if candidate_surprisals[c] > drop_threshold and frozenset(list(context)+[c]) not in closedlist]
					else:
						if fixed_discovery is not None:
							additions = [(-abs(candidate_surprisals[c]),frozenset(list(context)+[c])) for c in candidate_surprisals.keys() if frozenset(list(context)+[c]) not in closedlist and fixed_discovery not in list(context)+[c]]
						else:
							additions = [(-abs(candidate_surprisals[c]),frozenset(list(context)+[c])) for c in candidate_surprisals.keys() if frozenset(list(context)+[c]) not in closedlist]
				for a in additions:
					heapq.heappush(openlist,a)
				openlist.extend(additions)
				closedlist.update(additions)
				if beam_width > 0:
					openlist = heapq.nsmallest(beam_width,openlist)
		return foundlist

	def evaluate_data_surprise(self, metadata, override_query, sample_sizes = 1000, n_iter=50, start=0,stop=-1, threshold=2, depth_limit=3):
		if len(override_query.keys()):
			q = override_query
		else:
			q = deepcopy(metadata["query"])
		self.metadata = metadata
		data_slice = monary_load(self.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=q, split = None, start=start, stop=stop, type=self.fixed_hypers["monary_type"]).X

		client = pymongo.MongoClient()
		db = client.creeval
		coll = db[metadata["name"]]
		coll2 = db[metadata["name"]+"_evals"]
		cursor = coll.find(q, skip=start, limit=stop)

		for design in data_slice:
			record = cursor.next()
			print record
			d = self.binarise_features(self.features_from_design_vector(design))
			print "Surprising combinations in",d," (from data):"
			s = self.surprising_sets(d, threshold=threshold, n_samples=sample_sizes, depth_limit = depth_limit, beam_width=len(d)*depth_limit)
			self.print_surprise(s)
			record["surprise_sets"] = [{"discovery":k.discovery,"context":list(k.context),"value":v} for k,v in s.iteritems()]
			record["max_surprise"] = self.max_surprise(s)
			coll2.save(record)



	def synthesise_with_reformulation(self, surp, reform_type="replicate", n_samples=10, n_iter = 25, drop_bad=False):
		if reform_type == "replicate":
			conditional = [surp.discovery]+list(surp.context)
		elif reform_type == "same_context":
			conditional = list(surp.context)
		elif reform_type == "same_discovery":
			conditional = [surp.discovery]
		designs = self.estimate_conditional_dists(conditional, samples=n_samples,probs="random", n_iter=n_iter,return_samples=True, drop_bad_recons=drop_bad)[1]
		return designs

	def co_occurence_matrix(self, metadata, n_samples=1000, n_iter=20, outpath = None):
		self.metadata = metadata
		m = np.zeros((len(self.metadata["fields_x"]),len(self.metadata["fields_x"])))

		batch_size = 1000
		R = self.model.sample(min(batch_size,n_samples), return_sample_means=False)
		_sample = theano.function([],R)
		sampled_designs = np.zeros((n_samples,len(self.metadata["fields_x"])))

		print "   --Generating samples:"
		for i in range(n_samples/batch_size):
			sampled_designs[i*batch_size:(i+1)*batch_size,:] = _sample()
		sample_means = np.array(sampled_designs).mean(axis=0)

		print sample_means
		for i,ing in enumerate(self.metadata["fields_x"]):
			m[i,:] = self.estimate_conditional_dists([ing],samples=n_samples,n_iter=n_iter)
			m[i,:]/= sample_means
			m[i,i] = sample_means[i]
			print m[i,:]
		if outpath is not None:
			import csv
			with open(outpath,"w") as f:
				writer = csv.writer(f)
				writer.writerow([""]+self.metadata["fields_x"])
				for f,l in zip(self.metadata["fields_x"],m):
					l[self.metadata["fields_x"].index(f)] = 1
					writer.writerow([f]+list(l))
		return m


	def __init__(self, domain_name, scratch_path = "", selected_hypers={}, max_epochs = None):
		if max_epochs is not None:
			self.fixed_hypers["max_epochs"] = max_epochs
		ConceptualSpace.__init__(self, domain_name, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)

class DBNConceptualSpace(ConceptualSpace): # INCOMPLETE -- LARGELY A COPY OF THE VAE CLASS
	fixed_hypers = {"batch_size": 500,
					"monitoring_batch_size": 500,
					"max_epochs": 10,
					"save_path": ".",
					"yaml_path": "../../../../model_yamls/",
					"layer_fn": "vae",
					"num_layers": 1,
					"n_folds": 5,
	                "nhid_mlp1": 200,
	                "nhid_mlp2": 200,
	                "mom_max": 0.95,
					"monary_type": "exists"
	}
	hyper_space = { "nhid": range(100,501,100),
	                "mom_init": list(np.linspace(0,0.5, 11)),
	                "mom_fin": list(np.linspace(0,0.5, 11)),
					#"nhid_mlp1": range(1000,1001,50),
					#"nhid_mlp2": range(1000,1001,50),
					"learn_rate": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
	}
	def load(self, f):
		self.model = pickle.load(open(f))

	def run(self,data, hypers, logging=False, pretrained = [], cv = True):
		from pylearn2.config import yaml_parse
		import theano
		import os.path
		import theano.tensor as T

		hypers["mom_fin"] += hypers["mom_init"]
		hypers["mom_fin"] = min(hypers["mom_fin"],hypers["mom_max"])

		print hypers

		objectives = []
		objectives_by_layer = []
		model_files = []
		models = []
		encoders = []
		decoders = []
		costs = []
		test_indices = []
		yaml_ext = ".yaml"
		if not cv:
			yaml_ext = "_nocv"+yaml_ext
		yaml = open(hypers["yaml_path"]+hypers["layer_fn"]+yaml_ext, 'r').read()
		###################### Rewriting from here to work with VAE not SDAE
		hypers["nvis"] = data.X.shape[1]
		yaml = yaml % (hypers)
		print "-----YAML-----"
		print yaml
		print "-----------------"
		train = yaml_parse.load(yaml)
		if len(pretrained):
			train.model = pickle.load(open(pretrained[0]))
			print "Replaced model with pretrained one from",pretrained[0]
			train.model.monitor = Monitor(train.model)
		train.main_loop()


		if 'model' in dir(train):
			self.model = train.model
			obj = train.model.monitor.channels['objective'].val_record[-1] #This is a single Train object with no k-fold CV happening.
			objectives_by_layer.append([float(i) for i in train.model.monitor.channels['objective'].val_record])

			I = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
			t_int = T.iscalar()
			reconf = train.model.reconstruct(I, t_int)
			self._recon = theano.function([I, t_int], reconf, on_unused_input='warn', allow_input_downcast=True)
			R = train.model.log_likelihood_approximation(I, 20)
			self._recon_cost = theano.function([I],R, on_unused_input='warn', allow_input_downcast=True)

			batch_result_list = []
			for i in xrange(data.X.shape[0] / hypers["batch_size"]):
				X_batch = data.X[hypers["batch_size"] * i: hypers["batch_size"] * (i + 1)]
				batch_result_list.append(np.mean(self._recon_cost(X_batch)))
			deep_recon_error = float(-1 * np.mean(batch_result_list))

			''' #For now, guessing that we don't need this stuff - VAE has a proper deep recon function built in.
			I = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
			E = train.model.encode_phi(I)
			encoders.append(theano.function( [I], E ))

			H = train.model.get_output_space().make_theano_batch(batch_size=hypers["batch_size"])
			D = train.model.decode(H)
			decoders.append(theano.function( [H,train.model], D ))

			I2 = train.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
			costs.append(theano.function([I,I2, train.model], self.VAEComparitiveCost(I,I2,train.model)))
			#'''
		else:
			obj = np.mean([i.model.monitor.channels['test_objective'].val_record[-1] for i in train.trainers]) #This is a TrainCV object that's doing k-fold CV.
			objectives_by_layer.append([float(j) for j in np.mean([i.model.monitor.channels['test_objective'].val_record for i in train.trainers],axis=0)])

			recon_costs = []
			recon_errors = []
			for k,fold in enumerate(train.trainers):
				I = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])


				R = fold.model.log_likelihood_approximation(I, 20)
				recon_cost = theano.function([I], R, allow_input_downcast=True)

				batch_result_list = []
				for i in xrange(data.X.shape[0] / hypers["batch_size"]):
					X_batch = data.X[hypers["batch_size"] * i: hypers["batch_size"] * (i + 1)]
					batch_result_list.append(np.mean(recon_cost(X_batch)))
				recon_errors.append(np.mean(batch_result_list))

				#t_int1, t_int2, t_int3 = T.iscalars(3)
				#R = fold.model.log_likelihood_lower_bound(I,  t_int1,t_int2,t_int3)

				#recon_costs.append(theano.function([I, t_int1,t_int2,t_int3],R, on_unused_input='warn', allow_input_downcast=True))
				#recon_errors.append(np.mean(np.vstack(recon_costs[-1](data.X, 1, False, False))))

			deep_recon_error = float(-1 * np.mean(np.vstack(recon_errors)))

			''' #For now, guessing that we don't need this stuff - VAE has a proper recon function built in.
			enc = []
			dec = []
			cst = []

			for k,fold in enumerate(train.trainers):
				I = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
				E = fold.model.encode_phi(I)
				enc.append(theano.function( [I], E ))

				H = fold.model.get_output_space().make_theano_batch(batch_size=hypers["batch_size"])
				D = fold.model.decode(H)
				dec.append(theano.function( [H], D ))

				I2 = fold.model.get_input_space().make_theano_batch(batch_size=hypers["batch_size"])
				cst.append(theano.function([I,I2, fold.model], self.VAEComparitiveCost(I,I2,fold.model)))

				test_indices.append(train.dataset_iterator.subset_iterator[k][1]) # subset_iterator is a list of (train,test) tuples of indices.
			encoders.append(enc)
			decoders.append(dec)
			costs.append(cst)
			#'''
		print "obj:",obj
		print "objectives_by_layer:",objectives_by_layer[-1]
		objectives.append(obj)
		models.append(train)
		model_files.append(os.path.join(hypers['save_path'],hypers["layer_fn"]+".pkl"))


		print "deep_recon_error:",deep_recon_error
		#objective = float(np.sum(objectives))
		print "errors_by_layer:",objectives_by_layer
		if logging:
			return {"objective": deep_recon_error,
					"training_objectives": objectives_by_layer,
					"model_files": model_files}
					#"models": models,
					#"training_data": data}
		else:
			return deep_recon_error

	def reconstruct(self, design, noisy_encoding=False):
		if type(design) is list:
			design = self.design_vector_from_features(design)
		return self._recon(np.atleast_2d(design), noisy_encoding)

	def recon_cost(self, design):
		if type(design) is list:
			design = self.design_vector_from_features(design)
		return self._recon_cost(np.atleast_2d(design))[0]

	# Takes dict of features with 0/1 values, sets them, makes everything else 0.
	def design_vector_from_features(self,features):
		feature_list = self.metadata["fields_x"]
		design = np.zeros(len(feature_list))
		for k,v in features.iteritems():
			design[feature_list.index(k)] = v
		return design


	# Takes dict of features with 0/1 values, sets them, makes everything else 0.5.
	def partial_design_vector_from_features(self, features):
		feature_list = self.metadata["fields_x"]
		design = np.zeros(len(feature_list))
		design += 0.5
		for k,v in features.iteritems():
			design[feature_list.index(k)] = v
		return design


	def features_from_design_vector(self, design):
		feature_list = self.metadata["fields_x"]
		return dict(zip(feature_list,np.ravel(design)))

	def binarise_features(self, features, threshold):
		new = {}
		for k,v in features.iteritems():
			if v > threshold:
				new[k] = v
		return new

	def __init__(self, domain_name, scratch_path = "", selected_hypers={}, max_epochs = 10):
		self.fixed_hypers["max_epochs"] = max_epochs
		ConceptualSpace.__init__(self, domain_name, self.hyper_space, self.fixed_hypers, scratch_path = scratch_path,selected_hypers=selected_hypers)


if __name__ == "__main__":
	cs = SDAConceptualSpace("data/")
	cs.train({}, threshold = 0.01, look_back = 5)
	#from sdae_l3_reconstruct import reconstruct
	#reconstruct("data/"+cs.exp_path+"/dae")
	#from sdae_l3_show_weights_decoder import show_weights
	#show_weights("data/"+cs.exp_path+"/dae")
