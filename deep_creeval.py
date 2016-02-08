__author__ = 'kazjon'
from conceptual_space import *
import pymongo, argparse, csv, sys, os, json

# Loads a new domain into mongodb and prepares it for deep_creeval use
def preprocess():
	pass
	# Currently we're doing this manually in other files -- there's no general preprocessor.


#Perform the escape character nonsense we need to put in the DB -- mongo keys can't start with $, we replaced with __
def escape_mongo(metadata):
	for k in metadata.keys():
		if type(metadata[k]) is dict:
			escape_mongo(metadata[k])
		if type(metadata[k]) is list:
			for l in metadata[k]:
				if type(l) is dict:
					escape_mongo(l)
		if k.startswith("$"):
			print 'found',k
			metadata["__"+k.lstrip("$")] = metadata[k]
			del metadata[k]

#Reverse the escape character nonsense we had to put in the DB -- mongo keys can't start with $, we replaced with __
def unescape_mongo(metadata):
	for k in metadata.keys():
		if type(metadata[k]) is dict:
			unescape_mongo(metadata[k])
		if type(metadata[k]) is list:
			for l in metadata[k]:
				if type(l) is dict:
					unescape_mongo(l)
		if k.startswith("__"):
			metadata["$"+k.lstrip("_")] = metadata[k]
			del metadata[k]


# Runs spearmint on a given domain and saves the best hypers for future use
def fit_hypers(domain_name, override_dataset_name=None, spearmint_params = {"look_back": 5,"stop_thresh": 0.05, 'datapath': "data/"}, hypers_to_file=True, override_query = {}, drop_fields = [], sample_limit = 0, training_epochs = 10, bypass_mongo=False):

	spearmint_params["datapath"] = os.path.join(spearmint_params["datapath"],"hyper_fitting")

	# retrieve the metadata from the db
	client = pymongo.MongoClient()
	db = client.creeval
	dataset_name = domain_name
	if override_dataset_name is not None:
		dataset_name = override_dataset_name
	metadata = db.datasets.find_one({"name": dataset_name})

	if metadata is not None:
		if bypass_mongo:
			metadata["query"] = None
		unescape_mongo(metadata)

		cs = globals()[metadata['model_class']](domain_name, spearmint_params['datapath'], max_epochs = training_epochs)
		if "monary_type" in metadata.keys():
			cs.set_monary_type(metadata["monary_type"])
		q = metadata["query"]
		if len(override_query.keys()):
			q = override_query
		metadata['best_hypers'] =  cs.train_mongo([f for f in metadata['fields_x'] if f not in drop_fields],[f for f in metadata['fields_y'] if f not in drop_fields], q, spearmint_params['stop_thresh'], spearmint_params['look_back'], sample_limit=sample_limit)

		# Save the best hypers to the database
		if len(metadata['best_hypers'].keys()):
			if hypers_to_file:
				with open(os.path.join(spearmint_params['datapath'],"hypers.txt"),"w") as f:
					json.dump(metadata['best_hypers'], f, sort_keys=True, indent=4, ensure_ascii=False)
			escape_mongo(metadata)
			db.datasets.save(metadata)
		else:
			print "Spearmint hyperparameter optimisation failed, check logs for details."
	else:
		print "Could not find a record for the dataset",dataset_name,"in the database."

# Builds the deep learning model over each timeslice of the data, saving successive states to the database
def train_expectations(domain_name, override_dataset_name=None, pretrain_start = None, pretrain_stop = None, train_stop = None, time_slice = None, datapath="data/", hypers_from_file=True, steps_to_file=True, override_query = {}, drop_fields = [], training_epochs = 10, sample_limit=0, bypass_mongo=False):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	dataset_name = domain_name
	if override_dataset_name is not None:
		dataset_name = override_dataset_name
	metadata = db.datasets.find_one({"name": dataset_name})

	if metadata is not None:
		#Use the previously-fit hypers to train a single model with the provided timeslices (or ones from the data)
		if hypers_from_file:
			with open(os.path.join(datapath,"../hyper_fitting/hypers.txt")) as f:
				metadata["best_hypers"] = json.load(f)
		unescape_mongo(metadata)
		if "best_hypers" in metadata.keys():
			if pretrain_start is not None:
				metadata["pretrain_start"] = pretrain_start
			if pretrain_stop is not None:
				metadata["pretrain_stop"] = pretrain_stop
			if train_stop is not None:
				metadata["train_stop"] = train_stop
			if time_slice is not None:
				metadata["time_slice"] = time_slice

			if all(i in metadata.keys() for i in ["pretrain_start","pretrain_stop","train_stop","time_slice"]):
				original_fields_x = metadata["fields_x"]
				original_fields_y = metadata["fields_y"]
				metadata["fields_x"] = [f for f in metadata['fields_x'] if f not in drop_fields]
				metadata["fields_y"] = [f for f in metadata['fields_y'] if f not in drop_fields]
				cs = globals()[metadata['model_class']](domain_name, datapath,selected_hypers=metadata["best_hypers"], max_epochs = training_epochs)
				if "monary_type" in metadata.keys():
					cs.set_monary_type(metadata["monary_type"])
				# Fit the initial model
				step_0 = cs.pretrain(metadata, override_query,sample_limit)
				#pickle.dump(pretrained_model,open("pretrain_test.pkl","wb"))
				#pretrained_model = pickle.load(open("pretrain_test.pkl","rb"))
				#print "loaded."
				step_0["start"] = metadata["pretrain_start"]
				step_0["stop"] =  metadata["pretrain_stop"]
				#step_0 = "pretrain_skipped"
				metadata['steps'] = [step_0]
				cs.stepwise_train(metadata, override_query,sample_limit)

				if steps_to_file:
					with open(os.path.join(datapath,"steps.txt"),"w") as f:
						json.dump(metadata['steps'], f, sort_keys=True, indent=4, ensure_ascii=False)
				metadata["fields_x"] = original_fields_x
				metadata["fields_y"] = original_fields_y
				escape_mongo(metadata)
				db.datasets.save(metadata)
			else:
				print "Need valid pretrain_start, pretrain_stop, time_stop and time_slice parameters to train a model."
		else:
			print "The database",dataset_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",dataset_name,"in the database."

# Builds the deep learning model over each timeslice of the data, saving successive states to the database
def train_holistic_expectations(domain_name, override_dataset_name=None, datapath="data/", hypers_from_file=True, update_metadata=True, override_query = {}, drop_fields = [], steps_to_file=True, training_epochs = None, sample_limit=0, bypass_mongo=False):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

	if metadata is not None:
		if bypass_mongo:
			metadata["query"] = None
		#Use the previously-fit hypers to train a single model with the provided timeslices (or ones from the data)
		if hypers_from_file:
			with open(os.path.join(datapath,"../hyper_fitting/hypers.txt")) as f:
				metadata["best_hypers"] = json.load(f)
		unescape_mongo(metadata)
		if "best_hypers" in metadata.keys():
			original_fields_x = metadata["fields_x"]
			original_fields_y = metadata["fields_y"]
			metadata["fields_x"] = [f for f in metadata['fields_x'] if f not in drop_fields]
			metadata["fields_y"] = [f for f in metadata['fields_y'] if f not in drop_fields]
			cs = globals()[metadata['model_class']](domain_name, datapath,selected_hypers=metadata["best_hypers"], max_epochs = training_epochs)
			if "monary_type" in metadata.keys():
				cs.set_monary_type(metadata["monary_type"])

			# Fit the initial model
			pretrain_result = cs.pretrain(metadata, override_query, sample_limit)
			if ["pretrain_start"] in metadata.keys() and metadata["pretrain_start"] is None:
				del metadata["pretrain_start"]
			if ["pretrain_stop"] in metadata.keys() and metadata["pretrain_stop"] is None:
				del metadata["pretrain_stop"]
			metadata["steps"] = [pretrain_result]

			if steps_to_file:
				with open(os.path.join(datapath,"steps.txt"),"w") as f:
					json.dump(metadata['steps'], f, sort_keys=True, indent=4, ensure_ascii=False)

			if update_metadata:
				metadata["fields_x"] = original_fields_x
				metadata["fields_y"] = original_fields_y
				if metadata["pretrain_start"] == None:
					del metadata["pretrain_start"]
				if metadata["pretrain_stop"] == None:
					del metadata["pretrain_stop"]
				escape_mongo(metadata)
				db.datasets.save(metadata)
		else:
			print "The database",domain_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",domain_name,"in the database."

# Measures the trend-based unexpectedness of a given saved model.
def unexpectedness(domain_name, override_dataset_name=None, pretrain_start = None, pretrain_stop = None, train_stop = None, time_slice = None, sample_size= 50000, datapath="data/", steps_from_file=True, override_query = {}, drop_fields = [], start_step = 0, bypass_mongo=False):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	dataset_name = domain_name
	if override_dataset_name is not None:
		dataset_name = override_dataset_name
	metadata = db.datasets.find_one({"name": dataset_name})

	if metadata is not None:
		if bypass_mongo:
			metadata["query"] = None
		#Use the previously-trained expectation model to investigate changes in the dataset.
		if steps_from_file:
			with open(os.path.join(datapath,"steps.txt")) as f:
				metadata["steps"] = json.load(f)
		unescape_mongo(metadata)
		if "best_hypers" in metadata.keys():
			if pretrain_start is not None:
				metadata["pretrain_start"] = pretrain_start
			if pretrain_stop is not None:
				metadata["pretrain_stop"] = pretrain_stop
			if train_stop is not None:
				metadata["train_stop"] = train_stop
			if time_slice is not None:
				metadata["time_slice"] = time_slice

			if all(i in metadata.keys() for i in ["pretrain_start","pretrain_stop","train_stop","time_slice"]):
				original_fields_x = metadata["fields_x"]
				original_fields_y = metadata["fields_y"]
				metadata["fields_x"] = [f for f in metadata['fields_x'] if f not in drop_fields]
				metadata["fields_y"] = [f for f in metadata['fields_y'] if f not in drop_fields]
				print "Generating conceptual space for",domain_name,"using",metadata['model_class']+"."
				cs = globals()[metadata['model_class']](domain_name, datapath,selected_hypers=metadata["best_hypers"])
				if "monary_type" in metadata.keys():
					cs.set_monary_type(metadata["monary_type"])

				# Inspect the model
				cs.stepwise_inspect(metadata, override_query, sample_size=sample_size, start_step=start_step)
			else:
				print "Need valid pretrain_start, pretrain_stop, time_stop and time_slice parameters to train a model."
		else:
			print "The database",dataset_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",dataset_name,"in the database."

# Measures the feature-based unexpectedness of a given saved model.
def feature_unexpectedness(domain_name, eval_name, override_dataset_name=None, datapath="data/", steps_from_file=True, override_query = {}, drop_fields = [], bypass_mongo=False, start=0, stop=-1):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	dataset_name = domain_name
	if override_dataset_name is not None:
		dataset_name = override_dataset_name
	metadata = db.datasets.find_one({"name": dataset_name})

	if metadata is not None:
		#Use the previously-trained expectation model to investigate changes in the dataset.
		if steps_from_file:
			with open(os.path.join(datapath,"steps.txt")) as f:
				metadata["steps"] = json.load(f)
		unescape_mongo(metadata)
		if "best_hypers" in metadata.keys():
			original_fields_x = metadata["fields_x"]
			original_fields_y = metadata["fields_y"]
			metadata["fields_x"] = [f for f in metadata['fields_x'] if f not in drop_fields]
			metadata["fields_y"] = [f for f in metadata['fields_y'] if f not in drop_fields]
			print "Generating conceptual space for",domain_name,"using",metadata['model_class']+"."
			cs = globals()[metadata['model_class']](domain_name, datapath,selected_hypers=metadata["best_hypers"])
			if "monary_type" in metadata.keys():
				cs.set_monary_type(metadata["monary_type"])

			cs.load(metadata["steps"][0]["model_files"][0])
			cs.init_model_functions()
			# Inspect the model
			#cs.featurewise_inspect(metadata, override_query, n_iter=10)
			#cs.featurewise_inspect(metadata, override_query, n_iter=20)
			#cs.co_occurence_matrix(metadata,outpath="co_occurence_matrix.csv")
			#cs.featurewise_inspect(metadata, override_query, n_iter=20)
			cs.evaluate_data_surprise(metadata, eval_name, override_query, start=start,stop=stop)

			#recipe_list = ["Beef Pita, Greek Style"]
			#cs.generate_from_reformulation(metadata,eval_name,recipe_list)
		else:
			print "The database",dataset_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",dataset_name,"in the database."

if __name__ == "__main__":
	print "Started creeval."
	parser = argparse.ArgumentParser(description='Use this to run creeval and discovery temporal unexpectedness at the domain level')
	parser.add_argument('dataset',help="Name of the dataset to work with")
	parser.add_argument("--override_dataset",help="Dataset entry to use (if not the same as the dataset itself)", default=None)
	parser.add_argument('-e','--epochs',help="How many epochs to train for",required=False, type=int, default=None)
	parser.add_argument('-m','--mode',choices=["fit_hypers",'train_exp',"unex", "unex_fwise"], help="Run the creeval step you're interested in.",required=False)
	parser.add_argument('-i','--sample_limit',help="How many samples to pull from the dataset (every mode)",type=int, required=False,default=0)
	parser.add_argument('-b','--bypass_mongo',help="Bypass mongo (load the data through the yaml or elsewhere)",action='store_true')

	#Args for fit_hypers
	parser.add_argument('-l','--look_back',help="How many steps to look back for determining spearmint stall",type=int, required=False, default=3)
	parser.add_argument('-s','--stop_thresh',help="The epsilon for spearmint stalling",type=float, required=False, default=0.1)

	#Args for train_exp
	parser.add_argument("-n","--exp_name",help="The name under which to train the expectation model",required=False,default="expectations")

	#Args for train_exp and unex
	parser.add_argument("-t","--pretrain_start",help="The timestamp at which to start pre-training",required=False,type=float,default=None)
	parser.add_argument("-o","--pretrain_stop",help="The timestamp at which to stop pre-training",required=False,type=float,default=None)
	parser.add_argument("-r","--train_stop",help="The timestamp at which to stop training expectations (training automatically starts after pre-training)",required=False,type=float,default=None)
	parser.add_argument("-c","--time_slice",help="The amount of time between each expectation-training step.",required=False,type=float,default=None)

	#Args for unex
	parser.add_argument("-p","--starting_step",help="The step at which to start calculating unexpectedness.",required=False,default=0, type=int)

	#Args for unex_fwise
	parser.add_argument("-k","--skip",help="The number of examples to skip when evaluating the data", required=False,type=int,default=0)
	parser.add_argument("-v","--eval_name",help="The name of the collection to which to save surprise evaluations", required=False,type=str,default=0)


	parser.add_argument("-d","--condition",help="The condition (used for DCC2016 paper: 0=all, 1=onlysugar, 2=nosugar", required=False,type=int,default=0)

	args = parser.parse_args()
	collname = args.dataset
	query = {}
	if args.condition == 0:
		override_query = {}
	elif args.condition == 1:
		override_query = {"$or" : [{"i_sugar": {"$exists":True}}, {"i_brown_sugar": {"$exists":True}}]}
	elif args.condition == 2:
		override_query = {"$and" : [{"i_sugar": {"$exists":False}}, {"i_brown_sugar": {"$exists":False}}]}
	ignore_fields = []
	if "ebird" in args.dataset:
		#ten_species = ['Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Turdus_migratorius', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens']
		#collname = "ebird_top10_2008_2012"
		#ignore_fields = ["LATITUDE","LONGITUDE",'Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens']
		ignore_fields = ['Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens']

		species = ['Turdus_migratorius']
		for s in species:
			query[s] = {"$gt": 0}
		override_query["$or"] = [{k:query[k]} for k in query.keys()]


	if args.mode == "fit_hypers":
		print "Initiating hyperparameter fit on",collname+"."
		fit_hypers(collname, override_dataset_name=args.override_dataset, spearmint_params = {"look_back": args.look_back,"stop_thresh": args.stop_thresh, 'datapath': os.path.join("data/",collname)},override_query=override_query, drop_fields = ignore_fields, sample_limit=args.sample_limit, training_epochs = args.epochs, bypass_mongo=args.bypass_mongo)
	elif args.mode == "train_exp":
		print "Initiating expectation training on",args.exp_name+"."
		p = os.path.join("data/",collname,args.exp_name)
		if not os.path.exists(p):
			os.makedirs(p)
		if args.pretrain_start is not None:
			train_expectations(collname, override_dataset_name=args.override_dataset, pretrain_start = args.pretrain_start, pretrain_stop = args.pretrain_stop, train_stop = args.train_stop, time_slice = args.time_slice, datapath = p, override_query=override_query, drop_fields = ignore_fields, training_epochs = args.epochs, sample_limit=args.sample_limit, bypass_mongo=args.bypass_mongo)
		else:
			train_holistic_expectations(collname, override_dataset_name=args.override_dataset, datapath = p, override_query=override_query, drop_fields = ignore_fields, training_epochs = args.epochs, sample_limit=args.sample_limit, bypass_mongo=args.bypass_mongo)
	elif args.mode == "unex":
		print "Initiating trend unexpectedness evaluation of",args.exp_name+"."
		unexpectedness(collname, override_dataset_name=args.override_dataset, pretrain_start = args.pretrain_start, pretrain_stop = args.pretrain_stop, train_stop = args.train_stop, time_slice = args.time_slice, datapath = os.path.join("data/",collname,args.exp_name), override_query=override_query, drop_fields = ignore_fields, sample_size=args.sample_limit, start_step=args.starting_step, bypass_mongo=args.bypass_mongo)
	elif args.mode == "unex_fwise":
		print "Initiating featurewise unexpectedness evaluation of",args.exp_name+"."
		feature_unexpectedness(collname, args.eval_name, override_dataset_name=args.override_dataset, datapath = os.path.join("data/",collname,args.exp_name), override_query=override_query, drop_fields = ignore_fields, bypass_mongo=args.bypass_mongo, start=args.skip,stop=args.sample_limit)
