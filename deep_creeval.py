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
def fit_hypers(domain_name, spearmint_params = {"look_back": 5,"stop_thresh": 0.05, 'datapath': "data/"}, hypers_to_file=True, override_query = {}, drop_fields = [], sample_limit = 0, training_epochs = 10):

	spearmint_params["datapath"] = os.path.join(spearmint_params["datapath"],"hyper_fitting")

	# retrieve the metadata from the db
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

	if metadata is not None:
		unescape_mongo(metadata)

		cs = globals()[metadata['model_class']](domain_name, spearmint_params['datapath'], max_epochs = training_epochs)
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
		print "Could not find a record for the dataset",domain_name,"in the database."

# Builds the deep learning model over each timeslice of the data, saving successive states to the database
def train_expectations(domain_name,pretrain_start = None, pretrain_stop = None, train_stop = None, time_slice = None, datapath="data/", hypers_from_file=True, steps_to_file=True, override_query = {}, drop_fields = [], training_epochs = 10, sample_limit=0):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

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
			print "The database",domain_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",domain_name,"in the database."

# Measures the unexpectedness of a given saved model.
def unexpectedness(domain_name,pretrain_start = None, pretrain_stop = None, train_stop = None, time_slice = None, sample_size= 50000, datapath="data/", steps_from_file=True, override_query = {}, drop_fields = []):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

	if metadata is not None:
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
				# Inspect the model
				cs.stepwise_inspect(metadata, override_query, sample_size=sample_size)
			else:
				print "Need valid pretrain_start, pretrain_stop, time_stop and time_slice parameters to train a model."
		else:
			print "The database",domain_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",domain_name,"in the database."

if __name__ == "__main__":
	print "Started creeval."
	parser = argparse.ArgumentParser(description='Use this to run creeval and discovery temporal unexpectedness at the domain level')
	parser.add_argument('dataset',help="Name of the dataset to work with")
	parser.add_argument('-e','--epochs',help="How many epochs to train for",required=False, type=int, default=10)
	parser.add_argument('-m','--mode',choices=["fit_hypers",'train_exp',"unex"], help="Run the creeval step you're interested in.",required=False)
	parser.add_argument('-i','--sample_limit',help="How many samples to pull from the dataset (every mode)",type=int, required=False,default=0)

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

	args = parser.parse_args()
	collname = args.dataset
	#ten_species = ['Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Turdus_migratorius', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens']
	#collname = "ebird_top10_2008_2012"
	ignore_fields = ["LATITUDE","LONGITUDE",'Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens']

	species = ['Turdus_migratorius']
	query = {}
	for s in species:
		query[s] = {"$gt": 0}
	override_query = {}
	override_query["$or"] = [{k:query[k]} for k in query.keys()]

	if args.mode == "fit_hypers":
		print "Initiating hyperparameter fit on",collname+"."
		fit_hypers(collname,spearmint_params = {"look_back": args.look_back,"stop_thresh": args.stop_thresh, 'datapath': os.path.join("data/",collname)},override_query=override_query, drop_fields = ignore_fields, sample_limit=args.sample_limit, training_epochs = args.epochs)
	elif args.mode == "train_exp":
		print "Initiating expectation training on",args.exp_name+"."
		p = os.path.join("data/",collname,args.exp_name)
		if not os.path.exists(p):
			os.makedirs(p)
		train_expectations(collname, pretrain_start = args.pretrain_start, pretrain_stop = args.pretrain_stop, train_stop = args.train_stop, time_slice = args.time_slice, datapath = p, override_query=override_query, drop_fields = ignore_fields, training_epochs = args.epochs, sample_limit=args.sample_limit)
	elif args.mode == "unex":
		print "Initiating unexpectedness evaluation of",args.exp_name+"."
		unexpectedness(collname, pretrain_start = args.pretrain_start, pretrain_stop = args.pretrain_stop, train_stop = args.train_stop, time_slice = args.time_slice, datapath = os.path.join("data/",collname,args.exp_name), override_query=override_query, drop_fields = ignore_fields, sample_size=args.sample_limit)
