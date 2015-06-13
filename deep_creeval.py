__author__ = 'kazjon'
from conceptual_space import *
import pymongo, argparse, csv, sys, os

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
def fit_hypers(domain_name, spearmint_params = {"look_back": 5,"stop_thresh": 0.05, 'datapath': "data/"}):

	# retrieve the metadata from the db
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

	if metadata is not None:
		unescape_mongo(metadata)

		cs = globals()[metadata['model_class']](domain_name, spearmint_params['datapath'])
		metadata['best_hypers'] =  cs.train_mongo(metadata['fields_x'], metadata['fields_y'], metadata['query'], spearmint_params['stop_thresh'], spearmint_params['look_back'])

		# Save the best hypers to the database
		if len(metadata['best_hypers'].keys()):
			escape_mongo(metadata)
			db.datasets.save(metadata)
		else:
			print "Spearmint hyperparameter optimisation failed, check logs for details."
	else:
		print "Could not find a record for the dataset",domain_name,"in the database."

# Builds the deep learning model over each timeslice of the data, saving successive states to the database
def train_expectations(domain_name,pretrain_start = None, pretrain_stop = None, train_stop = None, time_slice = None, datapath="data/"):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

	if metadata is not None:
		#Use the previously-fit hypers to train a single model with the provided timeslices (or ones from the data)
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
				cs = globals()[metadata['model_class']](domain_name, datapath,selected_hypers=metadata["best_hypers"])
				# Fit the initial model
				step_0 = cs.pretrain(metadata)
				#pickle.dump(pretrained_model,open("pretrain_test.pkl","wb"))
				#pretrained_model = pickle.load(open("pretrain_test.pkl","rb"))
				#print "loaded."
				step_0["start"] = metadata["pretrain_start"]
				step_0["stop"] =  metadata["pretrain_stop"]
				#step_0 = "pretrain_skipped"
				metadata['steps'] = [step_0]
				cs.stepwise_train(metadata)
				escape_mongo(metadata)
				db.datasets.save(metadata)
			else:
				print "Need valid pretrain_start, pretrain_stop, time_stop and time_slice parameters to train a model."
		else:
			print "The database",domain_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",domain_name,"in the database."

# Measures the unexpectedness of a given saved model.
def unexpectedness(domain_name,pretrain_start = None, pretrain_stop = None, train_stop = None, time_slice = None, datapath="data/"):
	#Pull best hypers out of the database
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": domain_name})

	if metadata is not None:
		#Use the previously-fit hypers to train a single model with the provided timeslices (or ones from the data)
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
				cs = globals()[metadata['model_class']](domain_name, datapath,selected_hypers=metadata["best_hypers"])
				# Fit the initial model
				cs.stepwise_inspect(metadata)
			else:
				print "Need valid pretrain_start, pretrain_stop, time_stop and time_slice parameters to train a model."
		else:
			print "The database",domain_name,"does not contain fitted hyperparameters.  Run fit_hypers() on it first."
	else:
		print "Could not find a record for the dataset",domain_name,"in the database."

def ebird_metadata_setup():
	cat_metadata = []
	num_metadata = [u'ASTER2011_DEM', u'CAUS_FIRST_AUTUMN_32F_EXTREME', u'CAUS_FIRST_AUTUMN_32F_MEAN', u'CAUS_FIRST_AUTUMN_32F_MEDIAN', u'CAUS_LAST_SPRING_32F_EXTREME', u'CAUS_LAST_SPRING_32F_MEAN', u'CAUS_LAST_SPRING_32F_MEDIAN', u'CAUS_PREC', u'CAUS_PREC01', u'CAUS_PREC02', u'CAUS_PREC03', u'CAUS_PREC04', u'CAUS_PREC05', u'CAUS_PREC06', u'CAUS_PREC07', u'CAUS_PREC08', u'CAUS_PREC09', u'CAUS_PREC10', u'CAUS_PREC11', u'CAUS_PREC12', u'CAUS_PREC13', u'CAUS_SNOW', u'CAUS_SNOW01', u'CAUS_SNOW02', u'CAUS_SNOW03', u'CAUS_SNOW04', u'CAUS_SNOW10', u'CAUS_SNOW11', u'CAUS_SNOW12', u'CAUS_TEMP_AVG', u'CAUS_TEMP_AVG01', u'CAUS_TEMP_AVG02', u'CAUS_TEMP_AVG03', u'CAUS_TEMP_AVG04', u'CAUS_TEMP_AVG05', u'CAUS_TEMP_AVG06', u'CAUS_TEMP_AVG07', u'CAUS_TEMP_AVG08', u'CAUS_TEMP_AVG09', u'CAUS_TEMP_AVG10', u'CAUS_TEMP_AVG11', u'CAUS_TEMP_AVG12', u'CAUS_TEMP_AVG13', u'CAUS_TEMP_MAX', u'CAUS_TEMP_MAX01', u'CAUS_TEMP_MAX02', u'CAUS_TEMP_MAX03', u'CAUS_TEMP_MAX04', u'CAUS_TEMP_MAX05', u'CAUS_TEMP_MAX06', u'CAUS_TEMP_MAX07', u'CAUS_TEMP_MAX08', u'CAUS_TEMP_MAX09', u'CAUS_TEMP_MAX10', u'CAUS_TEMP_MAX11', u'CAUS_TEMP_MAX12', u'CAUS_TEMP_MAX13', u'CAUS_TEMP_MIN', u'CAUS_TEMP_MIN01', u'CAUS_TEMP_MIN02', u'CAUS_TEMP_MIN03', u'CAUS_TEMP_MIN04', u'CAUS_TEMP_MIN05', u'CAUS_TEMP_MIN06', u'CAUS_TEMP_MIN07', u'CAUS_TEMP_MIN08', u'CAUS_TEMP_MIN09', u'CAUS_TEMP_MIN10', u'CAUS_TEMP_MIN11', u'CAUS_TEMP_MIN12', u'CAUS_TEMP_MIN13', u'DAY', u'DIST_FROM_FLOWING_BRACKISH', u'DIST_FROM_FLOWING_FRESH', u'DIST_FROM_STANDING_BRACKISH', u'DIST_FROM_STANDING_FRESH', u'DIST_FROM_WET_VEG_BRACKISH', u'DIST_FROM_WET_VEG_FRESH', u'DIST_IN_FLOWING_BRACKISH', u'DIST_IN_FLOWING_FRESH', u'DIST_IN_STANDING_BRACKISH', u'DIST_IN_STANDING_FRESH', u'DIST_IN_WET_VEG_BRACKISH', u'DIST_IN_WET_VEG_FRESH', u'EFFORT_AREA_HA', u'EFFORT_DISTANCE_KM', u'EFFORT_HRS', u'ELEV_GT', u'ELEV_NED', u'HOUSING_DENSITY', u'HOUSING_PERCENT_VACANT', u'LATITUDE', u'LONGITUDE', u'NLCD01_CANOPYMEAN_RAD75', u'NLCD01_CANOPYMEAN_RAD750', u'NLCD01_CANOPYMEAN_RAD7500', u'NLCD01_IMPERVMEAN_RAD75', u'NLCD01_IMPERVMEAN_RAD750', u'NLCD01_IMPERVMEAN_RAD7500', u'NLCD2001_FS_C11_7500_ED', u'NLCD2001_FS_C11_7500_LPI', u'NLCD2001_FS_C11_7500_PD', u'NLCD2001_FS_C11_7500_PLAND', u'NLCD2001_FS_C11_750_ED', u'NLCD2001_FS_C11_750_LPI', u'NLCD2001_FS_C11_750_PD', u'NLCD2001_FS_C11_750_PLAND', u'NLCD2001_FS_C11_75_ED', u'NLCD2001_FS_C11_75_LPI', u'NLCD2001_FS_C11_75_PD', u'NLCD2001_FS_C11_75_PLAND', u'NLCD2001_FS_C12_7500_ED', u'NLCD2001_FS_C12_7500_LPI', u'NLCD2001_FS_C12_7500_PD', u'NLCD2001_FS_C12_7500_PLAND', u'NLCD2001_FS_C12_750_ED', u'NLCD2001_FS_C12_750_LPI', u'NLCD2001_FS_C12_750_PD', u'NLCD2001_FS_C12_750_PLAND', u'NLCD2001_FS_C12_75_ED', u'NLCD2001_FS_C12_75_LPI', u'NLCD2001_FS_C12_75_PD', u'NLCD2001_FS_C12_75_PLAND', u'NLCD2001_FS_C21_7500_ED', u'NLCD2001_FS_C21_7500_LPI', u'NLCD2001_FS_C21_7500_PD', u'NLCD2001_FS_C21_7500_PLAND', u'NLCD2001_FS_C21_750_ED', u'NLCD2001_FS_C21_750_LPI', u'NLCD2001_FS_C21_750_PD', u'NLCD2001_FS_C21_750_PLAND', u'NLCD2001_FS_C21_75_ED', u'NLCD2001_FS_C21_75_LPI', u'NLCD2001_FS_C21_75_PD', u'NLCD2001_FS_C21_75_PLAND', u'NLCD2001_FS_C22_7500_ED', u'NLCD2001_FS_C22_7500_LPI', u'NLCD2001_FS_C22_7500_PD', u'NLCD2001_FS_C22_7500_PLAND', u'NLCD2001_FS_C22_750_ED', u'NLCD2001_FS_C22_750_LPI', u'NLCD2001_FS_C22_750_PD', u'NLCD2001_FS_C22_750_PLAND', u'NLCD2001_FS_C22_75_ED', u'NLCD2001_FS_C22_75_LPI', u'NLCD2001_FS_C22_75_PD', u'NLCD2001_FS_C22_75_PLAND', u'NLCD2001_FS_C23_7500_ED', u'NLCD2001_FS_C23_7500_LPI', u'NLCD2001_FS_C23_7500_PD', u'NLCD2001_FS_C23_7500_PLAND', u'NLCD2001_FS_C23_750_ED', u'NLCD2001_FS_C23_750_LPI', u'NLCD2001_FS_C23_750_PD', u'NLCD2001_FS_C23_750_PLAND', u'NLCD2001_FS_C23_75_ED', u'NLCD2001_FS_C23_75_LPI', u'NLCD2001_FS_C23_75_PD', u'NLCD2001_FS_C23_75_PLAND', u'NLCD2001_FS_C24_7500_ED', u'NLCD2001_FS_C24_7500_LPI', u'NLCD2001_FS_C24_7500_PD', u'NLCD2001_FS_C24_7500_PLAND', u'NLCD2001_FS_C24_750_ED', u'NLCD2001_FS_C24_750_LPI', u'NLCD2001_FS_C24_750_PD', u'NLCD2001_FS_C24_750_PLAND', u'NLCD2001_FS_C24_75_ED', u'NLCD2001_FS_C24_75_LPI', u'NLCD2001_FS_C24_75_PD', u'NLCD2001_FS_C24_75_PLAND', u'NLCD2001_FS_C31_7500_ED', u'NLCD2001_FS_C31_7500_LPI', u'NLCD2001_FS_C31_7500_PD', u'NLCD2001_FS_C31_7500_PLAND', u'NLCD2001_FS_C31_750_ED', u'NLCD2001_FS_C31_750_LPI', u'NLCD2001_FS_C31_750_PD', u'NLCD2001_FS_C31_750_PLAND', u'NLCD2001_FS_C31_75_ED', u'NLCD2001_FS_C31_75_LPI', u'NLCD2001_FS_C31_75_PD', u'NLCD2001_FS_C31_75_PLAND', u'NLCD2001_FS_C41_7500_ED', u'NLCD2001_FS_C41_7500_LPI', u'NLCD2001_FS_C41_7500_PD', u'NLCD2001_FS_C41_7500_PLAND', u'NLCD2001_FS_C41_750_ED', u'NLCD2001_FS_C41_750_LPI', u'NLCD2001_FS_C41_750_PD', u'NLCD2001_FS_C41_750_PLAND', u'NLCD2001_FS_C41_75_ED', u'NLCD2001_FS_C41_75_LPI', u'NLCD2001_FS_C41_75_PD', u'NLCD2001_FS_C41_75_PLAND', u'NLCD2001_FS_C42_7500_ED', u'NLCD2001_FS_C42_7500_LPI', u'NLCD2001_FS_C42_7500_PD', u'NLCD2001_FS_C42_7500_PLAND', u'NLCD2001_FS_C42_750_ED', u'NLCD2001_FS_C42_750_LPI', u'NLCD2001_FS_C42_750_PD', u'NLCD2001_FS_C42_750_PLAND', u'NLCD2001_FS_C42_75_ED', u'NLCD2001_FS_C42_75_LPI', u'NLCD2001_FS_C42_75_PD', u'NLCD2001_FS_C42_75_PLAND', u'NLCD2001_FS_C43_7500_ED', u'NLCD2001_FS_C43_7500_LPI', u'NLCD2001_FS_C43_7500_PD', u'NLCD2001_FS_C43_7500_PLAND', u'NLCD2001_FS_C43_750_ED', u'NLCD2001_FS_C43_750_LPI', u'NLCD2001_FS_C43_750_PD', u'NLCD2001_FS_C43_750_PLAND', u'NLCD2001_FS_C43_75_ED', u'NLCD2001_FS_C43_75_LPI', u'NLCD2001_FS_C43_75_PD', u'NLCD2001_FS_C43_75_PLAND', u'NLCD2001_FS_C52_7500_ED', u'NLCD2001_FS_C52_7500_LPI', u'NLCD2001_FS_C52_7500_PD', u'NLCD2001_FS_C52_7500_PLAND', u'NLCD2001_FS_C52_750_ED', u'NLCD2001_FS_C52_750_LPI', u'NLCD2001_FS_C52_750_PD', u'NLCD2001_FS_C52_750_PLAND', u'NLCD2001_FS_C52_75_ED', u'NLCD2001_FS_C52_75_LPI', u'NLCD2001_FS_C52_75_PD', u'NLCD2001_FS_C52_75_PLAND', u'NLCD2001_FS_C71_7500_ED', u'NLCD2001_FS_C71_7500_LPI', u'NLCD2001_FS_C71_7500_PD', u'NLCD2001_FS_C71_7500_PLAND', u'NLCD2001_FS_C71_750_ED', u'NLCD2001_FS_C71_750_LPI', u'NLCD2001_FS_C71_750_PD', u'NLCD2001_FS_C71_750_PLAND', u'NLCD2001_FS_C71_75_ED', u'NLCD2001_FS_C71_75_LPI', u'NLCD2001_FS_C71_75_PD', u'NLCD2001_FS_C71_75_PLAND', u'NLCD2001_FS_C81_7500_ED', u'NLCD2001_FS_C81_7500_LPI', u'NLCD2001_FS_C81_7500_PD', u'NLCD2001_FS_C81_7500_PLAND', u'NLCD2001_FS_C81_750_ED', u'NLCD2001_FS_C81_750_LPI', u'NLCD2001_FS_C81_750_PD', u'NLCD2001_FS_C81_750_PLAND', u'NLCD2001_FS_C81_75_ED', u'NLCD2001_FS_C81_75_LPI', u'NLCD2001_FS_C81_75_PD', u'NLCD2001_FS_C81_75_PLAND', u'NLCD2001_FS_C82_7500_ED', u'NLCD2001_FS_C82_7500_LPI', u'NLCD2001_FS_C82_7500_PD', u'NLCD2001_FS_C82_7500_PLAND', u'NLCD2001_FS_C82_750_ED', u'NLCD2001_FS_C82_750_LPI', u'NLCD2001_FS_C82_750_PD', u'NLCD2001_FS_C82_750_PLAND', u'NLCD2001_FS_C82_75_ED', u'NLCD2001_FS_C82_75_LPI', u'NLCD2001_FS_C82_75_PD', u'NLCD2001_FS_C82_75_PLAND', u'NLCD2001_FS_C90_7500_ED', u'NLCD2001_FS_C90_7500_LPI', u'NLCD2001_FS_C90_7500_PD', u'NLCD2001_FS_C90_7500_PLAND', u'NLCD2001_FS_C90_750_ED', u'NLCD2001_FS_C90_750_LPI', u'NLCD2001_FS_C90_750_PD', u'NLCD2001_FS_C90_750_PLAND', u'NLCD2001_FS_C90_75_ED', u'NLCD2001_FS_C90_75_LPI', u'NLCD2001_FS_C90_75_PD', u'NLCD2001_FS_C90_75_PLAND', u'NLCD2001_FS_C95_7500_ED', u'NLCD2001_FS_C95_7500_LPI', u'NLCD2001_FS_C95_7500_PD', u'NLCD2001_FS_C95_7500_PLAND', u'NLCD2001_FS_C95_750_ED', u'NLCD2001_FS_C95_750_LPI', u'NLCD2001_FS_C95_750_PD', u'NLCD2001_FS_C95_750_PLAND', u'NLCD2001_FS_C95_75_ED', u'NLCD2001_FS_C95_75_LPI', u'NLCD2001_FS_C95_75_PD', u'NLCD2001_FS_C95_75_PLAND', u'NLCD2001_FS_L_7500_ED', u'NLCD2001_FS_L_7500_LPI', u'NLCD2001_FS_L_7500_PD', u'NLCD2001_FS_L_750_ED', u'NLCD2001_FS_L_750_LPI', u'NLCD2001_FS_L_750_PD', u'NLCD2001_FS_L_75_ED', u'NLCD2001_FS_L_75_LPI', u'NLCD2001_FS_L_75_PD', u'NLCD2006_FS_C11_7500_ED', u'NLCD2006_FS_C11_7500_LPI', u'NLCD2006_FS_C11_7500_PD', u'NLCD2006_FS_C11_7500_PLAND', u'NLCD2006_FS_C11_750_ED', u'NLCD2006_FS_C11_750_LPI', u'NLCD2006_FS_C11_750_PD', u'NLCD2006_FS_C11_750_PLAND', u'NLCD2006_FS_C11_75_ED', u'NLCD2006_FS_C11_75_LPI', u'NLCD2006_FS_C11_75_PD', u'NLCD2006_FS_C11_75_PLAND', u'NLCD2006_FS_C12_7500_ED', u'NLCD2006_FS_C12_7500_LPI', u'NLCD2006_FS_C12_7500_PD', u'NLCD2006_FS_C12_7500_PLAND', u'NLCD2006_FS_C12_750_ED', u'NLCD2006_FS_C12_750_LPI', u'NLCD2006_FS_C12_750_PD', u'NLCD2006_FS_C12_750_PLAND', u'NLCD2006_FS_C12_75_ED', u'NLCD2006_FS_C12_75_LPI', u'NLCD2006_FS_C12_75_PD', u'NLCD2006_FS_C12_75_PLAND', u'NLCD2006_FS_C21_7500_ED', u'NLCD2006_FS_C21_7500_LPI', u'NLCD2006_FS_C21_7500_PD', u'NLCD2006_FS_C21_7500_PLAND', u'NLCD2006_FS_C21_750_ED', u'NLCD2006_FS_C21_750_LPI', u'NLCD2006_FS_C21_750_PD', u'NLCD2006_FS_C21_750_PLAND', u'NLCD2006_FS_C21_75_ED', u'NLCD2006_FS_C21_75_LPI', u'NLCD2006_FS_C21_75_PD', u'NLCD2006_FS_C21_75_PLAND', u'NLCD2006_FS_C22_7500_ED', u'NLCD2006_FS_C22_7500_LPI', u'NLCD2006_FS_C22_7500_PD', u'NLCD2006_FS_C22_7500_PLAND', u'NLCD2006_FS_C22_750_ED', u'NLCD2006_FS_C22_750_LPI', u'NLCD2006_FS_C22_750_PD', u'NLCD2006_FS_C22_750_PLAND', u'NLCD2006_FS_C22_75_ED', u'NLCD2006_FS_C22_75_LPI', u'NLCD2006_FS_C22_75_PD', u'NLCD2006_FS_C22_75_PLAND', u'NLCD2006_FS_C23_7500_ED', u'NLCD2006_FS_C23_7500_LPI', u'NLCD2006_FS_C23_7500_PD', u'NLCD2006_FS_C23_7500_PLAND', u'NLCD2006_FS_C23_750_ED', u'NLCD2006_FS_C23_750_LPI', u'NLCD2006_FS_C23_750_PD', u'NLCD2006_FS_C23_750_PLAND', u'NLCD2006_FS_C23_75_ED', u'NLCD2006_FS_C23_75_LPI', u'NLCD2006_FS_C23_75_PD', u'NLCD2006_FS_C23_75_PLAND', u'NLCD2006_FS_C24_7500_ED', u'NLCD2006_FS_C24_7500_LPI', u'NLCD2006_FS_C24_7500_PD', u'NLCD2006_FS_C24_7500_PLAND', u'NLCD2006_FS_C24_750_ED', u'NLCD2006_FS_C24_750_LPI', u'NLCD2006_FS_C24_750_PD', u'NLCD2006_FS_C24_750_PLAND', u'NLCD2006_FS_C24_75_ED', u'NLCD2006_FS_C24_75_LPI', u'NLCD2006_FS_C24_75_PD', u'NLCD2006_FS_C24_75_PLAND', u'NLCD2006_FS_C31_7500_ED', u'NLCD2006_FS_C31_7500_LPI', u'NLCD2006_FS_C31_7500_PD', u'NLCD2006_FS_C31_7500_PLAND', u'NLCD2006_FS_C31_750_ED', u'NLCD2006_FS_C31_750_LPI', u'NLCD2006_FS_C31_750_PD', u'NLCD2006_FS_C31_750_PLAND', u'NLCD2006_FS_C31_75_ED', u'NLCD2006_FS_C31_75_LPI', u'NLCD2006_FS_C31_75_PD', u'NLCD2006_FS_C31_75_PLAND', u'NLCD2006_FS_C41_7500_ED', u'NLCD2006_FS_C41_7500_LPI', u'NLCD2006_FS_C41_7500_PD', u'NLCD2006_FS_C41_7500_PLAND', u'NLCD2006_FS_C41_750_ED', u'NLCD2006_FS_C41_750_LPI', u'NLCD2006_FS_C41_750_PD', u'NLCD2006_FS_C41_750_PLAND', u'NLCD2006_FS_C41_75_ED', u'NLCD2006_FS_C41_75_LPI', u'NLCD2006_FS_C41_75_PD', u'NLCD2006_FS_C41_75_PLAND', u'NLCD2006_FS_C42_7500_ED', u'NLCD2006_FS_C42_7500_LPI', u'NLCD2006_FS_C42_7500_PD', u'NLCD2006_FS_C42_7500_PLAND', u'NLCD2006_FS_C42_750_ED', u'NLCD2006_FS_C42_750_LPI', u'NLCD2006_FS_C42_750_PD', u'NLCD2006_FS_C42_750_PLAND', u'NLCD2006_FS_C42_75_ED', u'NLCD2006_FS_C42_75_LPI', u'NLCD2006_FS_C42_75_PD', u'NLCD2006_FS_C42_75_PLAND', u'NLCD2006_FS_C43_7500_ED', u'NLCD2006_FS_C43_7500_LPI', u'NLCD2006_FS_C43_7500_PD', u'NLCD2006_FS_C43_7500_PLAND', u'NLCD2006_FS_C43_750_ED', u'NLCD2006_FS_C43_750_LPI', u'NLCD2006_FS_C43_750_PD', u'NLCD2006_FS_C43_750_PLAND', u'NLCD2006_FS_C43_75_ED', u'NLCD2006_FS_C43_75_LPI', u'NLCD2006_FS_C43_75_PD', u'NLCD2006_FS_C43_75_PLAND', u'NLCD2006_FS_C52_7500_ED', u'NLCD2006_FS_C52_7500_LPI', u'NLCD2006_FS_C52_7500_PD', u'NLCD2006_FS_C52_7500_PLAND', u'NLCD2006_FS_C52_750_ED', u'NLCD2006_FS_C52_750_LPI', u'NLCD2006_FS_C52_750_PD', u'NLCD2006_FS_C52_750_PLAND', u'NLCD2006_FS_C52_75_ED', u'NLCD2006_FS_C52_75_LPI', u'NLCD2006_FS_C52_75_PD', u'NLCD2006_FS_C52_75_PLAND', u'NLCD2006_FS_C71_7500_ED', u'NLCD2006_FS_C71_7500_LPI', u'NLCD2006_FS_C71_7500_PD', u'NLCD2006_FS_C71_7500_PLAND', u'NLCD2006_FS_C71_750_ED', u'NLCD2006_FS_C71_750_LPI', u'NLCD2006_FS_C71_750_PD', u'NLCD2006_FS_C71_750_PLAND', u'NLCD2006_FS_C71_75_ED', u'NLCD2006_FS_C71_75_LPI', u'NLCD2006_FS_C71_75_PD', u'NLCD2006_FS_C71_75_PLAND', u'NLCD2006_FS_C81_7500_ED', u'NLCD2006_FS_C81_7500_LPI', u'NLCD2006_FS_C81_7500_PD', u'NLCD2006_FS_C81_7500_PLAND', u'NLCD2006_FS_C81_750_ED', u'NLCD2006_FS_C81_750_LPI', u'NLCD2006_FS_C81_750_PD', u'NLCD2006_FS_C81_750_PLAND', u'NLCD2006_FS_C81_75_ED', u'NLCD2006_FS_C81_75_LPI', u'NLCD2006_FS_C81_75_PD', u'NLCD2006_FS_C81_75_PLAND', u'NLCD2006_FS_C82_7500_ED', u'NLCD2006_FS_C82_7500_LPI', u'NLCD2006_FS_C82_7500_PD', u'NLCD2006_FS_C82_7500_PLAND', u'NLCD2006_FS_C82_750_ED', u'NLCD2006_FS_C82_750_LPI', u'NLCD2006_FS_C82_750_PD', u'NLCD2006_FS_C82_750_PLAND', u'NLCD2006_FS_C82_75_ED', u'NLCD2006_FS_C82_75_LPI', u'NLCD2006_FS_C82_75_PD', u'NLCD2006_FS_C82_75_PLAND', u'NLCD2006_FS_C90_7500_ED', u'NLCD2006_FS_C90_7500_LPI', u'NLCD2006_FS_C90_7500_PD', u'NLCD2006_FS_C90_7500_PLAND', u'NLCD2006_FS_C90_750_ED', u'NLCD2006_FS_C90_750_LPI', u'NLCD2006_FS_C90_750_PD', u'NLCD2006_FS_C90_750_PLAND', u'NLCD2006_FS_C90_75_ED', u'NLCD2006_FS_C90_75_LPI', u'NLCD2006_FS_C90_75_PD', u'NLCD2006_FS_C90_75_PLAND', u'NLCD2006_FS_C95_7500_ED', u'NLCD2006_FS_C95_7500_LPI', u'NLCD2006_FS_C95_7500_PD', u'NLCD2006_FS_C95_7500_PLAND', u'NLCD2006_FS_C95_750_ED', u'NLCD2006_FS_C95_750_LPI', u'NLCD2006_FS_C95_750_PD', u'NLCD2006_FS_C95_750_PLAND', u'NLCD2006_FS_C95_75_ED', u'NLCD2006_FS_C95_75_LPI', u'NLCD2006_FS_C95_75_PD', u'NLCD2006_FS_C95_75_PLAND', u'NLCD2006_FS_L_7500_ED', u'NLCD2006_FS_L_7500_LPI', u'NLCD2006_FS_L_7500_PD', u'NLCD2006_FS_L_750_ED', u'NLCD2006_FS_L_750_LPI', u'NLCD2006_FS_L_750_PD', u'NLCD2006_FS_L_75_ED', u'NLCD2006_FS_L_75_LPI', u'NLCD2006_FS_L_75_PD', u'POP00_SQMI', u'TIME', u'UMD2011_FS_C0_1500_ED', u'UMD2011_FS_C0_1500_LPI', u'UMD2011_FS_C0_1500_PD', u'UMD2011_FS_C0_1500_PLAND', u'UMD2011_FS_C10_1500_ED', u'UMD2011_FS_C10_1500_LPI', u'UMD2011_FS_C10_1500_PD', u'UMD2011_FS_C10_1500_PLAND', u'UMD2011_FS_C12_1500_ED', u'UMD2011_FS_C12_1500_LPI', u'UMD2011_FS_C12_1500_PD', u'UMD2011_FS_C12_1500_PLAND', u'UMD2011_FS_C13_1500_ED', u'UMD2011_FS_C13_1500_LPI', u'UMD2011_FS_C13_1500_PD', u'UMD2011_FS_C13_1500_PLAND', u'UMD2011_FS_C16_1500_ED', u'UMD2011_FS_C16_1500_LPI', u'UMD2011_FS_C16_1500_PD', u'UMD2011_FS_C16_1500_PLAND', u'UMD2011_FS_C1_1500_ED', u'UMD2011_FS_C1_1500_LPI', u'UMD2011_FS_C1_1500_PD', u'UMD2011_FS_C1_1500_PLAND', u'UMD2011_FS_C2_1500_ED', u'UMD2011_FS_C2_1500_LPI', u'UMD2011_FS_C2_1500_PD', u'UMD2011_FS_C2_1500_PLAND', u'UMD2011_FS_C3_1500_ED', u'UMD2011_FS_C3_1500_LPI', u'UMD2011_FS_C3_1500_PD', u'UMD2011_FS_C3_1500_PLAND', u'UMD2011_FS_C4_1500_ED', u'UMD2011_FS_C4_1500_LPI', u'UMD2011_FS_C4_1500_PD', u'UMD2011_FS_C4_1500_PLAND', u'UMD2011_FS_C5_1500_ED', u'UMD2011_FS_C5_1500_LPI', u'UMD2011_FS_C5_1500_PD', u'UMD2011_FS_C5_1500_PLAND', u'UMD2011_FS_C6_1500_ED', u'UMD2011_FS_C6_1500_LPI', u'UMD2011_FS_C6_1500_PD', u'UMD2011_FS_C6_1500_PLAND', u'UMD2011_FS_C7_1500_ED', u'UMD2011_FS_C7_1500_LPI', u'UMD2011_FS_C7_1500_PD', u'UMD2011_FS_C7_1500_PLAND', u'UMD2011_FS_C8_1500_ED', u'UMD2011_FS_C8_1500_LPI', u'UMD2011_FS_C8_1500_PD', u'UMD2011_FS_C8_1500_PLAND', u'UMD2011_FS_C9_1500_ED', u'UMD2011_FS_C9_1500_LPI', u'UMD2011_FS_C9_1500_PD', u'UMD2011_FS_C9_1500_PLAND', u'UMD2011_FS_L_1500_ED', u'UMD2011_FS_L_1500_LPI', u'UMD2011_FS_L_1500_PD', u'UMD2011_LANDCOVER', u'YEAR']
	excluded_metadata = [u'BAILEY_ECOREGION', u'BCR',  u'COUNT_TYPE', u'OMERNIK_L3_ECOREGION', u'MONTH', u'COUNTRY',u'STATE_PROVINCE', u'NUMBER_OBSERVERS', u'OBSERVER_ID', u'PRIMARY_CHECKLIST_FLAG', u'GROUP_ID',  u'SUBNATIONAL2_CODE']
	metadata_excluded_prefixes = [u"NLCD",u"UMD", u"DIST_IN"]
	num_metadata = [s for s in num_metadata if not any([s.startswith(x) for x in metadata_excluded_prefixes])]
	cat_metadata = [s for s in cat_metadata if not any([s.startswith(x) for x in metadata_excluded_prefixes])]

	ten_species = ['Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Turdus_migratorius', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens']
	hundred_species = ['Zenaida_macroura', 'Corvus_brachyrhynchos', 'Cardinalis_cardinalis', 'Turdus_migratorius', 'Cyanocitta_cristata', 'Spinus_tristis', 'Sturnus_vulgaris', 'Melospiza_melodia', 'Agelaius_phoeniceus', 'Picoides_pubescens', 'Anas_platyrhynchos', 'Haemorhous_mexicanus', 'Branta_canadensis', 'Passer_domesticus', 'Poecile_atricapillus', 'Melanerpes_carolinus', 'Baeolophus_bicolor', 'Cathartes_aura', 'Ardea_herodias', 'Mimus_polyglottos', 'Sitta_carolinensis', 'Quiscalus_quiscula', 'Thryothorus_ludovicianus', 'Buteo_jamaicensis', 'Larus_delawarensis', 'Charadrius_vociferus', 'Colaptes_auratus', 'Phalacrocorax_auritus', 'Poecile_carolinensis', 'Columba_livia', 'Hirundo_rustica', 'Molothrus_ater', 'Junco_hyemalis', 'Zonotrichia_albicollis', 'Dumetella_carolinensis', 'Spizella_passerina', 'Geothlypis_trichas', 'Tachycineta_bicolor', 'Ardea_alba', 'Sialia_sialis', 'Megaceryle_alcyon', 'Sayornis_phoebe', 'Setophaga_coronata', 'Bombycilla_cedrorum', 'Fulica_americana', 'Regulus_calendula', 'Picoides_villosus', 'Troglodytes_aedon', 'Pipilo_erythrophthalmus', 'Falco_sparverius', 'Podilymbus_podiceps', 'Corvus_corax', 'Larus_argentatus', 'Zonotrichia_leucophrys', 'Polioptila_caerulea', 'Aix_sponsa', 'Setophaga_petechia', 'Pandion_haliaetus', 'Vireo_olivaceus', 'Dryocopus_pileatus', 'Chaetura_pelagica', 'Haliaeetus_leucocephalus', 'Passerina_cyanea', 'Buteo_lineatus', 'Egretta_thula', 'Bucephala_albeola', 'Archilochus_colubris', 'Accipiter_cooperii', 'Circus_cyaneus', 'Sitta_canadensis', 'Toxostoma_rufum', 'Anas_clypeata', 'Anas_strepera', 'Passerculus_sandwichensis', 'Tyrannus_tyrannus', 'Coragyps_atratus', 'Calypte_anna', 'Sayornis_nigricans', 'Pipilo_maculatus', 'Spizella_pusilla', 'Icterus_galbula', 'Myiarchus_crinitus', 'Oxyura_jamaicensis', 'Spinus_psaltria', 'Contopus_virens', 'Stelgidopteryx_serripennis', 'Butorides_virescens', 'Melospiza_georgiana', 'Regulus_satrapa', 'Corvus_ossifragus', 'Leucophaeus_atricilla', 'Catharus_guttatus', 'Anas_americana', 'Spinus_pinus', 'Tringa_melanoleuca', 'Streptopelia_decaocto', 'Aythya_collaris', 'Setophaga_ruticilla', 'Larus_marinus', 'Actitis_macularius', 'Anas_crecca']

	species = ['Turdus_migratorius']

	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": "ebird"})
	metadata["fields_y"] = species
	metadata["fields_x"] = num_metadata+cat_metadata

	query = {}
	for s in species:
		query[s] = {"__gt": 0}
	metadata["query"] = {}
	metadata["query"]["__or"] = [{k:query[k]} for k in query.keys()]
	pprint.pprint(metadata)
	db.datasets.save(metadata)

def ebird_insert_timefield():
	client = pymongo.MongoClient()
	db = client.creeval
	#db.eval('db.ebird.copyTo("ebird_bkp")') # This seems to cause a bunch of errors, run it manually in the mongo console first.
	all = db.ebird.find({})
	for i,row in enumerate(all):
		row['timefield'] = row['YEAR'] + (float(row['DAY']) / 366.0)
		if i % 100000 == 0:
			print i,":",row['timefield'],":",row['YEAR'],row['DAY']
		db.ebird.save(row)

if __name__ == "__main__":

	#ebird_insert_timefield()
	#ebird_metadata_setup()
	#fit_hypers("ebird",spearmint_params = {"look_back": 3,"stop_thresh": 0.05, 'datapath': "data/ebird/hyper_fitting/"})
	#train_expectations("ebird", datapath = 'data/ebird/expectations2/')
	unexpectedness("ebird", datapath = 'data/ebird/expectations2/')