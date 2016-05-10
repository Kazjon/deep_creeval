__author__ = 'kazjon'
from copy import deepcopy
import numpy as np
from conceptual_space import monary_load
import pymongo, sys, pprint, itertools, csv, os.path
from GA_Synthesiser import wrap_plausibility_and_surprise
import cPickle as pickle
from sklearn.neighbors import BallTree

#self, metadata, eval_name, override_query, sample_sizes = 1000, n_iter=50, start=0,stop=-1, threshold=2, depth_limit=3):
def gen_hidden_reps_in_db(domain_name, metadata, model, override_query = {}, start=0,stop=-1):
	if len(override_query.keys()):
		q = override_query
	else:
		q = deepcopy(metadata["query"])
	data_slice = monary_load(domain_name,metadata["fields_x"],metadata["fields_y"],find_args=q, split = None, start=start, stop=stop, type=model.fixed_hypers["monary_type"]).X

	client = pymongo.MongoClient()
	db = client.creeval
	coll = db[metadata["name"]]
	cursor = coll.find(q, skip=start, limit=stop)

	for design in data_slice:
		print design
		record = cursor.next()
		print record
		d = model.binarise_features(model.features_from_design_vector(design))
		h = model.construct_from_hidden(np.atleast_2d(np.array(d)))
		print h
		sys.exit()

def init_substitution(metadata):
	categories_file = "/Users/kazjon/Dropbox/Documents/PyCharmProjects/DeeplyUnexpected/data/mmf_0.01/mmf_0.01_feature_categories.csv"
	cats = {}
	with open(categories_file, "rb") as cf:
		reader = csv.DictReader(cf)
		for row in reader:
			for k in row.keys():
				if not len(row[k]):
					del row[k]
			n = row["NAME"]
			del row["NAME"]
			cats[n] = row.keys()
	print "-- Initialised substitution knowledge base."
	return cats

forbidden_sub_cats = ["BUILDING_BLOCK"]

def legal_substitution(cats, remove, replace, current_subs = None):
	if current_subs == None:
		current_subs = []
	for add in replace:
		for rem in remove:
			if any([i in cats[rem] for i in forbidden_sub_cats]):
				break
			for addcat in cats[add]:
				if addcat in cats[rem]:
					new_remove = deepcopy(list(remove))
					new_replace = deepcopy(list(replace))
					new_subs = deepcopy(current_subs)
					new_remove.remove(rem)
					new_replace.remove(add)
					new_subs.append((add,rem))
					return legal_substitution(cats,new_remove, new_replace, current_subs=new_subs)
	if not len(remove):
		for i in replace:
			if not any([i==s[0] for s in current_subs]):
				current_subs.append((i,None))
		return True, current_subs
	return False, []

def init_recipe_retrieval(model, metadata, modelpath, experiment, override_query = {}, treefn = "retrieval_tree.pkl"):
	treepath = os.path.join(modelpath,treefn)
	if os.path.exists(treepath):
		return pickle.load(open(treepath,"rb"))

	#Load entire database
	if len(override_query.keys()):
		q = override_query
	else:
		q = deepcopy(metadata["query"])
	data = monary_load(model.domain_name,metadata["fields_x"],metadata["fields_y"],find_args=q, split = None, type=model.fixed_hypers["monary_type"]).X
	#Evaluate hidden reps for database
	X = model.represent(data)

	#Load recipe ids
	ids=[]
	client = pymongo.MongoClient()
	db = client.creeval
	coll = db[metadata["name"]]
	cursor = coll.find(q)

	count = 0
	for design in data:
		count+=1
		record = cursor.next()
		ids.append(record["_id"])

	#Generate tree for efficient searching
	tree = BallTree(X)
	tree_dict = {"tree":tree,"data":data, "ids":ids}
	pickle.dump(tree_dict, open(treepath, 'wb'))
	print "-- Initialised recipe retrieval engine."
	return tree_dict


def find_closest_recipe(source, model, tree, num_to_return=1):
	#print "source:",source
	source_v = model.design_vector_from_features(source)
	source_rep = model.represent(np.atleast_2d(np.array(source_v)))[0].tolist()
	#print "source_rep:",source_rep
	_,target_ind = tree["tree"].query(source_rep, k=num_to_return)
	#print "target_ind:",target_ind
	targets = []
	for i in range(num_to_return):
		targets.append(model.binarise_features(model.features_from_design_vector(tree["data"][target_ind.T[i]])))
	#print "target:",target
	return targets, [tree["ids"][ind] for ind in target_ind.T]

#BROKAN!!!
'''
def find_and_adapt(source, model, metadata, experiment, cats=None, use_lower_bound=True, surprise_depth=2):
	if cats==None:
		cats = init_substitution(metadata)
	target = find_closest_recipe(source, model, metadata, experiment)
	adapt_recipe(source, target, None, model, metadata, experiment, cats, use_lower_bound=use_lower_bound, surprise_depth=surprise_depth)
'''

#Non-probabilistic case first
def adapt_recipe(source, target, target_text, model, metadata, experiment, cats, use_lower_bound=True, surprise_depth=2, desired_surprise=-1, max_surp_only=False, surprise_proximity_thresh = -1, min_shared=2, max_retrieved=25,max_generated=6, forbidden=[]):
	#Get shared ingredients
	ascii_target = [i.encode("utf-8") for i in target]
	ascii_source = [i.encode("utf-8") for i in source]
	shared_ings = [i for i in ascii_source if i in ascii_target]
	retrieved_adds = [i for i in ascii_target if i not in ascii_source]
	generated_adds = [i for i in ascii_source if i not in ascii_target]
	adds = generated_adds+retrieved_adds

	print "* Source:",ascii_source
	print "* Target:",ascii_target

	if len(shared_ings) < min_shared or len(retrieved_adds) > max_retrieved or len(generated_adds) > max_generated:
		print "* Recipes judged incompatible, aborting adaptation."
		return
	print "* shared_ings:",shared_ings
	print "* retrieved_adds:",retrieved_adds
	print "* generated_adds:",generated_adds

	print "* Retrieved recipe:"
	pprint.pprint(target_text)





	#for l in range(len(adds)):
	#	for c in itertools.combinations(adds,l+1):
	#		r = shared_ings+list(c)
	#		vals = wrap_plausibility_and_surprise(model.design_vector_from_features(r),model=model, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=False, errors_by_length=metadata["experiments"][experiment]["errors_by_length"],from_visible=True, feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound, surprise_dist=metadata["experiments"][experiment]["surprise_distribution"], surprise_depth=surprise_depth, samples=100)
	#		print "r:{0}: v:{1:.4f}, n:{2:.4f}".format(r,vals[0],vals[1])

	hypotheticals = [target]
	substitutions = [[]]
	for l in range(max(len(retrieved_adds),len(generated_adds))): #SOmehow this is causing duplicates!
		for remove in itertools.combinations(retrieved_adds,min(l+1,len(retrieved_adds))):
			for replace in itertools.combinations(generated_adds,min(l+1,len(generated_adds))):
				#print remove
				#print replace
				legal,subs = legal_substitution(cats, remove, replace)
				#print legal, subs
				#print "------"
				if legal:
					r = list(set(shared_ings) | (set(retrieved_adds) - set(remove)) | set(replace))
					if len(forbidden):
						for f in forbidden:
							if f in r:
								r.remove(f)
								subs.append((None,f))
					hypotheticals.append(r)
					substitutions.append(subs)
					#If this is a legal substitution, we should consider all the "this plus just add stuff" variants too
					remaining_generated_adds = [i for i in generated_adds if not i in replace]
					for l2 in range(len(remaining_generated_adds)):
						for add in itertools.combinations(remaining_generated_adds,l2+1):
							r2 = r + list(add)
							hypotheticals.append(r2)
							substitutions.append(subs + [(i,None) for i in add])
		for add in itertools.combinations(generated_adds,l+1):
			r = shared_ings + retrieved_adds + list(add)
			hypotheticals.append(r)
			substitutions.append([(i,None) for i in add])

	vals = []
	for i,r in enumerate(hypotheticals):
		vals.append(wrap_plausibility_and_surprise(model.design_vector_from_features(r),
		                                           model=model,
		                                           plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"],
		                                           weight_by_length=False,
		                                           errors_by_length=metadata["experiments"][experiment]["errors_by_length"],
		                                           from_visible=True,
		                                           feature_list=metadata["fields_x"],
		                                           use_lower_bound=use_lower_bound,
		                                           surprise_dist=metadata["experiments"][experiment]["surprise_distribution"],
		                                           surprise_depth=surprise_depth,
		                                           samples=1000,
		                                           max_surp_only=max_surp_only
												)) #, desired_surprise=desired_surprise))


	source_p,source_s = wrap_plausibility_and_surprise(model.design_vector_from_features(source),
		                                           model=model,
		                                           plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"],
		                                           weight_by_length=False,
		                                           errors_by_length=metadata["experiments"][experiment]["errors_by_length"],
		                                           from_visible=True,
		                                           feature_list=metadata["fields_x"],
		                                           use_lower_bound=use_lower_bound,
		                                           surprise_dist=metadata["experiments"][experiment]["surprise_distribution"],
		                                           surprise_depth=surprise_depth,
		                                           samples=1000,
		                                           max_surp_only=max_surp_only
												)
	if desired_surprise >= 0:
		print "* Source:",sorted(ascii_source), ", v:",source_p, " n(-d):",-abs(source_s-desired_surprise)
		print "* Target:",sorted(ascii_target), ", v:",vals[0][0], " n(-d):",-abs(vals[0][1]-desired_surprise)
	else:
		print "* Source:",sorted(ascii_source), ", v:",source_p, " n:",source_s
		print "* Target:",sorted(ascii_target), ", v:",vals[0][0], " n:",vals[0][1]
	for i,r in enumerate(sorted(vals[1:], key=lambda x:(x[1],x[0]), reverse=True)):
		#adds = [ing for ing in hypotheticals[i+1] if ing not in ascii_target]
		#rems = [ing for ing in ascii_target if ing not in hypotheticals[i+1]]
		#print "+{0}:, -{1}, v:{2:.4f}, n:{3:.4f}".format(sorted(adds), sorted(rems),r[0]-vals[0][0],r[1]-vals[0][1])
		if surprise_proximity_thresh < 0 or abs(r[1]-desired_surprise) < surprise_proximity_thresh:
			print "    * subs:{0}, v:{1:.4f}, n:{2:.4f}".format(substitutions[i+1],r[0],r[1])
		#	if desired_surprise > 0:
		#		print "    * subs:{0}, v:{1:.4f}, n:{2:.4f}".format(substitutions[i+1],r[0]-vals[0][0],r[1]-desired_surprise)
		#	else:
		#		print "    * subs:{0}, v:{1:.4f}, n:{2:.4f}".format(substitutions[i+1],r[0]-vals[0][0],r[1]-vals[0][1])


	#Use the model to evaluate every possible inclusion/exclusion.

if __name__ == "__main__":
	from GA_Synthesiser import init_dataset, init_model, sanitise_for_mongo

	dataset = "mmf_0.01"
	experiment = "mmf_0.01_z5_all"
	modelpath = "/Users/kazjon/Dropbox/Documents/PyCharmProjects/DeeplyUnexpected/data/mmf_0.01/mmf_0.01_z5_all/step_0/"
	modelfn = "mmf_vae.pkl"


	#target_name = "Carrot Cake"

	#Random crap manually pulled from runs of the GA (which we haven't integrated with this yet)
	source_txt = "mint, cheese, garlic, salt, black pepper, garlic powder, olive oil, onions"
	source_txt = "mint, cheese, garlic, parsley, salt, tomatoes, black pepper, olive oil"
	source_txt = "lemon, stock, mint, cheese, garlic, salt, black pepper, olive oil"
	source_txt = "lemon, mint, cheese, vanilla, butter, milk, flour, sugar"
	source_txt = "mint, cheese, garlic, oregano, salt, black pepper, olive oil, onions"
	source_txt = "cheese, salt, eggs, milk, baking powder, black pepper, bacon, onions"
	#source_txt = "cheese, celery, eggs, milk, baking powder, onions, flour, potatoes"
	#source_txt = "pork, soy sauce, salt, baking powder, lemon, black pepper, vegetable oil, sugar"
	#source_txt = "pork, lemon, mint, white vinegar, tomatoes, salt, honey"
	#source_txt = "beef, cheese, eggs, milk, baking powder, black pepper, onions, flour"

	source_ings = source_txt.split(", ")

	#source_ings = ['i_zucchini', 'i_sugar', 'i_baking powder', 'i_eggs', 'i_flour', 'i_vanilla',  'i_salt', 'i_butter']
	#source_ings_prob = {"cinnamon":0.96, "butter":0.87, "sugar":0.87, "eggs":0.82, "flour":0.80, "salt":0.63, "nutmeg":0.56, "milk":0.54, "vanilla":0.39, "baking powder":0.31, "brown sugar":0.27, "lemon":0.27, "cloves":0.22, "raisins":0.21, "walnuts":0.11}
	#source_ings = source_ings_prob.keys()
	source_ings = ["i_"+i for i in source_ings]

	surprise_depth = 2

	experiment = sanitise_for_mongo(experiment)

	metadata = init_dataset(dataset)
	client = pymongo.MongoClient()
	db = client.creeval
	coll = db[metadata["name"]]

	model = init_model(dataset, metadata, os.path.join(modelpath,modelfn), surprise_depth, experiment)
	cats = init_substitution(metadata)
	tree = init_recipe_retrieval(model, metadata, modelpath, experiment)


	#Check up on the surprise distribution -- it's possible it's being loaded from the wrong place, and it's certainly being stored in the wrong place!
	#surprise_gloss_prefix = {0:"For something a little different", metadata["experiments"][experiment]["surprise_distribution"]["95%"]:"If you want to get a little crazy"}
	#surprise_gloss_text_sub = ", try this, but with {0} instead of {1}?"
	#surprise_gloss_text_add = ", try this, but add some {0}?"
	#surprise_gloss_text_sub_and_add = ", try this, but with {0} instead of {1}, and then add some {2}?"


	targets, target_ids = find_closest_recipe(source_ings, model, tree, num_to_return=10)
	#target = coll.find_one({"id":target_id})
	#pprint.pprint(targets)

	client = pymongo.MongoClient()
	db = client.creeval
	coll = db[metadata["name"]]

	for target,target_id in zip(targets, target_ids):
		cursor = coll.find_one({"_id": target_id})

		#pprint.pprint(cursor["raw_text"])
		#target_ings = [k.decode() for k in target.keys() if k[:2] == "i_"]
		#print target_ings

		#adapt_recipe(source_ings, target_ings, target["raw_text"], model, metadata, experiment, cats)
		adapt_recipe(source_ings, target, cursor["raw_text"], model, metadata, experiment, cats, max_surp_only=False, desired_surprise=0.75, surprise_proximity_thresh=0.25)