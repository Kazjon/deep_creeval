__author__ = 'kazjon'
import Recipe_Retrieval, pymongo, os

if __name__ == "__main__":
	from GA_Synthesiser import init_dataset, init_model, sanitise_for_mongo

	dataset = "mmf_0.01"
	experiment = "mmf_0.01_z5_all"
	modelpath = "/Users/kazjon/Dropbox/Documents/PyCharmProjects/DeeplyUnexpected/data/mmf_0.01/mmf_0.01_z5_all/step_0/"
	modelfn = "mmf_vae.pkl"

	surprise_depth = 2

	experiment = sanitise_for_mongo(experiment)

	metadata = init_dataset(dataset)
	client = pymongo.MongoClient()
	db = client.creeval
	coll = db[metadata["name"]]

	model = init_model(dataset, metadata, os.path.join(modelpath,modelfn), surprise_depth, experiment)
	cats = Recipe_Retrieval.init_substitution(metadata)
	tree = Recipe_Retrieval.init_recipe_retrieval(model, metadata, modelpath, experiment)

	cases = []
	#cases.append({"name":"Picky Kid","required":["i_cheese"],"forbidden":["i_hot sauce", "i_ground chillies", "i_chillies", "i_turmeric", "i_peanut butter", "i_spinach", "i_olives","i_vinegar"],"surprise_goal":0.33})
	#cases.append({"name":"Vegetarian","required":["i_beans", "i_garlic", "i_parmesan", "i_onions", "i_salt", "i_olives"],"forbidden":["i_fish", "i_beef", "i_chicken", "i_prawns", "i_bacon","i_sausage"],"surprise_goal":0.67})
	cases.append({"name":"Vegetarian","required":["i_beans", "i_salt", "i_eggs", "i_onions", "i_flour", "i_baking powder"],"forbidden":["i_fish", "i_beef", "i_chicken", "i_prawns", "i_bacon","i_sausage"],"surprise_goal":0.67})
	#cases.append({"name":"Halal Foodie","required":["i_chicken"],"forbidden":["i_pork", "i_sausage", "i_bacon", "i_red wine", "i_white wine", "i_sherry"],"surprise_goal":0.99})

	for case in cases:
		targets, target_ids = Recipe_Retrieval.find_closest_recipe(case["required"], model, tree, num_to_return=10)

		for target,target_id in zip(targets, target_ids):
			cursor = coll.find_one({"_id": target_id})
			print target
			if any([f in target for f in case["forbidden"]]):
				print "Forbidden ings found!"

			#pprint.pprint(cursor["raw_text"])
			#target_ings = [k.decode() for k in target.keys() if k[:2] == "i_"]
			#print target_ings

			#adapt_recipe(source_ings, target_ings, target["raw_text"], model, metadata, experiment, cats)
			Recipe_Retrieval.adapt_recipe(case["required"], target, cursor["raw_text"], model, metadata, experiment, cats, max_surp_only=False, forbidden=case["forbidden"], min_shared=1, max_retrieved=25,max_generated=6)