import pymongo, pprint, bisect, copy
import numpy as np

base_collection = "mmf_0.01"
collections_to_compare = ["mmf_0.01_nosugar_evals", "mmf_0.01_onlysugar_evals", "mmf_0.01_all_evals"]
threshold = 6 # Minimum number of wows we give a shit about
results_to_return = 20 # Number of results to return for each condition -- -1 for all

def max_below_inf(recipe):
	vals = np.array([s["value"] for s in recipe["surprise_sets"]])
	vals = vals[vals<float("inf")]
	if len(vals):
		return np.round(np.max(vals),decimals=2)
	return 0

client = pymongo.MongoClient()
db = client.creeval
names = []
results = {}
surprise_sets = {}
ingredients =[]
full_texts = []
ingredient_list = db["datasets"].find_one({"name":base_collection})["fields_x"]
#print ingredient_list

for collection in collections_to_compare:
	results[collection] = []
	surprise_sets[collection] = []
for collection in collections_to_compare:
	cursor = db[collection].find({})
	for recipe in cursor:
		if recipe["name"] not in names:
			names.append(recipe["name"])
			base_recipe = db[base_collection].find_one({"name":recipe["name"]})
			full_texts.append(base_recipe["raw_text"])
			ingredients.append([i for i in base_recipe.keys() if i in ingredient_list])
			
			for collection2 in collections_to_compare:
				results[collection2].append(-1)
				surprise_sets[collection2].append([])
			results[collection][-1] = max_below_inf(recipe)
			surprise_sets[collection][-1] = recipe["surprise_sets"]
		else:
			i = names.index(recipe["name"])
			results[collection][i] = max_below_inf(recipe)
			surprise_sets[collection][i] = recipe["surprise_sets"]
			
#print names
#print results
result_tuples = zip(names,*[results[c] for c in collections_to_compare])
ranked_lists = {}

for k,c in enumerate(collections_to_compare[:-1]):
	ranked_lists[c] = copy.deepcopy(result_tuples)
	ranked_lists[c].sort(key=lambda r: r[k+1], reverse=True)
	rankings = np.array(zip(*ranked_lists[c])[k+1])
	rankings = rankings[rankings>threshold]
	if results_to_return > 0 and results_to_return < len(rankings):
		rankings = rankings[:results_to_return]
	ranked_lists[c] = ranked_lists[c][:len(rankings)]
	print "--------------------------------------------------",c,"--------------------------------------------------"
	for r in ranked_lists[c]:
		i = names.index(r[0])
		pprint.pprint(full_texts[i])
		print "  * {0}: {1:.2f}, {2:.2f}, {3:.2f}:".format(*r)
		print "    * Ingredients:",ingredients[i]
		
		for c2 in collections_to_compare:
			print "    * Surprises for",c2
			for s in sorted(surprise_sets[c2][i],key=lambda s:s["value"],reverse=True):
				if s["value"] > threshold:
					print "       * ",s["discovery"],"given",s["context"],": ",s["value"],"wows."
