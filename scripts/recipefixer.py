import pymongo,argparse,sys

parser = argparse.ArgumentParser(description='Use this to fix buggerised recipes in mongodb.')
parser.add_argument('--dataset',help="Name of the dataset to work with")
parser.add_argument('--recipe',help="Name of the recipe to edit")
parser.add_argument('--ingredient',help="Name of the ingredient to add or remove.")

args = parser.parse_args()

client = pymongo.MongoClient()
db = client.creeval

client = pymongo.MongoClient()
db = client.creeval
metadata = db.datasets.find_one({"name": args.dataset})

recipe = db[args.dataset].find_one({"name": args.recipe})
print recipe

if not args.ingredient in metadata["fields_x"]:	
	sys.exit(args.ingredient + " not found in dataset " + args.dataset)

if args.ingredient in recipe:
	del recipe[args.ingerdient]
else:
	recipe[args.ingredient] = 1

db[args.dataset].replace_one({"_id": recipe["_id"]},recipe)

