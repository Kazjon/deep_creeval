__author__ = 'kazjon'
import argparse, csv, os, os.path, json, shutil

parser = argparse.ArgumentParser(description='Use this to grab the error differences out of an experiment folder.')
parser.add_argument('dataset',help="Name of the dataset to work with")
parser.add_argument("-n","--exp_name",help="The name of the expectation model that has already been run.",required=False,default="expectations")
args = parser.parse_args()

fn = "diff_errors.csv"
outdir = "diff_errors"
exp_root = os.path.join("data",args.dataset,args.exp_name)
outdir = os.path.join(exp_root,outdir)
if not os.path.exists(outdir):
	os.mkdir(outdir)

dErrors = []
print os.listdir(exp_root)
with open(os.path.join(exp_root,"steps.txt")) as stepf:
	steps = json.load(stepf)
for s in range(1,len(steps)):
	shutil.copyfile(os.path.join(exp_root,"step_{0}".format(s),fn),os.path.join(outdir,"step_{0}_".format(s)+fn))
	#with open(os.path.join(exp_root,"step_{%i}".format(s))) as csvf:
	#	r = csv.reader(csvf)
	#	dErrors.append(r.readlines())