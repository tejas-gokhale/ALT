import os 
import sys 
from glob import glob
import json 
import numpy as np 
import csv

import warnings
warnings.filterwarnings("ignore")

if sys.argv[1] == 'pacs':
	domains = ['photo', 'art_painting', 'cartoon', 'sketch']
elif sys.argv[1] == 'officehome':
	domains = ['real', 'art', 'clipart', 'product']

elif sys.argv[1] == 'digits':
	domains = ['mnist10k', 'mnist_m', 'svhn', 'usps', 'synth']
else:
	print("--- EXITING --- argv[1] should be pacs or digits")
	raise SystemExit

def stringmatch(s, l):
	''' check if all items of list l are in string s'''
	for ll in l:
		if ll not in s:
			return False
	return True

ckpt_path = os.path.join("./checkpoints/", sys.argv[1])
ckpt_dirs = glob(os.path.join(ckpt_path, "*"))
name = sys.argv[2:]
ckpt_dirs = [jj for jj in ckpt_dirs if stringmatch(jj, name)]
print(len(ckpt_dirs))


results_list = []
count = 0
for dd in ckpt_dirs:
	jsons = glob(os.path.join(dd, "log_seed*.json"))
	jsons = [jj for jj in jsons if stringmatch(jj, name)]
	
	if sys.argv[1] == 'pacs':
		results = {
			'photo': [], 'art_painting': [], 'cartoon': [], 'sketch': []
			}
	elif sys.argv[1] == 'officehome':
		results = {
			'real': [], 'art': [], 'clipart': [], 'product': []
			}
	elif sys.argv[1] == 'digits':
		results = {
			'mnist10k': [], 
			'mnist_m': [], 'svhn': [], 'usps': [], 'synth':[]
			}
	trg, all_dom = [], []
	all_avg = 0
	for jj in jsons:
		with open(jj, 'r') as f:
			res_jj = json.load(f)
		all_avg = 0
		for kk in domains:
			results[kk].append(res_jj[kk])
			all_avg += res_jj[kk]/len(domains)
		all_dom.append(all_avg)
		trg.append(res_jj['Target Average'])

	combined = {}
	combined['source'] = dd.split('/')[-1].split('-')[2]
	combined['name'] = dd

	for kk in domains:
		combined[kk] = float("{0:.3f}".format(np.mean(results[kk])))
		combined[kk+'_std'] = float("{0:.3f}".format(np.std(results[kk])))

	combined['trg'] = float("{0:.3f}".format(np.mean(trg)))
	combined['trg_std'] = float("{0:.3f}".format(np.std(trg)))
	combined['all'] = float("{0:.3f}".format(np.mean(all_dom)))
	combined['all_std'] = float("{0:.3f}".format(np.std(all_dom)))

	with open(os.path.join(dd, 'log_combined.json'), 'w') as f:
		json.dump(combined, f, indent=4)

	if len(jsons) > 0:
		results_list.append(combined)
		count += 1
	
		keys = results_list[0].keys()
		with open('./results/log_combined_{}_{}.csv'.format(
			sys.argv[1], '_'.join(name)), 'w', newline='')  as output_file:
		    dict_writer = csv.DictWriter(output_file, keys)
		    dict_writer.writeheader()
		    dict_writer.writerows(results_list)
		print(dd)
		print(combined)
		print("--------")
print(count)