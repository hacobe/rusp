import utils

import datasets
import jsonlines
import os
import numpy as np
import tqdm
import yaml


def get_ref_examples(config):
	dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", cache_dir=config["cache_dir"])

	ref_examples = []
	prompts = set()
	n_skipped = 0
	for key in dataset.keys():
		if key == "validation":
			split = "valid"
		else:
			split = key
		assert split in utils.SPLITS

		for i in tqdm.tqdm(range(len(dataset[key]))):
			line = dataset[key][i]

			prompt = "ARTICLE: " + line["article"].strip() + "\n"
			prompt += "TL;DR:"

			example = {}
			example["prompt"] = prompt
			example["completion"] = " " + line["highlights"].strip() + "<|endoftext|>"
			example["split"] = split
			example["example"] = line
			ref_examples.append(example)

	np.random.shuffle(ref_examples)

	return ref_examples


def get_comparison_examples(config):
	input_dir = os.path.join(config["data_dir"], "comparisons/")

	input_files = []
	fnames = os.listdir(input_dir)
	for fname in fnames:
		if fname.find("edit") != -1:
			continue
		if not fname.endswith(".json"):
			continue
		if fname.find("cnndm") == -1:
			continue
		input_files.append(os.path.join(input_dir, fname))

	comparison_examples = []
	for input_file in input_files:
		with jsonlines.open(input_file, "r") as fin:
			for line in fin:
				assert line["split"] == "valid2"

				prompt = "ARTICLE: " + line["info"]["article"].strip() + "\n"
				prompt += "TL;DR:"

				example = {}
				example["example"] = line
				example["prompt"] = prompt
				assert len(line["summaries"]) == 2
				assert line["choice"] in set([0, 1])
				# Note that we do not include "<|endoftext|>" here.
				example["completion0"] = " " + line["summaries"][0]["text"].strip()
				example["completion1"] = " " + line["summaries"][1]["text"].strip()
				example["choice"] = line["choice"]
				example["split"] = line["split"]
				example["policy0"] = line["summaries"][0]["policy"]
				example["policy1"] = line["summaries"][1]["policy"]

				comparison_examples.append(example)

	np.random.shuffle(comparison_examples)

	return comparison_examples


def get_test_prompts(comparison_examples):
	prompts = set()
	for example in comparison_examples:
		prompts.add(example["prompt"])
	return prompts


if __name__ == "__main__":
	np.random.seed(0)

	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	ref_examples = get_ref_examples(config)
	comparison_examples = get_comparison_examples(config)

	output_file = os.path.join(config["data_dir"], "comparisons_cnndm_all_test.jsonl")
	utils.write_examples(comparison_examples, output_file)

	supvsup_comparison_examples = utils.filter_comparison_examples(
		comparison_examples, set([("supcnndm3_6b_t.3", "supcnndm3_6b_t.3")]))
	output_file = os.path.join(config["data_dir"], "comparisons_cnndm_supvsup_test.jsonl")
	utils.write_examples(supvsup_comparison_examples, output_file)

	test_prompts = get_test_prompts(comparison_examples)
	train_ref_examples = utils.filter_examples(ref_examples, test_prompts)

	for split in utils.SPLITS:
		output_file = os.path.join(config["data_dir"], "refs_cnndm_all_" + split  + ".jsonl")
		utils.write_examples(utils.extract_split(train_ref_examples, split), output_file)	
