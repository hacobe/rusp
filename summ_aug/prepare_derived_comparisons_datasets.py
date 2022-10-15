import jsonlines
import os
import numpy as np
import tqdm
import yaml


def read_examples(config):
	examples = []
	input_fnames = [
		"comparisons_train.jsonl",
		"comparisons_valid.jsonl",
		"comparisons_test.jsonl"
	]
	for fname in input_fnames:
		input_file = os.path.join(config["data_dir"], fname)
		with jsonlines.open(input_file, "r") as fin:
			for line in tqdm.tqdm(fin):
				example = {}
				for key in line:
					example[key] = line[key]
				assert line["example"]["summaries"][0]["text"].strip() == line["completion0"].strip()
				assert line["example"]["summaries"][1]["text"].strip() == line["completion1"].strip()
				example["policy0"] = line["example"]["summaries"][0]["policy"]
				example["policy1"] = line["example"]["summaries"][1]["policy"]
				examples.append(example)
	return examples


def write_sup2_examples(examples):
	sup2_examples = []
	for example in examples:
		if example["policy0"] != "sup2":
			continue
		if example["policy1"] != "sup2":
			continue
		sup2_examples.append(example)

	sup2_prompts = set()
	for example in sup2_examples:
		sup2_prompts.add(example["prompt"])

	sup2_prompts_list = sorted(list(sup2_prompts))
	np.random.shuffle(sup2_prompts_list)
	n = len(sup2_prompts_list)
	n_train = n // 2
	sup2_train_prompts = set(sup2_prompts_list[:n_train])
	sup2_train_examples = []
	sup2_test_examples = []
	for example in sup2_examples:
		if example["prompt"] in sup2_train_prompts:
			sup2_train_examples.append(example)
		else:
			sup2_test_examples.append(example)
	np.random.shuffle(sup2_train_examples)
	np.random.shuffle(sup2_test_examples)

	print("len(sup2_train_examples): " + str(len(sup2_train_examples)))
	print("len(sup2_test_examples): " + str(len(sup2_test_examples)))

	output_file = os.path.join(config["data_dir"], "comparisons_sup2_train.jsonl")
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(sup2_train_examples)

	output_file = os.path.join(config["data_dir"], "comparisons_sup2_test.jsonl")
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(sup2_test_examples)


if __name__ == "__main__":
	np.random.seed(0)

	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	examples = read_examples(config)
	write_sup2_examples(examples)

