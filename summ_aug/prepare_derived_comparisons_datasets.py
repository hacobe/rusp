import jsonlines
import os
import numpy as np
import tqdm
import yaml


def read_base_dataset(config):
	examples = []
	input_fnames = [
		"comparisons_base_train.jsonl",
		"comparisons_base_valid.jsonl",
		"comparisons_base_test.jsonl"
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


def write(config, dataset_map, dataset_names=None):
	if not dataset_names:
		dataset_names = sorted(list(dataset_map.keys()))

	for dataset_name in dataset_names:
		output_file = os.path.join(
			config["data_dir"], "comparisons_" + dataset_name + ".jsonl")
		print("{0}: {1}".format(
			output_file, len(dataset_map[dataset_name])))
		with jsonlines.open(output_file, "w") as fout:
			fout.write_all(dataset_map[dataset_name])


def add_policy_comp_dataset(
	examples,
	dataset_map,
	dataset_name,
	allowed_policy_comps,
	should_split,
	disallowed_prompts=set()):
	filtered_examples = []
	for example in examples:
		comp = (example["policy0"], example["policy1"])
		opp_comp = (example["policy1"], example["policy0"])

		if (comp not in allowed_policy_comps) and (opp_comp not in allowed_policy_comps):
			continue

		if example["prompt"] in disallowed_prompts:
			continue

		filtered_examples.append(example)

	np.random.shuffle(filtered_examples)

	if not should_split:
		dataset_map[dataset_name + "_train"] = filtered_examples
		return

	prompts = set()
	for example in filtered_examples:
		prompts.add(example["prompt"])

	prompts_list = sorted(list(prompts))
	np.random.shuffle(prompts_list)
	n = len(prompts_list)
	n_train = n // 2
	train_prompts = set(prompts_list[:n_train])
	train_examples = []
	test_examples = []
	for example in filtered_examples:
		if example["prompt"] in train_prompts:
			train_examples.append(example)
		else:
			test_examples.append(example)
	np.random.shuffle(train_examples)
	np.random.shuffle(test_examples)

	dataset_map[dataset_name + "_train"] = train_examples
	dataset_map[dataset_name + "_test"] = test_examples


def replace_choice(examples, better_than_policy_pairs):
	new_examples = []
	for example in tqdm.tqdm(examples):
		comp = (example["policy0"], example["policy1"])
		opp_comp = (comp[1], comp[0])
		if comp in better_than_policy_pairs:
			new_choice = 0
		elif opp_comp in better_than_policy_pairs:
			new_choice = 1
		else:
			raise ValueError("Unrecognized policy comparison: <" + comp + ">")

		new_example = {
			"prompt": example["prompt"],
			"completion0": example["completion0"],
			"completion1": example["completion1"],
			"choice": new_choice,
			"example": example
		}
		new_examples.append(new_example)
	return new_examples


if __name__ == "__main__":
	np.random.seed(0)

	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	base_dataset = read_base_dataset(config)

	dataset_map = {}

	add_policy_comp_dataset(
		examples=base_dataset,
		dataset_map=dataset_map,
		dataset_name="sup2vsup2",
		allowed_policy_comps=set([("sup2", "sup2")]),
		should_split=True)

	sup2vsup2_test_prompts = set()
	for example in dataset_map["sup2vsup2_test"]:
		sup2vsup2_test_prompts.add(example["prompt"])

	add_policy_comp_dataset(
		examples=base_dataset,
		dataset_map=dataset_map,
		dataset_name="refvsup2",
		allowed_policy_comps=set([("ref", "sup2")]),
		should_split=False,
		disallowed_prompts=sup2vsup2_test_prompts)

	dataset_map["sup2vsup2+refvsup2_train"] = (
		dataset_map["sup2vsup2_train"] + dataset_map["refvsup2_train"])
	np.random.shuffle(dataset_map["sup2vsup2+refvsup2_train"])

	dataset_map["refvsup2policy_train"] = replace_choice(
		dataset_map["refvsup2_train"],
		better_than_policy_pairs=set([("ref", "sup2")]))

	write(config, dataset_map, dataset_names=["refvsup2policy_train"])

