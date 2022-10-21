import utils

import jsonlines
import os
import numpy as np
import yaml


def _replace_choice(examples, better_than_policy_pairs):
	new_examples = []
	for example in examples:
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


def _extract_prompt_fn(line):
	prompt = "SUBREDDIT: r/" + line["info"]["subreddit"].strip() + "\n"
	prompt += "TITLE: " + line["info"]["title"].strip() + "\n"
	prompt += "POST: " + line["info"]["post"].strip() + "\n"
	prompt += "TL;DR:"
	return prompt


def get_ref_examples(config):
	input_files = []
	for split in utils.SPLITS:
		input_files.append(os.path.join(config["data_dir"], split + ".jsonl"))

	ref_examples = []
	for input_file in input_files:
		split = os.path.basename(input_file).replace(".jsonl", "")

		with jsonlines.open(input_file, "r") as fin:
			for line in fin:
				example = {}
				example["prompt"] = _extract_prompt_fn({"info": line})
				example["completion"] = " " + line["summary"] + " <|endoftext|>"
				example["split"] = split
				example["example"] = line
				ref_examples.append(example)
	np.random.shuffle(ref_examples)
	return ref_examples


def get_comparison_examples(config):
	input_files = []
	input_dir = os.path.join(config["data_dir"], "comparisons/")
	fnames = os.listdir(input_dir)
	for fname in fnames:
		if fname.find("edit") != -1:
			continue
		if not fname.endswith(".json"):
			continue
		if fname.find("cnndm") != -1:
			continue
		input_files.append(os.path.join(input_dir, fname))

	comparison_examples = []
	for input_file in input_files:
		with jsonlines.open(input_file, "r") as fin:
			for line in fin:
				if line["split"] == "train":
					split = "train"
				elif line["split"] == "valid1":
					split = "valid"
				elif line["split"] == "valid2":
					split = "test"
				else:
					raise ValueError("Unrecognized split: <" + split + ">")

				example = {}
				example["example"] = line
				example["prompt"] = _extract_prompt_fn(line)
				assert len(line["summaries"]) == 2
				assert line["choice"] in set([0, 1])
				# Note that we do not include "<|endoftext|>" here.
				example["completion0"] = " " + line["summaries"][0]["text"].strip()
				example["completion1"] = " " + line["summaries"][1]["text"].strip()
				example["choice"] = line["choice"]
				example["split"] = split
				example["policy0"] = line["summaries"][0]["policy"]
				example["policy1"] = line["summaries"][1]["policy"]
				comparison_examples.append(example)
	np.random.shuffle(comparison_examples)
	return comparison_examples


def get_test_comparison_examples(comparison_examples, allowed_policy_comps, num_prompts):
	filtered_comparison_examples = utils.filter_comparison_examples(
		comparison_examples, allowed_policy_comps)

	prompts = set()
	for example in filtered_comparison_examples:
		prompts.add(example["prompt"])

	prompts_list = sorted(list(prompts))
	np.random.shuffle(prompts_list)
	test_prompts = set(prompts_list[:num_prompts])

	test_comparison_examples = []
	for example in filtered_comparison_examples:
		if example["prompt"] not in test_prompts:
			continue
		test_comparison_examples.append(example)

	np.random.shuffle(test_comparison_examples)
	return test_comparison_examples


if __name__ == "__main__":
	np.random.seed(0)

	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	ref_examples = get_ref_examples(config)
	comparison_examples = get_comparison_examples(config)

	allowed_policy_comps = [
		("ref", "sup1"),
		("ref", "sup2"),
		("sup1", "sup1"),
		("sup2", "sup2"),
	]
	test_comparison_examples = get_test_comparison_examples(
		comparison_examples, allowed_policy_comps, num_prompts=2000)
	output_file = os.path.join(
		config["data_dir"], "comparisons_tldr_refvsup+supvsup_test.jsonl")
	utils.write_examples(test_comparison_examples, output_file)

	test_prompts = set()
	for example in test_comparison_examples:
		test_prompts.add(example["prompt"])

	train_ref_examples = utils.filter_examples(ref_examples, test_prompts)
	train_comparison_examples = utils.filter_examples(comparison_examples, test_prompts)

	for split in utils.SPLITS:
		output_file = os.path.join(config["data_dir"], "refs_tldr_all_" + split  + ".jsonl")
		utils.write_examples(
			utils.extract_split(train_ref_examples, split), output_file)	

		output_file = os.path.join(config["data_dir"], "comparisons_tldr_all_" + split  + ".jsonl")
		utils.write_examples(
			utils.extract_split(train_comparison_examples, split), output_file)	

	filtered_train_comparison_examples = utils.filter_comparison_examples(
		train_comparison_examples, allowed_policy_comps)
	output_file = os.path.join(
		config["data_dir"], "comparisons_tldr_refvsup+supvsup_train.jsonl")
	utils.write_examples(filtered_train_comparison_examples, output_file)

	for policy_comp in allowed_policy_comps:
		policy_comp_name = policy_comp[0] + "v" + policy_comp[1]

		filtered_train_comparison_examples = utils.filter_comparison_examples(
			train_comparison_examples, set([policy_comp]))
		np.random.shuffle(filtered_train_comparison_examples)

		output_file = os.path.join(
			config["data_dir"], "comparisons_tldr_" + policy_comp_name + "_train.jsonl")
		utils.write_examples(filtered_train_comparison_examples, output_file)

		assert policy_comp[1] != "ref"
		if policy_comp[0] == "ref":
			output_file = os.path.join(
				config["data_dir"], "comparisons_tldr_" + policy_comp_name + "policy" + "_train.jsonl")
			utils.write_examples(
				_replace_choice(
					filtered_train_comparison_examples,
					better_than_policy_pairs=set([policy_comp])), output_file)

		filtered_test_comparison_examples = utils.filter_comparison_examples(
			test_comparison_examples, set([policy_comp]))
		np.random.shuffle(filtered_test_comparison_examples)

		output_file = os.path.join(
			config["data_dir"], "comparisons_tldr_" + policy_comp_name + "_test.jsonl")
		utils.write_examples(filtered_test_comparison_examples, output_file)