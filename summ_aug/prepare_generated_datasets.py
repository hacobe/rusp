import os
import jsonlines
import numpy as np
import yaml


def get_prompt_to_pred_example(input_file):
	prompt_to_example = {}
	with jsonlines.open(input_file, "r") as fin:
		for line in fin:
			assert line["prompt"] not in prompt_to_example
			example = {}
			example["completion"] = " " + line["predictions"][0]["text"].strip()
			example["example"] = line
			prompt_to_example[line["prompt"]] = example
	return prompt_to_example


def get_prompt_to_ref_example(input_file):
	prompt_to_example = {}
	with jsonlines.open(input_file, "r") as fin:
		for line in fin:
			assert line["prompt"] not in prompt_to_example
			example = {}
			example["completion"] = " " + line["example"]["completion"].replace("<|endoftext|>", "").strip()
			example["example"] = line
			prompt_to_example[line["prompt"]] = example
	return prompt_to_example


def add_comparisons_dataset(policy_to_prompt_to_example, good_policy, bad_policy, dataset_map):
	dataset_name = good_policy + "v" + bad_policy

	prompt_to_good_example = policy_to_prompt_to_example[good_policy]
	prompt_to_bad_example = policy_to_prompt_to_example[bad_policy]
	
	examples = []
	for prompt in prompt_to_good_example:
		good_example = prompt_to_good_example[prompt]
		if prompt not in prompt_to_bad_example:
			continue
		bad_example = prompt_to_bad_example[prompt]
		r = np.random.random()
		choice = 0 if r <= 0.5 else 1

		example = {}
		example["prompt"] = prompt
		example["completion" + str(choice)] = good_example["completion"]
		example["completion" + str(1-choice)] = bad_example["completion"]
		example["choice"] = choice
		example["policy" + str(choice)] = good_policy
		example["policy" + str(1-choice)] = bad_policy
		example["example" + str(choice)] = good_example
		example["example" + str(1-choice)] = bad_example
		examples.append(example)

	np.random.shuffle(examples)

	dataset_map[dataset_name + "_train"] = examples


def get_predictions_file(config, model_name, prediction_name):
	return os.path.join(config["models_dir"], model_name, "final_checkpoint",
		"predictions_" + prediction_name + ".jsonl")


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


if __name__ == "__main__":
	np.random.seed(0)

	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	policy_to_prompt_to_example = {}

	input_file = os.path.join(config["data_dir"], "refs_excludesup2vsup2testprompts.jsonl")
	policy_to_prompt_to_example["ref"] = get_prompt_to_ref_example(input_file)

	policy_to_input_file = {
		"gpt2": get_predictions_file(config, "refs_base_train_gpt2", "excludesup2vsup2testprompts"),
		"gpt2d0.3": get_predictions_file(config, "refs_base_train_gpt2", "excludesup2vsup2testprompts_d0.3"),
	}
	for policy in policy_to_input_file:
		input_file = policy_to_input_file[policy]
		policy_to_prompt_to_example[policy] = get_prompt_to_pred_example(input_file)

	dataset_map = {}
	add_comparisons_dataset(policy_to_prompt_to_example, "ref", "gpt2", dataset_map)
	add_comparisons_dataset(policy_to_prompt_to_example, "ref", "gpt2d0.3", dataset_map)
	add_comparisons_dataset(policy_to_prompt_to_example, "gpt2", "gpt2d0.3", dataset_map)

	write(config, dataset_map, dataset_names=None)
