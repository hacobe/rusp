import utils

import datasets
import jsonlines
import os
import numpy as np
import transformers
import tqdm
import yaml


_MAX_NUM_TOKENS = 48


def _fmt(text):
	prefix = "(CNN)"
	start = text.find(prefix)
	if start != -1:
		for i in range(start + len(prefix), len(text)):
			if text[i].isalpha():
				start = i
				break
		text = text[start:]
	text = text.replace("\n", " ").replace(" .", ".")
	text = " ".join([word for word in text.split(" ") if word != ""])
	text = text.replace("TL;DR:", "\nTL;DR:")
	text = text.strip()
	return text


def get_ref_examples(config, tokenizer):
	dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", cache_dir=config["cache_dir"])

	ref_examples = []
	n_skipped_prompt = 0
	n_skipped_completion = 0
	for key in dataset.keys():
		if key == "validation":
			split = "valid"
		else:
			split = key
		assert split in utils.SPLITS

		for i in tqdm.tqdm(range(len(dataset[key]))):
			line = dataset[key][i]

			prompt = line["article"].strip() + "\nTL;DR:"

			example = {}
			example["prompt"] = _fmt(prompt)
			example["completion"] = " " + _fmt(line["highlights"]) + "<|endoftext|>"

			if len(tokenizer.encode(example["prompt"])) + _MAX_NUM_TOKENS > tokenizer.model_max_length:
				n_skipped_prompt += 1
				continue

			if len(tokenizer.encode(example["completion"])) > _MAX_NUM_TOKENS:
				n_skipped_completion += 1
				continue

			example["split"] = split
			example["example"] = line
			ref_examples.append(example)

	np.random.shuffle(ref_examples)

	print("n_skipped_prompt: " + str(n_skipped_prompt))
	print("n_skipped_completion: " + str(n_skipped_completion))
	print("len(ref_examples): " + str(len(ref_examples)))

	return ref_examples


def get_comparison_examples(config, tokenizer):
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

	n_skipped_prompt = 0
	n_skipped_completion = 0
	comparison_examples = []
	for input_file in input_files:
		with jsonlines.open(input_file, "r") as fin:
			for line in fin:
				assert line["split"] == "valid2"

				prompt = line["info"]["article"].strip() + "\nTL;DR:"

				example = {}
				example["prompt"] = _fmt(prompt)
				assert len(line["summaries"]) == 2
				assert line["choice"] in set([0, 1])
				# Note that we do not include "<|endoftext|>" here.
				example["completion0"] = " " + _fmt(line["summaries"][0]["text"])
				example["completion1"] = " " + _fmt(line["summaries"][1]["text"])
				example["choice"] = line["choice"]
				example["split"] = line["split"]
				example["policy0"] = line["summaries"][0]["policy"]
				example["policy1"] = line["summaries"][1]["policy"]
				example["example"] = line

				if len(tokenizer.encode(example["prompt"])) + _MAX_NUM_TOKENS > tokenizer.model_max_length:
					n_skipped_prompt += 1
					continue

				l0 = len(tokenizer.encode(example["completion0"]))
				l1 = len(tokenizer.encode(example["completion1"]))
				if l0 > _MAX_NUM_TOKENS or l1 > _MAX_NUM_TOKENS:
					n_skipped_completion += 1
					continue

				comparison_examples.append(example)

	np.random.shuffle(comparison_examples)

	print("n_skipped_prompt: " + str(n_skipped_prompt))
	print("n_skipped_completion: " + str(n_skipped_completion))
	print("len(comparison_examples): " + str(len(comparison_examples)))

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

	tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir=config["cache_dir"])

	ref_examples = get_ref_examples(config, tokenizer)
	comparison_examples = get_comparison_examples(config, tokenizer)

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