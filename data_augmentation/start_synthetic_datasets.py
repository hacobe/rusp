import utils

import argparse
import collections
import jsonlines
import os
import nltk
import numpy as np
import spacy
import string
import transformers
import tqdm
import yaml


def read_comparison_examples(config, dataset_name):
	if dataset_name == "cnndm":
		return []

	comparison_examples = []
	input_fnames = [
		"comparisons_" + dataset_name + "_all_train.jsonl",
		"comparisons_" + dataset_name + "_all_valid.jsonl",
		"comparisons_" + dataset_name + "_all_test.jsonl"
	]
	for fname in input_fnames:
		input_file = os.path.join(config["data_dir"], fname)
		with jsonlines.open(input_file, "r") as fin:
			for line in tqdm.tqdm(fin):
				comparison_examples.append(line)
	return comparison_examples


def read_ref_examples(config, dataset_name):
	ref_examples = []
	input_fnames = [
		"refs_" + dataset_name + "_all_train.jsonl",
		"refs_" + dataset_name + "_all_valid.jsonl",
		"refs_" + dataset_name + "_all_test.jsonl"
	]
	for fname in input_fnames:
		input_file = os.path.join(config["data_dir"], fname)
		with jsonlines.open(input_file, "r") as fin:
			for line in tqdm.tqdm(fin):
				ref_examples.append(line)
	return ref_examples


def get_prompt_to_summary_and_example(ref_examples, comparison_examples):
	prompt_to_summary_and_examples = collections.defaultdict(list)

	for example in ref_examples:
		summary = example["completion"].strip().replace("<|endoftext|>", "")
		prompt_to_summary_and_examples[example["prompt"]].append((summary, example))

	for example in comparison_examples:
		if example["policy0"] == "ref":
			summary = example["completion0"].strip().replace("<|endoftext|>", "")
			prompt_to_summary_and_examples[example["prompt"]].append((summary, example))
		if example["policy1"] == "ref":
			summary = example["completion1"].strip().replace("<|endoftext|>", "")
			prompt_to_summary_and_examples[example["prompt"]].append((summary, example))

	prompt_to_summary_and_example = {}
	for prompt in prompt_to_summary_and_examples:
		summary_and_examples = prompt_to_summary_and_examples[prompt]
		# sort from largest to smallest summary
		summary_and_examples.sort(key=lambda x: -len(x[0]))
		prompt_to_summary_and_example[prompt] = summary_and_examples[0]

	return prompt_to_summary_and_example


def get_maskedref_prompt(summary, example, tokenizer, mask_prop):
	summary = summary.strip()

	tokens = tokenizer.encode(summary)

	num_mask_tokens = int(mask_prop * len(tokens))

	index = np.random.randint(0, len(tokens) - num_mask_tokens)

	new_tokens = []
	for j in range(len(tokens)):
		if j == index:
			new_tokens.append(tokenizer.cls_token_id)
		elif j > index and j < index + num_mask_tokens:
			continue
		else:
			new_tokens.append(tokens[j])

	new_summary = tokenizer.decode(new_tokens).strip()

	if "subreddit" in example["example"]:
		subreddit = example["example"]["subreddit"]
	elif "info" in example["example"]:
		subreddit = example["example"]["info"]["subreddit"]
	else:
		subreddit = None

	if subreddit:
		new_prompt = "SUBREDDIT: r/" + subreddit.strip() + "\nMASKED: " + new_summary + "\nTL;DR:"
	else:
		new_prompt = "MASKED: " + new_summary + "\nTL;DR:"

	return new_prompt


def add_dataset(
	prompt_to_summary_and_example,
	dataset_map,
	dataset_name,
	prompt_name,
	tokenizer,
	train_limit=None,
	test_limit=None,
	test_prompts=set()):
	doc_tokenizer = nltk.tokenize.RegexpTokenizer(r"\s+", gaps=True)
	nlp = spacy.load('en_core_web_sm')
	stops = nltk.corpus.stopwords.words('english')

	train_examples = []
	test_examples = []
	new_prompts = set()
	for prompt in tqdm.tqdm(prompt_to_summary_and_example):
		summary, example = prompt_to_summary_and_example[prompt]

		if prompt_name == "unmodifiedprompt":
			new_prompt = example["prompt"]
		elif prompt_name == "maskedrefprompt" and dataset_name == "tldr":
			new_prompt = get_maskedref_prompt(
				summary, example, tokenizer, mask_prop=1./3)
			assert new_prompt.find("SUBREDDIT:") != -1
		elif prompt_name == "maskedrefprompt" and dataset_name == "cnndm":
			text = summary.strip()
			summary_ids = tokenizer.encode(text)
			if len(summary_ids) > 48:
				continue

			sents = text.split("\n")
			
			index = np.random.randint(0, len(sents))

			add_period = sents[index].endswith(" .")

			sents[index] = "[CLS]"
			if add_period:
				sents[index] += " ."

			new_prompt = "MASKED: " + "\n".join(sents).strip() + "\nTL;DR:"
		elif prompt_name == "shuffledprompt":
			if "post" in example["example"]:
				text = example["example"]["post"]
			elif "info" in example["example"] and "post" in example["example"]["info"]:
				text = example["example"]["info"]["post"]
			elif "article" in example["example"]:
				text = example["example"]["article"]
			elif "article" in example["example"] and "article" in example["example"]["info"]:
				text = example["example"]["info"]["article"]
			else:
				text = None

			if text is None:
				continue

			if not text.strip():
				continue

			sents = nltk.sent_tokenize(text)
			
			np.random.shuffle(sents)
			new_prompt = " ".join(sents).strip() + "\nTL;DR:"
		else:
			raise ValueError("Unrecognized prompt_name: <" + prompt_name + ">")

		if new_prompt in new_prompts:
			continue
		new_prompts.add(new_prompt)

		ref_example = {}
		ref_example["prompt"] = new_prompt
		ref_example["completion"] = " " + summary.strip() + "<|endoftext|>"
		ref_example["example"] = example

		if prompt in test_prompts:
			test_examples.append(ref_example)
		else:
			train_examples.append(ref_example)

	np.random.shuffle(train_examples)
	np.random.shuffle(test_examples)

	key = "refs_" + dataset_name + "_" + prompt_name

	if train_limit is None:
		train_limit = len(train_examples)

	if test_limit is None:
		test_limit = len(test_examples)

	train_examples = train_examples[:train_limit]
	test_examples = test_examples[:test_limit]

	if not test_examples:
		dataset_map[key] = train_examples
		return

	dataset_map[key + "_train"] = train_examples
	dataset_map[key + "_test"] = test_examples


def main(args):
	dataset_name = args.dataset_name

	np.random.seed(0)

	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir=config["cache_dir"])
	tokenizer.pad_token = tokenizer.unk_token
	tokenizer.add_special_tokens({"cls_token": "[CLS]"})

	# Get as many references as possible.
	# Using the references that appear in the training set is not sufficient.
	ref_examples = read_ref_examples(config, dataset_name)
	comparison_examples = read_comparison_examples(config, dataset_name)
	prompt_to_summary_and_example = get_prompt_to_summary_and_example(ref_examples, comparison_examples)

	dataset_map = {}

	add_dataset(
		prompt_to_summary_and_example=prompt_to_summary_and_example,
		dataset_map=dataset_map,
		dataset_name=dataset_name,
		prompt_name="unmodifiedprompt",
		tokenizer=tokenizer,
		train_limit=40000)

	unmodified_prompt_to_summary_and_example = {}
	test_prompts = set()
	for example in dataset_map["refs_" + dataset_name + "_unmodifiedprompt"]:
		unmodified_prompt_to_summary_and_example[example["prompt"]] = (
			prompt_to_summary_and_example[example["prompt"]])
		test_prompts.add(example["prompt"])

	add_dataset(
		prompt_to_summary_and_example=prompt_to_summary_and_example,
		dataset_map=dataset_map,
		dataset_name=dataset_name,
		prompt_name="maskedrefprompt",
		tokenizer=tokenizer,
		train_limit=10000,
		test_prompts=test_prompts)

	add_dataset(
		prompt_to_summary_and_example=unmodified_prompt_to_summary_and_example,
		dataset_map=dataset_map,
		dataset_name=dataset_name,
		prompt_name="shuffledprompt",
		tokenizer=tokenizer)

	for key in dataset_map:
		output_file = os.path.join(config["data_dir"], key + ".jsonl")
		utils.write_examples(dataset_map[key], output_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Start preparation of synthetic datasets.")
	parser.add_argument("--dataset_name", default="tldr", type=str)
	args = parser.parse_args()
	main(args)