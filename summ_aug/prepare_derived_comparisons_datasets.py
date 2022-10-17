import collections
import jsonlines
import os
import nltk
import numpy as np
import string
import transformers
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


def split_by_prompt(examples):
	prompts = set()
	for example in examples:
		prompts.add(example["prompt"])

	prompts_list = sorted(list(prompts))
	np.random.shuffle(prompts_list)
	n = len(prompts_list)
	n_train = n // 2
	train_prompts = set(prompts_list[:n_train])
	train_examples = []
	test_examples = []
	for example in examples:
		if example["prompt"] in train_prompts:
			train_examples.append(example)
		else:
			test_examples.append(example)
	np.random.shuffle(train_examples)
	np.random.shuffle(test_examples)

	return train_examples, test_examples


def add_policy_comp_dataset(
	examples,
	dataset_map,
	dataset_name,
	allowed_policy_comps,
	should_split):
	filtered_examples = []
	for example in examples:
		comp = (example["policy0"], example["policy1"])
		opp_comp = (example["policy1"], example["policy0"])

		if (comp not in allowed_policy_comps) and (opp_comp not in allowed_policy_comps):
			continue

		filtered_examples.append(example)

	np.random.shuffle(filtered_examples)

	if not should_split:
		dataset_map[dataset_name + "_train"] = filtered_examples
		return

	train_examples, test_examples = split_by_prompt(filtered_examples)

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


def segment(summary, word_punct_tokenizer):
	if len(summary) >= 1:
		assert summary[0] not in (" ", "\n", "\r", "\t")
		assert summary[-1] not in (" ", "\n", "\r", "\t")

	token_spans = [span for span in word_punct_tokenizer.span_tokenize(summary)]
	sent_spans = []
	last = 0
	num_nonpunc_tokens = 0
	for start, end in token_spans:
		token = summary[start:end]

		is_punc = False
		for punc in [".", "?", "!", ",", ";"]:
			if token.startswith(punc):
				is_punc = True
				break

		if not is_punc:
			num_nonpunc_tokens += 1

		if (is_punc and num_nonpunc_tokens >= 4):
			sent_spans.append((last, end))
			last = end
			num_nonpunc_tokens = 0
	if len(summary) - last > 0:
		if len(sent_spans) != 0 and num_nonpunc_tokens < 4:
			sent_spans[-1] = (sent_spans[-1][0], len(summary))
		else:
			sent_spans.append((last, len(summary)))

	summary_parts = [summary[s[0]:s[1]] for s in sent_spans]
	assert "".join(summary_parts) == summary

	return sent_spans


def corrupt_ref_by_dup(ref, word_punct_tokenizer):
	summary = ref.strip()

	sent_spans = segment(summary, word_punct_tokenizer)

	if len(sent_spans) == 1:
		return None

	index = np.random.randint(0, len(sent_spans))

	prefix_end = sent_spans[index][0]
	suffix_start = sent_spans[index][1]
	if index + 1 < len(sent_spans):
		suffix_end = sent_spans[index+1][1]
	else:
		suffix_end = len(summary)

	prefix = summary[:prefix_end].strip()
	middle = summary[prefix_end:suffix_start].strip()
	suffix = summary[suffix_start:suffix_end].strip()

	new_summary = ""
	new_summary += prefix
	n_reps = 0
	while len(new_summary) + len(suffix) + len(middle) + 1 <= len(summary):
		m = [ch for ch in middle]
		if new_summary.endswith(",") or new_summary.endswith(";"):
			if (len(m) >= 2) and (m[0:2] in ("I", "I'")):
				m[0] = m[0] # do nothing
			elif m[0] == m[0].upper():
				m[0] = m[0].lower()

		if new_summary != "":
			new_summary += " "

		new_summary += "".join(m)
		n_reps += 1

		if n_reps == 2:
			break

	if n_reps == 1:
		return None

	new_summary = new_summary.strip()
	if suffix != "":
		new_summary += " "
	new_summary += suffix

	j = index + 1
	while (j < len(sent_spans)) and (len(new_summary) + sent_spans[j][1] - sent_spans[j][0] + 1 <= len(summary)):
		start, end = sent_spans[j]
		part = summary[start:end]
		if part != "":
			new_summary += " "
		new_summary += part
		j += 1

	if new_summary.endswith(",") or new_summary.endswith(";"):
		new_summary = new_summary[:-1]

	return new_summary


def count_num_tokens(text, word_punct_tokenizer):
	token_spans = [span for span in word_punct_tokenizer.span_tokenize(text)]
	num_nonpunc_tokens = 0
	num_punc_tokens = 0
	for token_start, token_end in token_spans:
		token = text[token_start:token_end]
		is_punc = False
		for punc in string.punctuation:
			if token.startswith(punc):
				is_punc = True
				break
		if is_punc:
			num_punc_tokens += 1
		else:
			num_nonpunc_tokens += 1
	return {"num_punc_tokens": num_punc_tokens, "num_nonpunc_tokens": num_nonpunc_tokens}


def corrupt_ref_by_drop(ref, word_punct_tokenizer):
	summary = ref.strip()

	sent_spans = segment(summary, word_punct_tokenizer)

	if len(sent_spans) == 1:
		return None

	summary_parts = [summary[s[0]:s[1]] for s in sent_spans]
	assert "".join(summary_parts) == summary

	num_nonpunc_tokens_in_sent_spans = []
	for start, end in sent_spans:
		summary_part = summary[start:end]
		num_nonpunc_tokens = count_num_tokens(
			summary_part, word_punct_tokenizer)["num_nonpunc_tokens"]
		num_nonpunc_tokens_in_sent_spans.append(num_nonpunc_tokens)

	eligible_indices = []
	for j, num_nonpunc_tokens in enumerate(num_nonpunc_tokens_in_sent_spans):
		if num_nonpunc_tokens >= 5:
			eligible_indices.append(j)

	if len(eligible_indices) <= 1:
		return None

	index = np.random.randint(0, len(eligible_indices))

	included_summary_parts = []
	prev_part = ""
	for j in range(len(summary_parts)):
		if j == eligible_indices[index]:
			continue
		part = summary_parts[j].strip()
		if (prev_part != "") and (prev_part[-1] in (",", ";")) and (part != "") and (
			part[0] == part[0].upper()) and (part[0:2] not in ("I ")) and (j - 1 == eligible_indices[index]):
			part = part[0].lower() + part[1:]
		included_summary_parts.append(part)
		prev_part = part

	new_summary = " ".join(included_summary_parts).strip()
	if new_summary.endswith(",") or new_summary.endswith(";"):
		new_summary = new_summary[:-1]

	return new_summary


def add_corrupt_ref_dataset(
	prompt_to_ref_examples,
	dataset_map,
	dataset_name,
	corrupt_fn,
	limit=None):
	assert dataset_name.startswith("refv")
	corrupt_policy_name = dataset_name.replace("refv", "")

	word_punct_tokenizer = nltk.tokenize.WordPunctTokenizer()

	examples = []
	for prompt in tqdm.tqdm(prompt_to_ref_examples):
		ref_examples = prompt_to_ref_examples[prompt]
		refs = [e["summary"] for e in ref_examples]
		# sort so that longest completion is first.
		refs.sort(key=lambda x: -len(x))
		ref = refs[0]

		corrupt_ref = corrupt_fn(ref, word_punct_tokenizer)

		if not corrupt_ref:
			continue

		completion0 = " " + ref.strip()
		completion1 = " " + corrupt_ref.strip()
		policy0 = "ref"
		policy1 = corrupt_policy_name
		choice = 0

		r = np.random.random()
		if r <= 0.5:
			completion0, completion1 = completion1, completion0
			policy0, policy1 = policy1, policy0
			choice = 1 - choice

		example = {
			"prompt": prompt,
			"completion0": completion0,
			"completion1": completion1,
			"policy0": policy0,
			"policy1": policy1,
			"choice": choice
		}
		examples.append(example)

	np.random.shuffle(examples)

	if limit is None:
		limit = len(examples)

	dataset_map[dataset_name + "_train"] = examples[:limit]


def write_masked_refs_dataset(config, prompt_to_ref_examples):
	word_punct_tokenizer = nltk.tokenize.WordPunctTokenizer()
	tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir=config["cache_dir"])
	tokenizer.pad_token = tokenizer.unk_token
	tokenizer.add_special_tokens({"cls_token": "[CLS]"})

	examples = []
	prompts = set()
	for prompt in tqdm.tqdm(prompt_to_ref_examples):
		ref_examples = prompt_to_ref_examples[prompt]
		# sort so that longest completion is first.
		ref_examples.sort(key=lambda x: -len(x["summary"]))
		ref_example = ref_examples[0]

		if ref_example["prompt"] in prompts:
			continue
		prompts.add(ref_example["prompt"])

		summary = ref_example["summary"].strip()

		sent_spans = segment(summary, word_punct_tokenizer)

		if len(sent_spans) == 1:
			continue

		summary_parts = [summary[s[0]:s[1]] for s in sent_spans]
		assert "".join(summary_parts) == summary

		num_nonpunc_tokens_in_sent_spans = []
		for start, end in sent_spans:
			summary_part = summary[start:end]
			num_nonpunc_tokens = count_num_tokens(
				summary_part, word_punct_tokenizer)["num_nonpunc_tokens"]
			num_nonpunc_tokens_in_sent_spans.append(num_nonpunc_tokens)

		eligible_indices = []
		for j, num_nonpunc_tokens in enumerate(num_nonpunc_tokens_in_sent_spans):
			if num_nonpunc_tokens >= 5:
				eligible_indices.append(j)

		if len(eligible_indices) <= 1:
			continue

		index = np.random.randint(0, len(eligible_indices))

		included_summary_parts = []
		mask_part = None
		for j in range(len(summary_parts)):
			summary_part = summary_parts[j].strip()
			if j == eligible_indices[index]:
				summary_part_prefix = summary_part
				summary_part_suffix = ""
				if len(summary_part) > 0 and summary_part[-1] in string.punctuation:
					start = len(summary_part) - 1
					for k in range(len(summary_part)-1, -1, -1):
						if summary_part[k] not in string.punctuation:
							start = k + 1
							break
					summary_part_prefix = summary_part[:start]
					summary_part_suffix = summary_part[start:]

				summary_part_prefix_tokens = tokenizer.encode(summary_part_prefix)
				summary_part_suffix_tokens = tokenizer.encode(summary_part_suffix)
				mask_tokens = ["[CLS]"] * len(summary_part_prefix_tokens)
				mask_part = " ".join(mask_tokens) + summary_part_suffix
				new_summary_part = mask_part 
			else:
				new_summary_part = summary_part
			included_summary_parts.append(new_summary_part)
		assert mask_part is not None

		new_summary = " ".join(included_summary_parts).strip()
		new_prompt = "SUBREDDIT: r/" + ref_example["subreddit"].strip() + "\n" + new_summary + "\nTL;DR:"

		example = {}
		example["prompt"] = new_prompt
		example["completion"] = " " + summary + "<|endoftext|>"
		example["example"] = ref_example
		examples.append(example)

	np.random.shuffle(examples)

	train_examples, test_examples = split_by_prompt(examples)

	output_file = os.path.join(config["data_dir"], "refs_masked_train.jsonl")
	print(output_file + ": " + str(len(train_examples)))
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(train_examples)

	output_file = os.path.join(config["data_dir"], "refs_masked_test.jsonl")
	print(output_file + ": " + str(len(test_examples)))
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(test_examples)


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

	filtered_base_dataset = []
	for example in base_dataset:
		if example["prompt"] in sup2vsup2_test_prompts:
			continue
		filtered_base_dataset.append(example)

	filtered_prompt_to_ref_examples = collections.defaultdict(list)
	with jsonlines.open(os.path.join(config["data_dir"], "refs_base_train.jsonl"), "r") as fin:
		for example in tqdm.tqdm(fin):
			if example["prompt"] in sup2vsup2_test_prompts:
				continue
			filtered_prompt_to_ref_examples[example["prompt"]].append(example)

	write_masked_refs_dataset(config, filtered_prompt_to_ref_examples)

	add_policy_comp_dataset(
		examples=filtered_base_dataset,
		dataset_map=dataset_map,
		dataset_name="refvsup2",
		allowed_policy_comps=set([("ref", "sup2")]),
		should_split=False)

	dataset_map["sup2vsup2+refvsup2_train"] = (
		dataset_map["sup2vsup2_train"] + dataset_map["refvsup2_train"])
	np.random.shuffle(dataset_map["sup2vsup2+refvsup2_train"])

	dataset_map["refvsup2policy_train"] = replace_choice(
		dataset_map["refvsup2_train"],
		better_than_policy_pairs=set([("ref", "sup2")]))

	add_policy_comp_dataset(
		examples=filtered_base_dataset,
		dataset_map=dataset_map,
		dataset_name="refvsup1",
		allowed_policy_comps=set([("ref", "sup1")]),
		should_split=False)

	dataset_map["refvsup1policy_train"] = replace_choice(
		dataset_map["refvsup1_train"],
		better_than_policy_pairs=set([("ref", "sup1")]))

	add_corrupt_ref_dataset(
		prompt_to_ref_examples=filtered_prompt_to_ref_examples,
		dataset_map=dataset_map,
		dataset_name="refvdup",
		corrupt_fn=corrupt_ref_by_dup)

	add_corrupt_ref_dataset(
		prompt_to_ref_examples=filtered_prompt_to_ref_examples,
		dataset_map=dataset_map,
		dataset_name="refvdrop",
		corrupt_fn=corrupt_ref_by_drop)

	write(config, dataset_map, dataset_names=None)

