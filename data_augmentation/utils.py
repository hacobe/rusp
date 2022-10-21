import jsonlines


SPLITS = ["train", "valid", "test"]


def write_examples(examples, output_file):
	print(output_file + ": " + str(len(examples)))
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(examples)


def filter_examples(examples, disallowed_prompts):
	filtered_examples = []
	for example in examples:
		if example["prompt"] in disallowed_prompts:
			continue
		filtered_examples.append(example)
	return filtered_examples


def filter_comparison_examples(examples, allowed_policy_comps):
	filtered_examples = []
	for example in examples:
		policy0 = example["policy0"]
		policy1 = example["policy1"]
		comp = (policy0, policy1)
		opp_comp = (policy1, policy0)

		if (comp not in allowed_policy_comps) and (opp_comp not in allowed_policy_comps):
			continue

		filtered_examples.append(example)
	return filtered_examples


def extract_split(examples, split):
	split_examples = []
	for example in examples:
		if example["split"] != split:
			continue
		split_examples.append(example)
	return split_examples
