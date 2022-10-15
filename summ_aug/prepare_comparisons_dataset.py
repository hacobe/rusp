import jsonlines
import os
import yaml


def extract_raw_prompt_fn(raw_example):
	prompt = "SUBREDDIT: r/" + raw_example["subreddit"].strip() + "\n"
	prompt += "TITLE: " + raw_example["title"].strip() + "\n"
	prompt += "POST: " + raw_example["post"].strip() + "\n"
	prompt += "TL;DR:"
	return prompt


if __name__ == "__main__":
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	input_dir = os.path.join(config["data_dir"], "comparisons/")
	output_dir = config["data_dir"]

	fnames = os.listdir(input_dir)
	input_files = []
	for fname in fnames:
		if fname.find("cnndm") != -1:
			continue
		if fname.find("edit") != -1:
			continue
		if not fname.endswith(".json"):
			continue
		input_files.append(os.path.join(input_dir, fname))

	split_to_examples = {}
	for input_file in input_files:
		print(input_file)
		with jsonlines.open(input_file, "r") as fin:
			for line in fin:
				if line["split"] == "train":
					split = "train"
				elif line["split"] == "valid1":
					split = "valid"
				elif line["split"] == "valid2":
					split = "test"
				else:
					raise ValueError("Unrecognized split")

				if split not in split_to_examples:
					split_to_examples[split] = []

				example = {}
				example["example"] = line
				example["prompt"] = extract_raw_prompt_fn(line["info"])
				assert len(line["summaries"]) == 2
				assert line["choice"] in set([0, 1])
				# Note that we do not include "<|endoftext|>" here.
				example["completion0"] = " " + line["summaries"][0]["text"].strip()
				example["completion1"] = " " + line["summaries"][1]["text"].strip()
				example["choice"] = line["choice"]
				split_to_examples[split] .append(example)

	for split in split_to_examples:
		print(split + ": " + str(len(split_to_examples[split])))
		output_file = os.path.join(output_dir, "comparisons_base_" + split + ".jsonl")
		with jsonlines.open(output_file, "w") as fout:
			fout.write_all(split_to_examples[split])


