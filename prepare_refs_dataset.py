import jsonlines
import os
import tqdm
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

	input_dir = config["data_dir"]

	input_filenames = ["train.jsonl", "valid.jsonl", "test.jsonl"]
	output_dir = config["data_dir"]

	for filename in input_filenames:
		split = filename.replace(".jsonl", "")
		path = os.path.join(input_dir, filename)

		examples = []
		with jsonlines.open(path, "r") as fin:
			for raw_example in tqdm.tqdm(fin):
				example = {}
				for key in raw_example:
					example[key] = raw_example[key]
				example["prompt"] = extract_raw_prompt_fn(raw_example)
				example["completion"] = " " + raw_example["summary"] + " <|endoftext|>"
				examples.append(example)

		output_file = os.path.join(output_dir, "refs_" + split + ".jsonl")
		with jsonlines.open(output_file, "w") as fout:
			fout.write_all(examples)
