import datasets
import jsonlines
import os
import tqdm
import yaml


def write(examples, output_path):
	with jsonlines.open(output_path, "w") as fout:
		pass
	with jsonlines.open(output_path, "a") as fout:
		for example in tqdm.tqdm(examples):
			fout.write(example)


def prepare_dataset(kind, split, ids, cache_dir):
	ds = datasets.load_dataset(
		"librispeech_asr",
		kind,
		split=split,
		cache_dir=cache_dir)

	examples = []
	for i in tqdm.tqdm(range(len(ds))):
		assert ds[i]["id"] not in ids
		ids.add(ds[i]["id"])
		examples.append({
			"id": ds[i]["id"],
			"prompt": ds[i]["audio"]["array"].tolist(),
			"reference": ds[i]["text"],
			"kind": kind
		})

	return examples


if __name__ == "__main__":
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	output_dir = config["data_dir"]
	test_clean_fname = "librispeechclean.jsonl"
	test_other_fname = "librispeechother.jsonl"

	cache_dir = os.path.join(config["data_dir"], "librispeech")
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	test_clean_path = os.path.join(output_dir, test_clean_fname)
	test_other_path = os.path.join(output_dir, test_other_fname)

	ids = set()

	test_clean_examples = prepare_dataset("clean", "test", ids, cache_dir)
	write(test_clean_examples, test_clean_path)

	test_other_examples = prepare_dataset("other", "test", ids, cache_dir)
	write(test_other_examples, test_other_path)
