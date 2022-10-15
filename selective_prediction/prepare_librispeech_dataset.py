import datasets
import jsonlines
import os
import numpy as np
import tqdm
import yaml


def write(examples, output_path):
	with jsonlines.open(output_path, "w") as fout:
		pass
	with jsonlines.open(output_path, "a") as fout:
		for example in tqdm.tqdm(examples):
			fout.write(example)


def prepare_and_write_train_dataset(k, ids, output_path, cache_dir):
	train100 = datasets.load_dataset(
		"librispeech_asr",
		"clean",
		split="train.100",
		cache_dir=cache_dir)	

	train360 = datasets.load_dataset(
		"librispeech_asr",
		"clean",
		split="train.360",
		cache_dir=cache_dir)	

	train500 = datasets.load_dataset(
		"librispeech_asr",
		"other",
		split="train.500",
		cache_dir=cache_dir)	

	n100 = len(train100)
	n360 = len(train360)
	n500 = len(train500)

	indices100 = np.arange(n100)
	indices360 = np.arange(n360)
	indices500 = np.arange(n500)

	np.random.seed(0)
	np.random.shuffle(indices100)
	np.random.shuffle(indices360)
	np.random.shuffle(indices500)

	n = n100 + n360 + n500
	k100 = int((n100/n) * k)
	k360 = int((n360/n) * k)
	k500 = k - k100 - k360

	with jsonlines.open(output_path, "w") as fout:
		pass

	with jsonlines.open(output_path, "a") as fout:
		for i in tqdm.tqdm(range(k100)):
			j = int(indices100[i])
			assert train100[j]["id"] not in ids
			ids.add(train100[j]["id"])
			fout.write({
				"id": train100[j]["id"],
				"prompt": train100[j]["audio"]["array"].tolist(),
				"reference": train100[j]["text"],
				"split": "train100"
			})

		for i in tqdm.tqdm(range(k360)):
			j = int(indices360[i])
			assert train360[j]["id"] not in ids
			ids.add(train360[j]["id"])
			fout.write({
				"id": train360[j]["id"],
				"prompt": train360[j]["audio"]["array"].tolist(),
				"reference": train360[j]["text"],
				"split": "train360"
			})

		for i in tqdm.tqdm(range(k500)):
			j = int(indices500[i])
			assert train500[j]["id"] not in ids
			ids.add(train500[j]["id"])
			fout.write({
				"id": train500[j]["id"],
				"prompt": train500[j]["audio"]["array"].tolist(),
				"reference": train500[j]["text"],
				"split": "train500"
			})


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

	cache_dir = os.path.join(config["data_dir"], "librispeech")
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	train_fname = "librispeech_train.jsonl"
	val_fname = "librispeech_validation.jsonl"
	test_clean_fname = "librispeechclean_test.jsonl"
	test_other_fname = "librispeechother_test.jsonl"

	train_path = os.path.join(output_dir, train_fname)
	val_path = os.path.join(output_dir, val_fname)
	test_clean_path = os.path.join(output_dir, test_clean_fname)
	test_other_path = os.path.join(output_dir, test_other_fname)

	ids = set()

	train_examples = prepare_and_write_train_dataset(10000, ids, train_path, cache_dir)

	val_examples = prepare_dataset("clean", "validation", ids, cache_dir)
	val_examples.extend(prepare_dataset("other", "validation", ids, cache_dir))
	write(val_examples, val_path)

	test_clean_examples = prepare_dataset("clean", "test", ids, cache_dir)
	write(test_clean_examples, test_clean_path)

	test_other_examples = prepare_dataset("other", "test", ids, cache_dir)
	write(test_other_examples, test_other_path)