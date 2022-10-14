import datasets
import jsonlines
import os
import tqdm
import yaml


def prepare_dataset(cache_dir, output_path):
	ds = datasets.load_dataset(
		"LIUM/tedlium",
		"release1",
		split="test",
		cache_dir=cache_dir)

	examples = []
	ids = set()
	for i in tqdm.tqdm(range(len(ds))):
		if ds[i]["text"].strip() == "ignore_time_segment_in_scoring":
			continue
		assert ds[i]["audio"]["sampling_rate"] == 16000
		assert ds[i]["id"] not in ids
		ids.add(ds[i]["id"])
		examples.append({
			"id": ds[i]["id"],
			"prompt": ds[i]["audio"]["array"].tolist(),
			"reference": ds[i]["text"]
		})

	with jsonlines.open(output_path, "w") as fout:
		for example in tqdm.tqdm(examples):
			fout.write(example)


if __name__ == "__main__":
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	cache_dir = os.path.join(config["data_dir"], "tedlium")
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	output_path = os.path.join(config["data_dir"], "tedlium.jsonl")

	prepare_dataset(cache_dir, output_path)