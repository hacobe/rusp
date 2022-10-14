import jsonlines
import os
import tqdm
import yaml


def read_lines(input_file, limit=-1):
	lines = []
	with open(input_file, 'r') as fin:
		for line in tqdm.tqdm(fin):
			lines.append(line)
			if len(lines) == limit:
				break
	return lines


def prepare_dataset(source_lines, reference_lines):
	assert len(source_lines) == len(reference_lines)
	examples = []
	for i in range(len(source_lines)):
		if not source_lines[i].startswith("<seg id="):
			assert not reference_lines[i].startswith("<seg id=")
			continue
		assert reference_lines[i].startswith("<seg id=")
		start = -1
		for j in range(len(source_lines[i])):
			if source_lines[i][j] == ">":
				start = j + 1
				break
		assert start != -1
		assert reference_lines[i][start-1] == ">"
		source_end = source_lines[i].find("</seg>")
		assert source_end != -1
		reference_end = reference_lines[i].find("</seg>")
		assert reference_end != -1

		examples.append({
			"prompt": source_lines[i][start:source_end],
			"reference": reference_lines[i][start:reference_end]})
	return examples


if __name__ == "__main__":
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	input_dir = os.path.join(config["data_dir"], "wmt")
	output_dir = config["data_dir"]

	train_en = read_lines(os.path.join(input_dir, "sgm16/newstest2016-ende-src.en.sgm"))
	train_de = read_lines(os.path.join(input_dir, "sgm16/newstest2016-ende-ref.de.sgm"))
	train_examples = prepare_dataset(train_en, train_de)

	train_en = read_lines(os.path.join(input_dir, "sgm17/newstest2017-ende-src.en.sgm"))
	train_de = read_lines(os.path.join(input_dir, "sgm17/newstest2017-ende-ref.de.sgm"))
	train_examples.extend(prepare_dataset(train_en, train_de))

	train_en = read_lines(os.path.join(input_dir, "sgm18/newstest2018-ende-src.en.sgm"))
	train_de = read_lines(os.path.join(input_dir, "sgm18/newstest2018-ende-ref.de.sgm"))
	train_examples.extend(prepare_dataset(train_en, train_de))

	output_file = os.path.join(output_dir, "wmt_train.jsonl")
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(train_examples)

	val_en = read_lines(os.path.join(input_dir, "sgm19/newstest2019-ende-src.en.sgm"))
	val_de = read_lines(os.path.join(input_dir, "sgm19/newstest2019-ende-ref.de.sgm"))
	val_examples = prepare_dataset(val_en, val_de)
	output_file = os.path.join(output_dir, "wmt_validation.jsonl")
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(val_examples)

	test_en = read_lines(os.path.join(input_dir, "sgm20/newstest2020-ende-src.en.sgm"))
	test_de = read_lines(os.path.join(input_dir, "sgm20/newstest2020-ende-ref.de.sgm"))
	test_examples = prepare_dataset(test_en, test_de)
	output_file = os.path.join(output_dir, "wmt_test.jsonl")
	with jsonlines.open(output_file, "w") as fout:
		fout.write_all(test_examples)