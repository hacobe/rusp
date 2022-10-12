import jsonlines
import rouge
import numpy as np
import tqdm


def compute(examples):
	rouges = []
	for example in tqdm.tqdm(examples):
		prediction = example["predictions"][0]["text"].strip()
		reference = example["completion"].replace("<|endoftext|>", "").strip()
		scores = rouge.Rouge().get_scores(prediction, reference)
		score = scores[0]["rouge-1"]["f"]
		rouges.append(score)
	return {"rouge": np.mean(rouges)}