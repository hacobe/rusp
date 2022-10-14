import language_models as lm

import jsonlines
import transformers
import torch
import os
import numpy as np
import sklearn.metrics
import tqdm


def compute(examples, checkpoint_dir=None, cache_dir=None):
	tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
	tokenizer.pad_token = tokenizer.unk_token
	tokenizer.add_special_tokens({"cls_token": "[CLS]"})

	model = transformers.GPT2DoubleHeadsModel.from_pretrained(checkpoint_dir)
	model.eval()
	model.cuda()

	data_collator = lm.finetune._DataCollatorForMultipleChoice(tokenizer)

	new_examples = []
	y_prob = []
	y_pred = []
	y_true = []
	i = 0
	for line in tqdm.tqdm(examples):
		example = lm.utils.build_example_for_multiple_choice(
			line["prompt"],
			line["completion0"],
			line["completion1"],
			line["choice"],
			tokenizer)

		batch = data_collator([example])

		with torch.no_grad():
			input_ids = torch.tensor(batch["input_ids"]).cuda()
			mc_token_ids = torch.tensor(batch["mc_token_ids"]).cuda()
			outputs = model(input_ids, mc_token_ids=mc_token_ids)

		logprobs = torch.nn.functional.log_softmax(outputs.mc_logits, dim=1)
		probs = torch.exp(logprobs)
		prob = probs[0, 1].tolist()
		prediction = torch.argmax(probs).tolist()

		new_example = {}
		new_example["example"] = line
		new_example["prob"] = prob
		new_example["prediction"] = prediction
		new_example["choice"] = line["choice"]
		new_examples.append(new_example)

		y_prob.append(prob)
		y_pred.append(prediction)
		y_true.append(line["choice"])

		i += 1

	y_prob = np.array(y_prob)
	y_pred = np.array(y_pred)
	y_true = np.array(y_true)

	metrics = {
		"auc": sklearn.metrics.roc_auc_score(y_true, y_prob),
		"acc": np.mean(y_pred == y_true)
	}

	return metrics
