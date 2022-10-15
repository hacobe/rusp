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

	for i in tqdm.tqdm(range(len(examples))):
		example = examples[i]

		multiple_choice_example = lm.utils.build_example_for_multiple_choice(
			example["prompt"],
			example["completion0"],
			example["completion1"],
			example["choice"],
			tokenizer)

		batch = data_collator([multiple_choice_example])

		with torch.no_grad():
			input_ids = torch.tensor(batch["input_ids"]).cuda()
			mc_token_ids = torch.tensor(batch["mc_token_ids"]).cuda()
			outputs = model(input_ids, mc_token_ids=mc_token_ids)

		logprobs = torch.nn.functional.log_softmax(outputs.mc_logits, dim=1)
		probs = torch.exp(logprobs)
		prob = probs[0, 1].tolist()
		prediction = torch.argmax(probs).tolist()

		examples[i]["prob"] = prob
		examples[i]["prediction"] = prediction

	y_prob = np.array([x["prob"] for x in examples])
	y_pred = np.array([x["prediction"] for x in examples])
	y_true = np.array([x["choice"] for x in examples])

	metrics = {
		"auc": sklearn.metrics.roc_auc_score(y_true, y_prob),
		"acc": np.mean(y_pred == y_true)
	}

	return metrics
