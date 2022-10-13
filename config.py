import ml_collections
import os
import yaml


def _get_model_dir(config, prefix, model):
	assert model in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
	return os.path.join(config["models_dir"], prefix + "_" + model)


def _get_checkpoint_dir(config, prefix, model):
	model_dir = _get_model_dir(config, prefix, model)
	return os.path.join(model_dir, "final_checkpoint")


def _get_refs_config(config, component, params_str):
	prefix = "refs"
	model = params_str.replace("refs_", "")

	if component == "huggingface_finetune":
		output_dir = _get_model_dir(config, prefix, model)

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		return ml_collections.ConfigDict({
			"pretrained_model_name_or_path": model,
			"model_classname": "GPT2LMHeadModel",
			"tokenizer_classname": "GPT2Tokenizer",
			"task": "DecoderOnlySeq2Seq",
			"cache_dir": config["cache_dir"],
			"train_data_file": os.path.join(config["data_dir"], "refs_train.jsonl"),
			"output_dir": output_dir,
			"seed": 0,
			"per_device_train_batch_size": 1,
			"gradient_accumulation_steps": 8,
			"gradient_checkpointing": True,
			"num_train_epochs": 1,
			"learning_rate": 5e-5,
			"evaluation_strategy": "no",
			"save_strategy": "no",
			"logging_strategy": "epoch",
			"report_to": "none"
		})
	elif component == "huggingface_generate":
		checkpoint_dir = _get_checkpoint_dir(config, prefix, model)
		return ml_collections.ConfigDict({
			"input_file": os.path.join(config["data_dir"], "refs_test.jsonl"),
			"model_dir": checkpoint_dir,
			"output_file": os.path.join(checkpoint_dir, "predictions_refs_test.jsonl"),
			"num_beams": 1,
			"num_return_sequences": 1,
			"max_num_tokens": 48,
			"top_k": 50,
			"cache_dir": config["cache_dir"],
			"include_input_data": True,
			"exclude_input_keys": "",
			"limit": -1,
			"start": 0,
			"temperature": 1.0
		})
	elif component == "evaluate":
		checkpoint_dir = _get_checkpoint_dir(config, prefix, model)
		return ml_collections.ConfigDict({
			"input_file": os.path.join(checkpoint_dir, "predictions_refs_test.jsonl"),
			"metric_file": "refs_metrics.py",
			"output_file": os.path.join(checkpoint_dir, "metrics.txt"),
		})	
	else:
		raise ValueError("Unrecognized component")


def _get_comparisons_config(config, component, params_str):
	prefix = "comparisons"
	model = params_str.replace("comparisons_", "")

	if component == "huggingface_finetune":
		output_dir = _get_model_dir(config, prefix, model)

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		return ml_collections.ConfigDict({
			"pretrained_model_name_or_path": _get_checkpoint_dir(config, "refs", model),
			"model_classname": "GPT2DoubleHeadsModel",
			"tokenizer_classname": "GPT2Tokenizer",
			"task": "MultipleChoice",
			"cache_dir": config["cache_dir"],
			"train_data_file": os.path.join(config["data_dir"], "comparisons_train.jsonl"),
			"output_dir": output_dir,
			"seed": 0,
			"per_device_train_batch_size": 1,
			"gradient_accumulation_steps": 64,
			"gradient_checkpointing": True,
			"num_train_epochs": 1,
			"learning_rate": 1.5e-5,
			"lr_scheduler_type": "cosine",
			"warmup_ratio": 0.05,
			"evaluation_strategy": "no",
			"save_strategy": "no",
			"logging_strategy": "epoch",
			"report_to": "none"
		})
	elif component == "evaluate":
		checkpoint_dir = _get_checkpoint_dir(config, prefix, model)
		return ml_collections.ConfigDict({
			"input_file": os.path.join(config["data_dir"], "comparisons_test.jsonl"),
			"metric_file": "comparisons_metrics.py",
			"output_file": os.path.join(checkpoint_dir, "metrics.txt"),
			"checkpoint_dir": checkpoint_dir,
			"cache_dir": config["cache_dir"]
		})	
	else:
		raise ValueError("Unrecognized component")


def get_config(experiment):
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)
	model_dir = config["models_dir"]
	cache_dir = config["cache_dir"]

	component, params_str = experiment.split(":")
	
	if params_str.startswith("refs_"):
		return _get_refs_config(config, component, params_str)
	elif params_str.startswith("comparisons_"):
		return _get_comparisons_config(config, component, params_str)
	else:
		raise ValueError("Invalid experiment")

