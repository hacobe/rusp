import ml_collections
import os
import yaml


MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def _get_model_dir(config, params_str):
	return os.path.join(config["models_dir"], params_str)


def _get_checkpoint_dir(config, params_str):
	model_dir = _get_model_dir(config, params_str)
	return os.path.join(model_dir, "final_checkpoint")


def _get_refs_config(config, component, params_str):
	# `params_str` has to have the format "refs_<data>_<model>"
	# in order for the reward model fine-tuning on
	# comparisons to know where to find the summarization
	# model to use at initialization.
	parts = params_str.split("_")
	assert len(parts) == 3
	assert parts[0] == "refs"
	data = parts[1]
	model = parts[2]
	assert model in MODELS

	if data == "base":
		pretrained_model_name_or_path = model
		task = "DecoderOnlySeq2Seq"
	elif data == "masked":
		pretrained_model_name_or_path = _get_checkpoint_dir(
			config, "refs_base_" + model)
		task = "DecoderOnlyMaskedSeq2Seq"
	else:
		raise ValueError("Unrecognized data <" + data + ">")

	if component == "huggingface_finetune":
		output_dir = _get_model_dir(config, params_str)

		assert not os.path.exists(output_dir)
		os.makedirs(output_dir)

		return ml_collections.ConfigDict({
			"pretrained_model_name_or_path": pretrained_model_name_or_path,
			"model_classname": "GPT2LMHeadModel",
			"tokenizer_classname": "GPT2Tokenizer",
			"task": task,
			"cache_dir": config["cache_dir"],
			"train_data_file": os.path.join(config["data_dir"], "refs_{0}_train.jsonl".format(data)),
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
		checkpoint_dir = _get_checkpoint_dir(config, params_str)
		return ml_collections.ConfigDict({
			"input_file": os.path.join(config["data_dir"], "refs_{0}_test.jsonl".format(data)),
			"model_dir": checkpoint_dir,
			"output_file": os.path.join(checkpoint_dir, "predictions_refs_{0}_test.jsonl".format(data)),
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
		checkpoint_dir = _get_checkpoint_dir(config, params_str)
		return ml_collections.ConfigDict({
			"input_file": os.path.join(checkpoint_dir, "predictions_refs_{0}_test.jsonl".format(data)),
			"metric_file": "refs_metrics.py",
			"output_file": os.path.join(checkpoint_dir, "metrics.txt"),
		})	
	else:
		raise ValueError("Unrecognized component")


def _get_comparisons_filename(config, params):
	return "comparisons_{0}_{1}.jsonl".format(params["data"], params["split"])


def _parse_comparisons_params_str(params_str):
	params = {}
	if not params_str:
		return params

	for param_str in params_str.replace("comparisons_", "").split("_"):
		if param_str in MODELS:
			params["model"] = param_str
		elif param_str in (
			"base",
			"sup2vsup2",
			"refvsup2",
			"sup2vsup2+refvsup2",
			"refvsup2policy",
			"refvsup1",
			"refvsup1policy",
			"refvdup",
			"refvdrop"):
			params["data"] = param_str
		elif param_str.startswith("n") and param_str[1].isdigit():
			num_str = "0"
			suffix = ""
			for i in range(1, len(param_str)):
				ch = param_str[i]
				if ch.isdigit():
					num_str += ch
				else:
					suffix = param_str[i:]
					break
			num = int(num_str)
			if suffix == "":
				num *= 1
			elif suffix == "k":
				num *= 1000
			elif suffix == "all":
				num = -1
			else:
				raise ValueError("Unrecognized suffix: <" + suffix + ">")
			params["train_data_limit"] = num
		elif param_str in ("train", "valid", "test"):
			params["split"] = param_str
		else:
			raise ValueError("Unrecognized params: <" + param_str + ">")
	return params


def _get_comparisons_config(config, component, params_str):
	assert params_str.startswith("comparisons_")
	parts = params_str.split("#")
	assert len(parts) == 1 or len(parts) == 2
	train_params_str = parts[0]
	eval_params_str = "" if len(parts) == 1 else parts[1]
	train_params = _parse_comparisons_params_str(train_params_str)
	eval_params = _parse_comparisons_params_str(eval_params_str)

	if component == "huggingface_finetune":
		output_dir = _get_model_dir(config, train_params_str)

		assert not os.path.exists(output_dir), output_dir
		os.makedirs(output_dir)

		train_data_file = os.path.join(
			config["data_dir"], _get_comparisons_filename(config, train_params))
		pretrained_model_name_or_path = _get_checkpoint_dir(
			config, "refs_base_" + train_params["model"])

		return ml_collections.ConfigDict({
			"pretrained_model_name_or_path": pretrained_model_name_or_path,
			"model_classname": "GPT2DoubleHeadsModel",
			"tokenizer_classname": "GPT2Tokenizer",
			"task": "MultipleChoice",
			"train_data_limit": train_params.get("train_data_limit", -1),
			"cache_dir": config["cache_dir"],
			"train_data_file": train_data_file,
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
		input_file = os.path.join(config["data_dir"], _get_comparisons_filename(config, eval_params))
		checkpoint_dir = _get_checkpoint_dir(config, train_params_str)

		base_filename = _get_comparisons_filename(config, eval_params)
		output_file = os.path.join(checkpoint_dir,
			base_filename.replace("comparisons_", "metrics_").replace(".jsonl", ".txt"))
		evaluated_examples_file = os.path.join(checkpoint_dir,
			base_filename.replace("comparisons_", "evaluated_examples_"))

		return ml_collections.ConfigDict({
			"input_file": input_file,
			"metric_file": "comparisons_metrics.py",
			"output_file": output_file,
			"evaluated_examples_file": evaluated_examples_file,
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

