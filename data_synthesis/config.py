import ml_collections
import os
import yaml


MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt-neo", "bart"]


def _get_model_dir(config, params_str):
	return os.path.join(config["models_dir"], params_str)


def _get_model_classname(model):
	if model == "gpt-neo":
		return "GPTNeoForCausalLM"
	elif model == "bart":
		return "BartForConditionalGeneration"
	return "GPT2LMHeadModel"


def _get_tokenizer_classname(model):
	if model == "bart":
		return "BartTokenizer"
	return "GPT2Tokenizer"


def _get_checkpoint_dir(config, params_str, checkpoint):
	model_dir = _get_model_dir(config, params_str)
	return os.path.join(model_dir, checkpoint)


def _get_refs_checkpoint_dir(config, params):
	params_str = "refs"
	params_str += "_" + params["data"]
	params_str += "_all_train"
	params_str += "_" + params["model"]
	return _get_checkpoint_dir(config, params_str, "final_checkpoint")


def _get_data_filename(params):
	filename = params["mode"]
	filename += "_" + params["data"]
	filename += "_" + params["policy_comp"]

	if "split" in params:
		filename += "_" + params["split"]

	filename += ".jsonl"
	return filename


def _get_predictions_filename(params_str):
	return "predictions_" + params_str + ".jsonl"


def _parse_params_str(params_str):
	params = {}
	if not params_str:
		return params

	for param_str in params_str.split("_"):
		if param_str in ("comparisons", "refs"):
			params["mode"] = param_str
		elif param_str in MODELS:
			params["model"] = param_str
		elif param_str in ("tldr", "cnndm"):
			params["data"] = param_str
		elif param_str in (
			"all",
			"refvsup+supvsup",
			"unmodifiedprompt",
			"maskedrefprompt",
			"maskedprompt",
			"shuffledprompt",
			"gpt2vgpt2d0.2",
			"refvgpt2d0.2",
			"refvgpt2",
			"refvgpt2maskedrefprompt",
			"refvgpt2shuffledprompt",
			"gpt2vgpt2maskedrefprompt",
			"gpt2vgpt2shuffledprompt",
			"gpt2vgpt2d0.2+refvgpt2maskedrefprompt",
			"supvsup",
			"gpt2-xlvgpt2-xld0.2",
			"refvgpt2-xlmaskedrefprompt",
			"refvgpt2-xlmaskedprompt",
			"refvgpt2-xl+gpt2-xlvgpt2-xl",
			"pseudogpt2n1k",
			"pseudogpt2n5k",
			"pseudogpt2-xln1k",
			"pseudogpt2-xln5k",
			"pseudo10kgpt2-xln5k",
			"pseudo20kgpt2-xln5k",
			"pseudo30kgpt2-xln5k",
			"pseudo40kgpt2-xln5k",
			"round2pseudo20ktraingpt2-xl",
			"round3pseudo20ktraingpt2-xl",
			"round4pseudo20ktraingpt2-xl",
			"round5pseudo20ktraingpt2-xl"):
			params["policy_comp"] = param_str
		elif param_str[0] == "n" and param_str[1].isdigit():
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
			params["limit"] = num
		elif param_str[0] == "d" and param_str[1].isdigit():
			params["dropout_prob"] = float(param_str[1:])
		elif param_str[0] == "t" and param_str[1].isdigit():
			params["temperature"] = float(param_str[1:])
		elif param_str.startswith("steps"):
			params["steps"] = int(param_str.replace("steps", ""))
		elif param_str.startswith("cp"):
			params["checkpoint"] = "checkpoint-" + param_str.replace("cp", "")
		elif param_str in ("train", "valid", "test"):
			params["split"] = param_str
		elif param_str.startswith("seed"):
			params["seed"] = int(param_str.replace("seed", ""))
		else:
			raise ValueError("Unrecognized params: <" + param_str + ">")
	return params


def get_config(experiment):
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	component, params_str = experiment.split(":")
	
	params_str_parts = params_str.split("#")

	if component == "huggingface_finetune":
		assert len(params_str_parts) == 1
		finetune_params_str = params_str_parts[0]
		finetune_params = _parse_params_str(finetune_params_str)

		train_data_limit = finetune_params.get("limit", -1)
		train_data_file = os.path.join(config["data_dir"], _get_data_filename(finetune_params))
		output_dir = _get_model_dir(config, finetune_params_str)

		assert not os.path.exists(output_dir), output_dir
		os.makedirs(output_dir)

		if finetune_params["policy_comp"].startswith("masked"):
			task = "DecoderOnlyMaskedSeq2Seq"
			pretrained_model_name_or_path = _get_refs_checkpoint_dir(
				config, finetune_params)
		else:
			task = "DecoderOnlySeq2Seq"
			if finetune_params["model"] == "gpt-neo":
				pretrained_model_name_or_path = "EleutherAI/gpt-neo-1.3B"
			else:
				pretrained_model_name_or_path = finetune_params["model"]

		if finetune_params["mode"] == "refs":
			cfg = ml_collections.ConfigDict({
				"pretrained_model_name_or_path": pretrained_model_name_or_path,
				"model_classname": _get_model_classname(finetune_params["model"]),
				"tokenizer_classname": _get_tokenizer_classname(finetune_params["model"]),
				"task": task,
				"cache_dir": config["cache_dir"],
				"train_data_file": train_data_file,
				"output_dir": output_dir,
				"seed": finetune_params.get("seed", 0),
				"per_device_train_batch_size": 1,
				"gradient_accumulation_steps": 8,
				"gradient_checkpointing": True,
				"num_train_epochs": 1,
				"learning_rate": 5e-5,
				"evaluation_strategy": "no",
				"save_strategy": "steps" if "steps" in finetune_params else "no",
				"save_steps": finetune_params.get("steps", 500),
				"logging_strategy": "epoch",
				"report_to": "none",
				"train_data_limit": train_data_limit,
			})
			return cfg
		elif finetune_params["mode"] == "comparisons":
			cfg = ml_collections.ConfigDict({
				"pretrained_model_name_or_path": _get_refs_checkpoint_dir(
					config, finetune_params),
				"model_classname": "GPT2DoubleHeadsModel",
				"tokenizer_classname": "GPT2Tokenizer",
				"task": "MultipleChoice",
				"train_data_limit": train_data_limit,
				"cache_dir": config["cache_dir"],
				"train_data_file": train_data_file,
				"output_dir": output_dir,
				"seed": finetune_params.get("seed", 0),
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
			return cfg
		else:
			raise ValueError("Unrecognized mode: <" + params["mode"] + ">")
	elif component == "huggingface_generate":
		assert len(params_str_parts) == 2
		checkpoint_params_str = params_str_parts[0]
		generate_params_str = params_str_parts[1]
		checkpoint_params = _parse_params_str(checkpoint_params_str)
		generate_params = _parse_params_str(generate_params_str)

		assert checkpoint_params["mode"] == "refs"
		if "mode" not in generate_params:
			generate_params["mode"] = checkpoint_params["mode"]
		assert checkpoint_params["mode"] == generate_params["mode"]

		checkpoint_dir = _get_checkpoint_dir(
			config,
			checkpoint_params_str,
			generate_params.get("checkpoint", "final_checkpoint"))

		cfg = ml_collections.ConfigDict({
			"input_file": os.path.join(config["data_dir"], _get_data_filename(generate_params)),
			"model_dir": checkpoint_dir,
			"model_classname": _get_model_classname(checkpoint_params["model"]),
			"tokenizer_classname": _get_tokenizer_classname(checkpoint_params["model"]),
			"output_file": os.path.join(checkpoint_dir, _get_predictions_filename(generate_params_str)),
			"num_beams": 1,
			"num_return_sequences": 1,
			"min_length": 24,
			"max_num_tokens": 48,
			"top_k": -1,
			"cache_dir": config["cache_dir"],
			"include_input_data": True,
			"exclude_input_keys": "",
			"limit": generate_params.get("limit", -1),
			"start": 0,
			"temperature": generate_params.get("temperature", 1.0),
			"dropout_prob": generate_params.get("dropout_prob", 0.0),
			"do_sample": ("temperature" in generate_params),
			"seed": generate_params.get("seed", 0),
			"truncation": True,
		})
		return cfg
	elif component == "evaluate":
		assert len(params_str_parts) == 2
		checkpoint_params_str = params_str_parts[0]
		evaluate_params_str = params_str_parts[1]
		checkpoint_params = _parse_params_str(checkpoint_params_str)
		evaluate_params = _parse_params_str(evaluate_params_str)

		if "mode" not in evaluate_params:
			evaluate_params["mode"] = checkpoint_params["mode"]
		assert checkpoint_params["mode"] == evaluate_params["mode"]

		checkpoint_dir = _get_checkpoint_dir(
			config,
			checkpoint_params_str,
			evaluate_params.get("checkpoint", "final_checkpoint"))

		if checkpoint_params["mode"] == "refs":
			filename = _get_predictions_filename(evaluate_params_str)
			cfg = ml_collections.ConfigDict({
				"input_file": os.path.join(checkpoint_dir, filename),
				"metric_file": checkpoint_params["mode"] + "_metrics.py",
				"output_file": os.path.join(checkpoint_dir,
					filename.replace("predictions_", "metrics_").replace(".jsonl", ".txt")),
				"evaluated_examples_file": os.path.join(checkpoint_dir,
					filename.replace("predictions_", "evaluated_examples_"))
			})
			return cfg
		elif checkpoint_params["mode"] == "comparisons":
			filename = _get_data_filename(evaluate_params)
			cfg = ml_collections.ConfigDict({
				"input_file": os.path.join(config["data_dir"], filename),
				"metric_file": checkpoint_params["mode"] + "_metrics.py",
				"output_file": os.path.join(checkpoint_dir,
					filename.replace("comparisons_", "metrics_").replace(".jsonl", ".txt")),
				"evaluated_examples_file": os.path.join(checkpoint_dir,
					filename.replace("comparisons_", "evaluated_examples_")),
				"checkpoint_dir": checkpoint_dir,
				"cache_dir": config["cache_dir"],
				"limit": evaluate_params.get("limit", None)
			})
			return cfg
	else:
		raise ValueError("Unrecognized component: <" + component + ">")
