import glob
import os
import ml_collections
import yaml


ASR_DATASETS = ["librispeech", "librispeechclean" , "librispeechother" , "tedlium"]
TRANSLATION_DATASETS = ["wmt"]


def _get_predictions_file(config, params_str):
	return os.path.join(config["results_dir"], "predictions_" + params_str + ".jsonl")


def _get_metrics_file(config, params_str):
	predictions_file = _get_predictions_file(config, params_str)
	return predictions_file.replace("predictions_", "metrics_").replace(".jsonl", ".txt")


def _get_evaluated_examples_file(config, params_str):
	predictions_file = _get_predictions_file(config, params_str)
	return predictions_file.replace("predictions_", "evaluated_examples_")


def get_config(experiment):
	with open("config.yaml", "r") as fin:
		config = yaml.load(fin, Loader=yaml.FullLoader)

	component, params_str = experiment.split(":")

	if component == "huggingface_generate":
		params = {}
		for param_str in params_str.split("_"):
			if param_str.startswith("ancestralSampling") or param_str.startswith("beamSearch"):
				end = -1
				for i, ch in enumerate(param_str):
					if ch.isdigit():
						end = i
						break

				num_return_sequences = -1
				if end == -1:
					params["decoding_strategy"] = param_str
					if params["decoding_strategy"] == "ancestralSampling":
						num_return_sequences = 20
					elif params["decoding_strategy"] == "beamSearch":
						if params["data"] in TRANSLATION_DATASETS:
							num_return_sequences = 5
						else:
							num_return_sequences = 20
					else:
						raise ValueError(
							"Unrecognized decoding_strategy: <" + params["decoding_strategy"] + ">")
				else:
					params["decoding_strategy"] = param_str[:end]
					num_return_sequences = int(param_str[end:])
				assert num_return_sequences != -1
				params["num_return_sequences"] = num_return_sequences

			elif param_str.startswith("topp"):
				params["topp"] = float(param_str.replace("topp", ""))
			elif param_str.startswith("seed"):
				params["seed"] = int(param_str.replace("seed", ""))
			elif param_str in ASR_DATASETS + TRANSLATION_DATASETS:
				params["data"] = param_str
			elif param_str in ("train", "validation", "test"):
				params["split"] = param_str
			elif param_str == "features":
				params["include_features"] = True
			else:
				raise ValueError("Unrecognized params: <" + param_str + ">")

		decoding_strategy_to_generate_args = {
			"beamSearch": {
				"num_beams": params["num_return_sequences"],
				"num_return_sequences": params["num_return_sequences"],
				"do_sample": False
			},
			"ancestralSampling": {
				"num_beams": 1,
				"num_return_sequences": params["num_return_sequences"],
				"do_sample": True
			},
		}

		data_fname = params["data"]
		if "split" in params:
			data_fname += "_" + params["split"]
		data_fname += ".jsonl"
		input_file = os.path.join(config["data_dir"], data_fname)

		if params["data"] in TRANSLATION_DATASETS:
			model_dir = "facebook/wmt19-en-de"
			tokenizer_dir = "facebook/wmt19-en-de"
			model_classname = "FSMTForConditionalGeneration"
			tokenizer_classname = "FSMTTokenizer"
			exclude_input_keys = ""
			sampling_rate = -1
		elif params["data"] in ASR_DATASETS:
			model_dir = "facebook/s2t-small-librispeech-asr"
			tokenizer_dir = "facebook/s2t-small-librispeech-asr"
			model_classname = "Speech2TextForConditionalGeneration"
			tokenizer_classname = "Speech2TextProcessor"
			exclude_input_keys = "prompt"
			sampling_rate = 16000
		else:
			raise ValueError("Unrecognized dataset")

		return ml_collections.ConfigDict({
				"input_file": input_file,
				"output_file": _get_predictions_file(config, params_str),
				"model_dir": model_dir,
				"tokenizer_dir": tokenizer_dir,
				"temperature": 1.0,
				"model_classname": model_classname,
				"tokenizer_classname": tokenizer_classname,
				"cache_dir": config["cache_dir"],
				"max_num_tokens": -1,
				"include_input_data": True,
				"exclude_input_keys": exclude_input_keys,
				"limit": -1,
				"start": 0,
				"sampling_rate": sampling_rate,
				"top_k": -1,
				"top_p": params.get("topp", 1.0),
				"seed": params.get("seed", 0),
				"include_features": params.get("include_features", False),
				**decoding_strategy_to_generate_args[params["decoding_strategy"]]})
	elif component == "merge_files_generate":
		input_files = []
		for part in params_str.split("+"):
			input_files.extend(glob.glob(_get_predictions_file(config, part)))
		output_file = _get_predictions_file(config, params_str.replace("*", "X"))
		return ml_collections.ConfigDict({
			"input_files": ",".join(input_files),
			"output_file": output_file})
	elif component == "evaluate":
		return ml_collections.ConfigDict({
			"input_file": _get_predictions_file(config, params_str),
			"metric_file": "metrics.py",
			"output_file": _get_metrics_file(config, params_str),
			"evaluated_examples_file": _get_evaluated_examples_file(config, params_str)
		})

	raise ValueError("Unrecognized component")