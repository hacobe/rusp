import jiwer
import numpy as np
import string


def _get_xy(scores, wers):
	rank_to_index = {i: x for i, x in enumerate(np.argsort(scores)[::-1])}
	rwers = [w for w in wers]
	x = [0.]
	y = [np.mean(rwers)]
	for i in range(len(wers)):
		index = rank_to_index[i]
		rwers[index] = 0.
		m = np.mean(rwers)
		x.append((i+1)/float(len(rwers)))
		y.append(np.mean(rwers))
	x = np.array(x)
	y = np.array(y)
	return x, y


def _get_y_random(x, wers):
	x0 = 0.
	y0 = np.mean(wers)
	x1 = 1.0
	y1 = 0.
	m = (y1 - y0)/(x1 - x0)
	b = y0 - m*x0
	y_random = m * x + b
	return y_random


def _calc_prr(scores, wers):
	x, y_oracle = _get_xy(scores=wers, wers=wers)
	_, y = _get_xy(scores=scores, wers=wers)
	y_random = _get_y_random(x, wers)
	auc_random = np.trapz(y_random, x)
	auc = np.trapz(y, x)
	auc_oracle = np.trapz(y_oracle, x)
	ar_oracle = auc_random - auc_oracle
	ar = auc_random - auc
	prr = ar / ar_oracle
	return prr


def _calc_wer(reference, prediction):
	r = reference.strip().lower()
	p = prediction.strip().lower()

	if r == p:
		return 0

	if r == "":
		return 1

	if p == "":
		return 1

	return jiwer.wer(r, p)


def _calc_importance_weighted_entropy(predictions):
	# Malinin and Gales 2021, Equation 36
	wts = []
	for prediction in predictions:
		prob = np.exp(sum(prediction["token_logprobs"]))
		wts.append(prob)
	wts = np.array(wts)
	wts /= sum(wts)
	
	# Malinin and Gales 2021, Equation 41
	terms = []
	for j, prediction in enumerate(predictions):
		logprob = sum(prediction["token_logprobs"])
		pi = wts[j]
		L = len(prediction["token_logprobs"])
		terms.append((pi / L) * logprob)

	entropy = -1. * sum(terms)

	return entropy


def compute(examples):
	for i in range(len(examples)):
		beam_search_span = (-1, -1)
		for key in examples[i]["api_response"]:
			if key.find("beamSearch") != -1:
				assert beam_search_span == (-1, -1)
				beam_search_span = examples[i]["api_response"][key]
				assert len(beam_search_span) == 2
				assert isinstance(beam_search_span[0], int)
				assert isinstance(beam_search_span[1], int)
		assert beam_search_span != (-1, -1)		
		start, end = beam_search_span
		hypotheses = examples[i]["predictions"][start:end]
		samples = examples[i]["predictions"][end:]
		reference = examples[i]["reference"].lower()

		# sort from smallest logprob to largest
		hypotheses.sort(key=lambda x: np.mean(x["token_logprobs"]))
		# sort from largest logprob to smallest
		hypotheses = hypotheses[::-1]
		index_best = 0
		prediction_best = hypotheses[index_best]

		example_metrics = {}

		# True WER
		example_metrics["wer"] = _calc_wer(reference, prediction_best["text"])

		example_metrics["scores"] = {}
		example_metrics["scores"]["neg_mean_token_logprobs"] = -1 * np.mean(prediction_best["token_logprobs"])
		example_metrics["scores"]["neg_sum_token_logprobs"] = -1 * sum(prediction_best["token_logprobs"])

		# Average word error rate between hypothesis and samples
		samp_wers = []
		for j in range(len(samples)):
			samp_wers.append(_calc_wer(samples[j]["text"], prediction_best["text"])) 
		example_metrics["scores"]["mean_samp_wers"] = np.mean(samp_wers)

		# Entropy from importance weighting hypotheses
		example_metrics["scores"]["hypo_entropy"] = _calc_importance_weighted_entropy(hypotheses)

		# Entropy from samples
		example_metrics["scores"]["samp_entropy"] = _calc_importance_weighted_entropy(samples)

		examples[i]["metrics"] = example_metrics

	wers = np.array([x["metrics"]["wer"] for x in examples])
	metrics = {
		"mean_wer": np.mean(wers),
	}
	methods = sorted(list(examples[0]["metrics"]["scores"].keys()))
	for method in methods:
		key = "prr_" + method
		scores = np.array([x["metrics"]["scores"][method] for x in examples])
		metrics[key] = _calc_prr(scores=scores, wers=wers)

	return metrics