import jiwer
import nltk
import numpy as np
import sacrebleu
import string
import tqdm


def _get_xy(scores, costs):
	rank_to_index = {i: x for i, x in enumerate(np.argsort(scores)[::-1])}
	rcosts = [c for c in costs]
	x = [0.]
	y = [np.mean(rcosts)]
	for i in range(len(costs)):
		index = rank_to_index[i]
		rcosts[index] = 0.
		m = np.mean(rcosts)
		x.append((i+1)/float(len(rcosts)))
		y.append(np.mean(rcosts))
	x = np.array(x)
	y = np.array(y)
	return x, y


def _get_y_random(x, costs):
	x0 = 0.
	y0 = np.mean(costs)
	x1 = 1.0
	y1 = 0.
	m = (y1 - y0)/(x1 - x0)
	b = y0 - m*x0
	y_random = m * x + b
	return y_random


def _calc_prr(scores, costs):
	x, y_oracle = _get_xy(scores=costs, costs=costs)
	_, y = _get_xy(scores=scores, costs=costs)
	y_random = _get_y_random(x, costs)
	auc_random = np.trapz(y_random, x)
	auc = np.trapz(y, x)
	auc_oracle = np.trapz(y_oracle, x)
	ar_oracle = auc_random - auc_oracle
	ar = auc_random - auc
	prr = ar / ar_oracle
	return prr


def _fmt(text):
	ftext = text.lower().strip()
	if not ftext:
		return " "
	return ftext


def _calc_wer(reference, prediction):
	r = _fmt(reference)
	p = _fmt(prediction)

	if r == p:
		return 0

	if r.strip() == "":
		return 1

	if p.strip() == "":
		return 1

	return jiwer.wer(r, p)


def _calc_corpus_bleu(references, predictions):
	assert len(predictions) == len(references)
	return sacrebleu.BLEU().corpus_score(
		[_fmt(x) for x in predictions],
		[[_fmt(x) for x in references]]).score / 100.0


def _calc_neg_sent_bleu(reference, prediction):
	r = _fmt(reference)
	p = _fmt(prediction)
	return -1 * _calc_corpus_bleu(references=[r], predictions=[p])


def _calc_neg_meteor(reference, prediction):
	r = _fmt(reference)
	p = _fmt(prediction)
	return -1 * nltk.translate.meteor_score.single_meteor_score(r.split(" "), p.split(" "))


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
	for i in tqdm.tqdm(range(len(examples))):
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

		# True cost
		example_metrics["costs"] = {}
		example_metrics["costs"]["wer"] = _calc_wer(reference, prediction_best["text"])
		example_metrics["costs"]["neg_sent_bleu"] = _calc_neg_sent_bleu(reference, prediction_best["text"])
		example_metrics["costs"]["neg_meteor"] = _calc_neg_meteor(reference, prediction_best["text"])

		example_metrics["scores"] = {}
		example_metrics["scores"]["neg_mean_token_logprobs"] = -1 * np.mean(prediction_best["token_logprobs"])
		example_metrics["scores"]["neg_sum_token_logprobs"] = -1 * sum(prediction_best["token_logprobs"])

		# Average word error rate between hypothesis and samples
		samp_wers = []
		for j in range(len(samples)):
			samp_wers.append(_calc_wer(samples[j]["text"], prediction_best["text"])) 
		example_metrics["scores"]["mean_samp_wers"] = np.mean(samp_wers)

		# Average negative sentence BLEU between hypothesis and samples
		samp_neg_bleus = []
		for j in range(len(samples)):
			samp_neg_bleus.append(_calc_neg_sent_bleu(samples[j]["text"], prediction_best["text"])) 
		example_metrics["scores"]["mean_samp_neg_bleus"] = np.mean(samp_neg_bleus)

		# Average negative METEOR between hypothesis and samples
		samp_neg_meteors = []
		for j in range(len(samples)):
			samp_neg_meteors.append(_calc_neg_meteor(samples[j]["text"], prediction_best["text"])) 
		example_metrics["scores"]["mean_samp_neg_meteors"] = np.mean(samp_neg_meteors)

		# Entropy from importance weighting hypotheses
		example_metrics["scores"]["hypo_entropy"] = _calc_importance_weighted_entropy(hypotheses)

		# Entropy from samples
		example_metrics["scores"]["samp_entropy"] = _calc_importance_weighted_entropy(samples)

		examples[i]["metrics"] = example_metrics
		examples[i]["prediction_best"] = prediction_best

	wers = np.array([x["metrics"]["costs"]["wer"] for x in examples])
	meteors = np.array([-1 * x["metrics"]["costs"]["neg_meteor"] for x in examples])
	metrics = {
		"mean_wer": np.mean(wers),
		"corpus_bleu": _calc_corpus_bleu(
			references=[x["reference"] for x in examples],
			predictions=[x["prediction_best"]["text"] for x in examples]),
		"mean_meteor": np.mean(meteors)
	}
	cost_keys = sorted(list(examples[0]["metrics"]["costs"].keys()))
	score_keys = sorted(list(examples[0]["metrics"]["scores"].keys()))
	for cost_key in tqdm.tqdm(cost_keys):
		costs = np.array([x["metrics"]["costs"][cost_key] for x in examples])
		for score_key in score_keys:
			scores = np.array([x["metrics"]["scores"][score_key] for x in examples])
			key = "prr_" + cost_key + "_" + score_key
			metrics[key] = _calc_prr(scores=scores, costs=costs)
	return metrics