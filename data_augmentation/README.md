# data_augmentation

Experiments related to data augmentation for language reward models for summarization.

## TL;DR datasets

First download the data:

* Download AzCopy from https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
* tar -xvzf azcopy_linux_amd64_10.16.1.tar.gz
* azcopy_linux_amd64_10.16.1/azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/*" . --recursive
* azcopy_linux_amd64_10.16.1/azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/*" . --recursive
* Move the data into the data_dir specified in config.yaml

(The documentation for the data is available at https://github.com/openai/summarize-from-feedback)

Then run:

```
python prepare_tldr_datasets.py
python prepare_cnndm_datasets.py
```

## Experiments

```
python start_synthetic_datasets.py --dataset_name=tldr

sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python ../expand.py \
		huggingface_finetune:refs_{d}_all_train_{m} \
		huggingface_generate:refs_{d}_all_train_{m}#{d}_all_test \
		evaluate:refs_{d}_all_train_{m}#{d}_all_test \
		huggingface_finetune:comparisons_{d}_all_train_{m} \
		evaluate:comparisons_{d}_all_train_{m}#{d}_all_test \
		huggingface_generate:refs_{d}_all_train_{m}#{d}_unmodifiedprompt \
		huggingface_generate:refs_{d}_all_train_{m}#{d}_unmodifiedprompt_d0.2 \
		huggingface_generate:refs_{d}_all_train_{m}#{d}_shuffledprompt \
		huggingface_finetune:refs_{d}_maskedrefprompt_train_{m} \
		huggingface_generate:refs_{d}_maskedrefprompt_train_{m}#{d}_maskedrefprompt_test \
		--d=tldr \
		--m=gpt2)

python finish_synthetic_datasets.py --dataset_name=tldr --model=gpt2

sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python ../expand.py \
		huggingface_finetune:comparisons_{d}_{p}_train_{m}_n{n} \
		evaluate:comparisons_{d}_{p}_train_{m}_n{n}#{d}_refvsup+supvsup_test \
		--d=tldr \
		--m=gpt2 \
		--p=refvsup+supvsup,gpt2vgpt2d0.2,\
			refvgpt2,refvgpt2d0.2,\
			refvmaskedrefprompt,refvshuffledprompt,\
			gpt2vgpt2d0.2+refvmaskedrefprompt \
		--n=1k,5k,10k,20k,30k,40k)

sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python ../expand.py \
		evaluate:comparisons_tldr_refvsup+supvsup_train_{m}_n{n}#cnndm_supvsup_test \
		--m=gpt2 \
		--n=1k,5k,10k,20k,30k,40k)
```