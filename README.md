# augqe

This is a research directory.

## Dependencies

To install dependencies, follow the instructions at https://github.com/hacobe/language_models

## Configuration

* Set the paths in config.yaml
* Set the environment variables in run.sh

## TL;DR datasets

* Download AzCopy from https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
* tar -xvzf azcopy_linux_amd64_10.16.1.tar.gz
* azcopy_linux_amd64_10.16.1/azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/*" . --recursive
* azcopy_linux_amd64_10.16.1/azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/*" . --recursive
* Move the data into the data_dir specified in config.yaml
* Run: python prepare_refs_dataset.py
* Run: python prepare_comparisons_dataset.py

The documentation for the datasets is available at https://github.com/openai/summarize-from-feedback

## Summarization models

```
sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python expand.py \
		huggingface_finetune:refs_{m} \
		huggingface_generate:refs_{m} \
		evaluate:refs_{m} \
		--m=gpt2,gpt2-medium,gpt2-large,gpt2-xl)
```

## Reward models

```
sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python expand.py \
		huggingface_finetune:comparisons_{m} \
		evaluate:comparisons_{m} \
		--m=gpt2,gpt2-medium,gpt2-large,gpt2-xl)
```
