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
python prepare_refs_dataset.py
python prepare_comparisons_dataset.py
```

## Experiments

```
python prepare_derived_datasets.py

sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python ../expand.py \
		huggingface_finetune:refs_base_train_{m} \
		huggingface_generate:refs_base_train_{m}#base_test \
		evaluate:refs_base_train_{m}#base_test \
		huggingface_finetune:comparisons_{d}_train_{m}_n{n} \
		evaluate:comparisons_{d}_train_{m}_n{n}#sup2vsup2_test \
		huggingface_finetune:refs_maskedref_train_{m} \
		huggingface_generate:refs_maskedref_train_{m}#maskedref_test \
		huggingface_generate:refs_base_train_gpt2#excludesup2vsup2testprompts \
		--m=gpt2 \
		--d=base,refvsup2,sup2vsup2+refvsup2,refvsup2policy,refvsup1,refvsup1policy \
		--n=8k)

python prepare_generated_datasets.py

sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 run.sh \
	$(python ../expand.py \
		huggingface_finetune:comparisons_{d}_train_{m}_n{n} \
		evaluate:comparisons_{d}_train_{m}_n{n}#sup2vsup2_test \
		--m=gpt2 \
		--d=sup2vsup2,refvdup,refvdrop,refvmaskedref,gpt2vgpt2d0.3 \
		--n=1k,2k,3k,4k,5k,6k,7k,8k)
```