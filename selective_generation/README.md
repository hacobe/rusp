# selective_generation

## Datasets

```
python prepare_librispeech_dataset.py
python prepare_tedlium_dataset.py

bash download_wmt.sh # run this from `data_dir` directory set in config.yaml
python prepare_wmt_dataset.py
```

## Experiments

```
sbatch --partition=jsteinhardt -w sunstone --gres=gpu:1 run.sh \
    $(python ../expand.py \
        huggingface_generate:{d}_beamSearch20 \
        huggingface_generate:{d}_ancestralSampling_topp{p}_seed0 \
        huggingface_generate:{d}_ancestralSampling_topp{p}_seed1 \
        huggingface_generate:{d}_ancestralSampling_topp{p}_seed2 \
        huggingface_generate:{d}_ancestralSampling_topp{p}_seed3 \
        huggingface_generate:{d}_ancestralSampling_topp{p}_seed4 \
        merge_files_generate:{d}_beamSearch+{d}_ancestralSampling_topp{p}_seed* \
        evaluate:{d}_beamSearch+{d}_ancestralSampling_topp{p}_seedX \
        --d=tedlium,librispeechclean_test,librispeechother_test,wmt_test \
        --p=1.0,0.9)
```