# selective_asr

```
sbatch --partition=jsteinhardt -w balrog --gres=gpu:1 selective_asr.sh \
    $(python expand.py \
        huggingface_generate:{d}_beamSearch20 \
        huggingface_generate:{d}_ancestralSampling20_topp{p}_seed0 \
        huggingface_generate:{d}_ancestralSampling20_topp{p}_seed1 \
        huggingface_generate:{d}_ancestralSampling20_topp{p}_seed2 \
        huggingface_generate:{d}_ancestralSampling20_topp{p}_seed3 \
        huggingface_generate:{d}_ancestralSampling20_topp{p}_seed4 \
        merge_files_generate:{d}_beamSearch20+{d}_ancestralSampling20_topp{p}_seed* \
        evaluate:{d}_beamSearch20+{d}_ancestralSampling20_topp{p}_seedX \
        --d=tedlium,librispeechclean,librispeechother \
        --p=1.0,0.9)
```