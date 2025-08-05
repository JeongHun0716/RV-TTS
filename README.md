# RV-TTS

This repository contains the Re-implementation of the following paper:
> **Revival with Voice: Multi-modal Controllable Text-to-Speech Synthesis**<br>
> Minsu Kim, Pingchuan Ma, Honglie Chen, Stavros Petridis, Maja Pantic <br>
> \[[Paper](https://arxiv.org/abs/2505.18972)\]

<div align="center"><img width="75%" src="Image.png?raw=true" /></div>

## Requirements
- python 3.9
- pytorch 2.4.1
- torchvision
- torchaudio
- transformers
- av
- tensorboard
- librosa
- einops
- easydict
- mxnet
- insightface
- onnxruntime-gpu
- onnx

After cloning this repository, please run belows to add face encoder part.
```shell
pip install git+https://github.com/huggingface/parler-tts.git@5d0aca9753ab74ded179732f5bd797f7a8c6f8ee
git clone --filter=blob:none --sparse https://github.com/deepinsight/insightface.git
cd insightface
git sparse-checkout add arcface_torch
cd ..
```

## Preparing data
- Put face images ('.jpg') in `./imgs` dir. 
- Put input text ('text.txt') in `./text` dir. Each line will be read by the model.
- Put natural descriptive text ('description.txt') in `./description` dir. Each line corresponds to each line of input text.

Therefore, if 2 images are in `./imgs` and 10 lines in both `text.txt` and `description.txt`, then 10 samples will be generated for each image.

## Pre-trained model checkpoints
Put the pre-trained models in `./pretrained` dir <br>

**Face encoder**: [link](https://drive.google.com/file/d/1q18F_jQ0W5MsqDex62IXG_XE7HuT3zRI/view?usp=sharing) <br>
**RV-TTS**: [link](https://drive.google.com/file/d/1c7tj9CmC5aGqt9Z5q8KyDLXl_sM9VzV6/view?usp=sharing)

## Testing the Model
To test the model, run following command:
```shell
# Constant voice generation for each image
python test.py \
--checkpoint ./pretrained/RV-TTS.ckpt \
--fe_model_ckpt ./pretrained/Face_Encoder.ckpt \
--batch_size 8 \
--max_sp_len 24 \
--top_k 30 \
--repetition_penalty 1.2 \
--temperature 0.9 \
--seed 123 \
--constant_gen
```

```shell
# Constant voice generation for each image (closer sound; by indicating the recording is not done in public space)
python test.py \
--checkpoint ./pretrained/RV-TTS.ckpt \
--fe_model_ckpt ./pretrained/Face_Encoder.ckpt \
--batch_size 8 \
--max_sp_len 24 \
--top_k 30 \
--repetition_penalty 1.2 \
--temperature 0.9 \
--seed 123 \
--constant_gen \
--no_public
```

```shell
# Multiple voice generation for each sample
python test.py \
--checkpoint ./pretrained/RV-TTS.ckpt \
--fe_model_ckpt ./pretrained/Face_Encoder.ckpt \
--batch_size 8 \
--max_sp_len 24 \
--top_k 30 \
--repetition_penalty 1.2 \
--temperature 0.9 \
--seed 123 \
```


## Citation
If you find this work useful in your research, please cite the paper:
```bibtex
@inproceedings{kim2025revival,
  title={Revival with Voice: Multi-modal Controllable Text-to-Speech Synthesis},
  author={Kim, Minsu and Ma, Pingchuan and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={Proc. Interspeech},
  year={2025}
}
```
