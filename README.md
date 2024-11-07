## Boosting Diffusion Models with an Adaptive Momentum Sampler (IJCAI 2024)<br><sub>Official Pytorch Implementation</sub>

## Dependencies
The codebase is based on `pytorch`. Please install `pytorch` form https://pytorch.org/ and `apex` form https://github.com/NVIDIA/apex.
The additional dependencies are listed below.
```sh
pip install yacs termcolor
```


## Pretrained models and precalculated statistics

* CIFAR10 model: [[checkpoint](https://heibox.uni-heidelberg.de/seafhttp/files/227fbbbb-2938-422b-abc2-901291919cfd/model-790000.ckpt)] from https://github.com/pesser/pytorch_diffusion

* CelebA 64x64 model: [[checkpoint](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view?usp=sharing)] from https://github.com/ermongroup/ddim

* Imagenet 64x64 model: [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)] from https://github.com/openai/improved-diffusion

* LSUN Bedroom model: [[checkpoint](https://heibox.uni-heidelberg.de/seafhttp/files/e21906d8-aa92-4a52-a6b1-67b5a31924c0/model-2388000.ckpt)] from https://github.com/pesser/pytorch_diffusion

* LSUN Church model: [[checkpoint](https://heibox.uni-heidelberg.de/seafhttp/files/7c9a8332-903a-443b-a4d7-5f9378ea487f/model-4432000.ckpt)] from https://github.com/pesser/pytorch_diffusion

## Sample images
```sh
python image_generator.py --cfg configs/[DATASET].yaml \
--b1-max [B1_MAX] --eta [ETA] --st [SAMPLE_TIMESTEPS] --b2 [B2] \
--checkpoint-path [PATH]
```
DATASET is the dataset name \
B1_MAX is the b_max in paper \
B2 is the c in paper \
ETA 0 for DDIM, 1 for DDPM-small, 2 for DDPM-large \
SAMPLE_TIMESTEPS is the sampling time steps \
PATH is the path to the pre-trained model 

## BibTeX

If this repo is useful to you, please cite our corresponding technical paper.

```bibtex
@inproceedings{ijcai2024p157,
  title     = {Boosting Diffusion Models with an Adaptive Momentum Sampler},
  author    = {Wang, Xiyu and Dinh, Anh-Dung and Liu, Daochang and Xu, Chang},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {1416--1424},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/157},
  url       = {https://doi.org/10.24963/ijcai.2024/157},
}
```

## Acknowledgement

We would like to express our gratitude for the contributions of several previous works to the development of VGen. This includes, but is not limited to [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [OpenCLIP](https://github.com/mlfoundations/open_clip), [guided-diffusion](https://github.com/openai/guided-diffusion), and [DDPM](https://github.com/hojonathanho/diffusion). We are committed to building upon these foundations in a way that respects their original contributions.

