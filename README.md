# [NEUCOM 2024] Advancing Zero-Shot Semantic Segmentation through Attribute Correlations
[Runtong Zhang]<sup>1</sup>, [Fanman Meng]<sup>1</sup>, [Shuai Chen]<sup>1</sup>, [Qingbo Wu]<sup>1</sup>, [Linfeng Xu]<sup>1</sup>, [Hongliang Li]<sup>1</sup> <br />
<sup>1</sup> University of Electronic Science and Techonology of China

## Abstract
We advance the zero-shot semantic segmentation through attribute correlations. Specifically, we propose a hierarchical semantic segmentation framework incorporating an attribute prompt tuning method. Correspondingly, we construct a Visual Hierarchical Semantic Classes (VHSC) benchmark, meticulously annotating shared-attributes at the pixel level to conduct the experiments. Extensive experiments on the VHSC benchmark showcase the superior performance of our method compared to existing zero-shot semantic segmentation methods, achieving mIoU of 73.0\% and FBIoU of 87.5\%. <br />
![Overview](figures/overview.png)

## Requirements

* Python == 3.7.16
* Pytorch == 1.10.1
* Cuda == 11.3
* Torchvision == 0.11.2
* GPU == NVIDIA Titan XP


## Dataset
Visual Hierarchical Semantic Classes (VHSC) benchmark. <br />
![Dataset](figures/dataset.png) <br />
Download Link: [OneDrive](https://1drv.ms/u/s!AlKD6m_5g-8SbUBIUBurgoRu9eI?e=qYCZ9Z) <br />
The validation data is from the MS COCO dataset. Please download it from [here](https://cocodataset.org/#download) and put the "JPEGImages" folder in path: VHSC/novel/JPEGImages <br />

## Checkpoints
The pretrained ViT model: [OneDrive](https://1drv.ms/u/s!AlKD6m_5g-8SdEaPmXqo82yAqjU?e=hG5CXD) <br />
(Please put it in the "pretrained" folder) <br />
Our trained model: [OneDrive](https://1drv.ms/u/s!AlKD6m_5g-8SbpkEre1stRxWyvM?e=MqBkYQ) <br />
(Please put it in the "checkpoints" folder) <br />

## Training
```
python train.py configs/vhsc/vpt_seg_attr20_bg_AttrPrompt_vit-b_512x512.py
```
## Evaluation
```
python test.py configs/vhsc/vpt_seg_attr20_bg_AttrPrompt_vit-b_512x512.py checkpoints/AttrPromptCheckpoint.pth
```

## Citation
If you find our code or data helpful, please cite our paper:
```bibtex
@article{ZHANG2024127829,
  title = {Advancing zero-shot semantic segmentation through attribute correlations},
  journal = {Neurocomputing},
  volume = {594},
  pages = {127829},
  year = {2024},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2024.127829},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231224006003},
  author = {Runtong Zhang and Fanman Meng and Shuai Chen and Qingbo Wu and Linfeng Xu and Hongliang Li},
  keywords = {Zero-shot learning, Image segmentation, Attribute learning}
}
```

## Acknowledgement
Our implementation is mainly based on following repositories. Thanks for their authors.
* [CLIP](https://github.com/openai/CLIP)
* [ZegCLIP](https://github.com/ZiqinZhou66/ZegCLIP)
* [Visual Prompt Tuning](https://github.com/KMnP/vpt)

