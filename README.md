# [NEUCOM 2024] Advancing Zero-Shot Semantic Segmentation through Attribute Correlations

## Abstract
we advance the zero-shot semantic segmentation through attribute correlations. Specifically, we introduce a set of shared-attribute labels, of which the design fully considers the structural relations between attributes and classes, to provide rational and sufficient attribute-class correlations. Besides, due to the minor intra-class variations of shared attributes, the text features are more easily mapped to image features, thereby alleviating the domain gap issue. Furthermore, we propose a hierarchical semantic segmentation framework incorporating an attribute prompt tuning method. This approach is designed to enhance the model's adaptation to the attribute segmentation task and effectively leverage attribute features to produce better semantic segmentation results. Correspondingly, we construct a Visual Hierarchical Semantic Classes (VHSC) benchmark, meticulously annotating shared-attributes at the pixel level to conduct the experiments. Extensive experiments on the VHSC benchmark showcase the superior performance of our method compared to existing zero-shot semantic segmentation methods, achieving mIoU of 73.0\% and FBIoU of 87.5\%.

## Requirements

* Python == 3.7.3
* Pytorch == 1.8.1
* Cuda == 10.1
* Torchvision == 0.4.2
* Tensorflow == 1.14.0
* GPU == TITAN XP

## DataSets


## Training


## Evaluation


## Acknowledgement


