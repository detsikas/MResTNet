# MResTNet
The MResTNet architecture is a deep learning network that achieves state of the art performance for the semantic segmentation task in the real-time domain. The architecture is the following

![](/images/MResTNet.png?raw=true)

The architecture is described in detail in the paper ```"MResTNet: A Multi-resolution Transformer framework with CNN extensions for Semantic Segmentation"``` by ```Nikolaos Detsikas , Nikolaos Mitianoudis and Ioannis Pratikakis (Electrical and Computer Engineering Department, Democritus University of Thrace, University Campus Xanthi-Kimmeria, Xanthi 67100, Greece)```.

## Code structure
The code consists of 4 directories. 

| Directory | Description |
| --------- | ----------- |
| mrestnet/   | The model architectural blocks |
| segm/ | Training and evaluation scripts as well supporting code for the training and evaluation pipeline |

## Datasets
The architecture is trained and evaluated in the Cityscapes and the ADE20K datasets.

## Training
The model can be trained with various arguments and configuration combinations. The followg is a typical command for training the model with the Cityscapes dataset

```python
python -m segm.train --log-dir output_directory --dataset cityscapes --backbone vit_tiny_patch16_384 --decoder mask_transformer
```

## MIoU evaluation
The model can be evaluated with respect to the MIoU metric with the following command

```python
python -m segm.eval.miou output_directory/checkpoint.pth cityscapes --save-images --no-blend
```

## Copyright notice
The training and evaluation pipelines (not the model) are largely based on the following work
https://github.com/rstrudel/segmenter?tab=MIT-1-ov-file


