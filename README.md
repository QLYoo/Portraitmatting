# Portraitmatting

Portrait matting model for academic use only.

## Introduction

Portraitmatting is a deep learning-based model designed for portrait matting tasks. It is intended for academic use only, and usage for commercial purposes is restricted due to potential copyright concerns with the training data. 

## Usage

You may check eval_image_folder.py or eval_webcam.py for an example of how to use the model.

## Requirements

- OpenCV
- PyTorch
- NumPy
- Pymatting (for more accurate foreground extraction)

## Hardware Requirements

A CUDA-ready GPU is required for efficient inference. The model has been tested on an NVIDIA GeForce RTX 2080 Ti.

## Model Download

A demo model is available for download at the following links:

- [Portraitmatting-SwinT](https://mega.nz/file/KIwHkAYB#U0TyXVIBCE9qBoUOcFGYsQs5JGkQdRe2ZnVXGVx0Dxw)
- [Portraitmatting-SwinT(EMA)](https://mega.nz/file/XdZnSQ5S#0M7-YHrShoGY8iBP0kHvUkia5KUYhvkob12L78KdHLI)

## Network Architecture

The model adopts an encoder-decoder structure based on Swin-Tiny. It incorporates the Pyramid Pooling Module (PPM) in the middle to extract multi-scale features, although the effectiveness of this module has not been thoroughly evaluated.

## Training Details

The model was trained on a mixed dataset, and as a result, it should perform well on a variety of portrait images. However, the specific details of the model's effectiveness on different types of data are yet to be measured.

## License

This model is provided without a license and is solely intended for academic use. No commercial usage is allowed due to potential copyright restrictions on the training data.

## Citation

If you use this model in your research, please cite this project to acknowledge its contribution.

```plaintext
@misc{liu2023mat,
  author       = {Qinglin Liu},
  title        = {Portraitmatting: A Portrait Matting Model},
  year         = {2023},
  howpublished = {\url{https://github.com/qlyoo/Portraitmatting}}
}
```

## 介绍

Portraitmatting 是一个基于深度学习的模型，专为人像抠图任务设计。由于训练数据可能涉及版权问题，仅限于学术用途，禁止用于商业用途。同时由于标注问题，模型对一些装饰物可能预测不准，但应该是公开的最靠谱的模型了。

## 环境要求

- OpenCV
- PyTorch
- NumPy
- Pymatting (用于前景提取)

## 模型下载

请访问以下链接下载

- [Portraitmatting-SwinT](https://mega.nz/file/KIwHkAYB#U0TyXVIBCE9qBoUOcFGYsQs5JGkQdRe2ZnVXGVx0Dxw)
- [Portraitmatting-SwinT(EMA)](https://mega.nz/file/XdZnSQ5S#0M7-YHrShoGY8iBP0kHvUkia5KUYhvkob12L78KdHLI)

## 硬件要求

为了高效推断模型，应使用Nvidia GPU设备。

## 网络架构

该模型采用基于Swin-Tiny的编码器-解码器结构。中间使用了金字塔池化模块（PPM）来提取多尺度特征，但该模块的有效性尚未评估。

## 许可证

该模型不提供许可证，仅供学术使用。由于训练数据可能存在版权限制，禁止商业使用。

## 补充信息

需要注意，我们在eval_image_folder.py提取素材时使用了Pymatting库提取前景。该过程也可以使用Closed form matting提取，不过速度会慢一些。

## 引用

如果您在研究中使用了该模型，请引用此项目以表示对其贡献的认可。

```plaintext
@misc{liu2023mat,
  author       = {Qinglin Liu},
  title        = {Portraitmatting: A Portrait Matting Model},
  year         = {2023},
  howpublished = {\url{https://github.com/qlyoo/Portraitmatting}}
}
```