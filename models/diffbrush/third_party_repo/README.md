# DiffBrush inference sources

Gang Dai, Yifan Zhang, Yutao Qin, Qiangya Guo, Shuangping Huang and Shuicheng Yan introduced DiffBrush in *Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation* (ICCV 2025). This directory contains their inference code and the visual archetypes used by the released IAM checkpoint. [Paper](https://arxiv.org/abs/2508.03256) · [Original repository and benchmark instructions](https://github.com/dailenson/DiffBrush/tree/da9addc1140bdfec463b2d41a974a0a392f80798)

Handgen’s local changes in the diffusion sampler and U-Net cache the style and content conditioning across denoising steps. Handgen supplies the writer references and prompts through its CLI. The model guide explains how to install weights and run the integration. [Model setup](../../README.md) · [Handgen integration](../../../handgen/diffbrush) · [Sampling measurements](../../../docs/findings.md)

The upstream MIT license covers the DiffBrush code. GNU Unifont has a separate font notice. [DiffBrush license](LICENSE) · [Unifont notice](files/UNIFONT-LICENSE.txt)
