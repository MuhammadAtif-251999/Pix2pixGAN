Official implementation of the paper: **"Low-Light Image Enhancement with Multi-Stage Interconnected Autoencoders Integration in Pix-to-Pix GAN"**.

## 📖 Abstract
The enhancement of low-light images is a significant area of study aimed at enhancing the quality of captured images in challenging lighting environments. Recently, methods based on Convolutional Neural Networks (CNN) have gained prominence as they offer state-of-the-art performance. However, many approaches based on CNN rely on increasing the size and complexity of the neural network. In this study, we propose an alternative method for improving low-light images using an Autoencoders-based multiscale knowledge transfer model. Our method leverages the power of three autoencoders, where the encoders of the first two autoencoders are directly connected to the decoder of the third autoencoder. Additionally, the decoder of the first two autoencoders is connected to the encoder of the third autoencoder. This architecture enables effective knowledge transfer, allowing the third autoencoder to learn and benefit from the enhanced knowledge extracted by the first two autoencoders. We further integrate the proposed model into the Pix-to-Pix GAN framework. By integrating our proposed model as the generator in the GAN framework, we aim to produce enhanced images that not only exhibit improved visual quality but also possess a more authentic and realistic appearance. These experimental results, both qualitative and quantitative, show that our method is better than the state-of-the-art methodologies. 

## 🏗️ Architecture
The generator consists of three specialized autoencoders where the knowledge is shared across stages:
* **Encoders 1 & 2** extract multi-scale features.
* **Encoder 3** integrates these features using skip-connections from previous decoders.
* **Discriminator:** A PatchGAN-based structure using `Adadelta` optimization for stable binary cross-entropy loss.


## ✨ Key Features
* **Multi-Stage Interconnected Autoencoders:** Enables effective knowledge transfer between different layers of the network.
* **Custom Loss Function:** A weighted combination of:
    * **MAC Loss:** Multi-Scale Adaptive Contrast Loss for local/global normalization.
    * **MAE Loss:** Mean Absolute Error for pixel-level accuracy.
    * **SSIM Loss:** Structural Similarity Index for perceptual quality preservation.
* **Dilation and Concatenation:** Uses dilated convolutions to increase the receptive field without losing resolution.

## 📊 Datasets
The following datasets were used for training and evaluating **Self-RefineGAN**:

* **[LOLv1 Dataset](https://huggingface.co/datasets/geekyrakshit/LoL-Dataset)**: 500 paired low-light/normal-light images.
* **[SICE Dataset](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view)**: Multi-exposure image sequences.
* **Input:** Low-exposure images.
* **Target:** High-quality ground truth labels.
* **Preprocessing:** Images are resized to **600x400** and normalized to [0, 1].0

## 1. Requirements
* TensorFlow 2.x
* Keras
* NumPy
* Pillow
* Matplotlib

## Primary Citation
Standard Format: M. Atif and C. Yan, "Low Light Image Enhancement with Multi-Stage Interconnected Autoencoders Integration in Pix-to-Pix GAN," International Journal of Computer and Information Engineering, vol. 18, no. 11, pp. 645-653, 2024.

@article{atif2024lowlight,
title={Low Light Image Enhancement with Multi-Stage Interconnected Autoencoders Integration in Pix-to-Pix GAN},
author={Atif, Muhammad and Yan, Cang},
journal={International Journal of Computer and Information Engineering},
volume={18},
number={11},
pages={645--653},
year={2024},
publisher={World Academy of Science, Engineering and Technology}
}

