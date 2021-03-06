# README

In this project, two denoising experiments mentioned in the paper are included. Here we provide the test code for two datasets. The following is a detailed description

## Platform

1. python==3.6.9
2. tensorflow-gpu==1.8.1
3. keras==2.2.4

Currently we can only guarantee that this demo can run normally under Linux.

## Project Structure

- `./data` Include the images of two datasets, namely the BSD-100 (100 images) and Kodak-24 (24 images)
- `./model` Include the model parameter file (.h5) which already trained
- `./src` Include scripts including data processing methods and other tool functions

## Instructions

1. Pre-processing: Before running the program, please run the following command to pre-process the project

  ```bash
  python -W ignore main.py --preprocess 
  python -W ignore main.py --preprocess --dataset=Kodak 
  ```

2. Showing test image

  We provide tests for two datasets, BSD-100 and Kodak-24, which include four different noise levels of `[10, 20, 30, 40]`. For example, if you want to test the denoising result of No.3 image with `sigma=20` in BSD-100, you can run the following command

  ```bash
  python -W ignore main.py --dataset=BSD --test_model --sigma=20 --test_pic_num=2 --pic_show
  ```

3. Saving test image

  E.g.

  ```bash
  python -W ignore main.py --dataset=BSD --test_model --sigma=20 --test_pic_num=2 --pic_save
  ```

## Example

Here is an example in `./example/noise_20_2.png`, which the first sub-figure count from left to right is the original image, the second sub-figure is the noisy image, and the last sub-figure is the denoising result generated by RQV-CNN model. 
