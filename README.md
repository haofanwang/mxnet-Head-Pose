# mxnet-headpose
An MXNet implementation of Fine-Grained Head Pose.

The original Pytorch version is [here](https://github.com/natanielruiz/deep-head-pose)

# Quick Start
We Do Not implement Dlib to detect face yet, input should be cropped head image with size of 48x48x1, we provide a pre-trained model here:
```bash
python code/test.py --mode 1
```

You can also train on your datasets.
```bash
python train.py
```
On a single TITAN X, it only takes about 5 minutes to finish 15 epochs on around 300k datasets. Also support for CPU mode.

# What's more
The original method follows the [paper](https://arxiv.org/abs/1710.00925).

We further exploits this problem and achieve state of the art performance. But due to secrecy agreement, the code will NOT be released until the project is finished or paper comes out. We only release the reproduced version and show some results here now.

<div align="center">
  <img src="https://github.com/haofanwang/mxnet-headpose/blob/master/example.jpg" width="380"><br><br>
</div>


```
{
  author    = {Haofan Wang}
  institute  = {Horizon Robotics inc}
  contact = {haofan.wang@horizon.ai}
}
```
