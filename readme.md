###### 复旦大学大数据学院，计算机视觉第一次作业。

### 题目描述

手工搭建三层神经网络分类器，在数据集[Fashion-MNIST](https:// github.com/zalandoresearch/fashion-mnist)上进行训练以实现图像分类。具体见：https://elearning.fudan.edu.cn/courses/74313/assignments/87031

### 测试与训练

打开demo.ipynb， 执行Training和Testing程序块即可。

#### 文件说明

layers.py 存放线性层和各种激活函数

loader.py 存放数据集loader

Loss.py 存放交叉熵损失函数

net.py 预设网络架构

optim.py 存放优化器，目前只有SGD

#### 参考内容

[1] [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

[2] [NN-by-Numpy](https://github.com/leeroee/NN-by-Numpy/tree/master)