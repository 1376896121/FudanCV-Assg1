import numpy as np
from layers import Softmax

class CrossEntropyLoss(object):
    def __init__(self):
        # 内置一个softmax作为分类器
        self.classifier = Softmax()

    def gradient(self):
        return self.grad

    def __call__(self, a, y, requires_acc=True):
        '''
        a: 批量的样本输出
        y: 批量的样本真值
        requires_acc: 是否输出正确率
        '''
        # 网络的输出不应该经过softmax分类，而在交叉熵损失函数中进行
        a = self.classifier.forward(a)
        # 提前计算好梯度
        self.grad = a - y
        # 样本整体损失
        # L_{i}=-\sum_{j}^{C} y_{ij} \ln a_{ij}
        # 样本的平均损失
        # L_{mean}=\frac{1}{N} \sum_{i}^{N} L_{i}=-\frac{1}{N} \sum_{i}^{N} \sum_{j}^{C} y_{ij} \ln a_{ij}
        loss = -1 * np.einsum('ij,ij->', y, np.log(a), optimize=True) / y.shape[0]
        if requires_acc:
            acc = np.argmax(a, axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        return loss