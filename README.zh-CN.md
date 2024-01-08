[English](README.md) | 简体中文

# 机器学习之力
![PyPI](https://img.shields.io/pypi/v/mlforce)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XavierSpycy/MLForce/blob/main/basics_test.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 简介
我们的库，名为`MLForce`，代表`Machine Learning Force`，是一个旨在帮助机器学习初学者理解各种算法而从零实现多种机器学习算法的Python工具包。

`MLForce`中的每个模块都为不同目的而服务。从基础机器学习算法到神经网络、非负矩阵分解，您将从监督学习、深度学习、无监督学习等等各个角度，深入、全面地探索和理解各种机器学习算法。

通过结合易用性和功效性，这个库旨在夯实您在机器学习上的理论理解和实践技巧，为您打开机器学习的大门。所以，拥抱`MLForce`，踏上回报满满的机器学习艺术之旅！

## 安装
您可以使用`pip`安装`MLForce`:

```
$ pip install mlforce
```

## 模块
### `basics` 模块:

在这个模块中，正如这个模块名一样，我们实现了多种非常基础的机器学习算法，包括K最近邻、线性回归、决策树特征选择、隐马尔可夫、KMeans、层次聚类等、DBSCAN等。

部分算法仅在特定场景下针对特定数据输入生效。因此，针对该模块，我们构建了内置的`StandardDataset`（`标准数据集`），以规范输入数据和标签，并验证我们的算法。

由于该模块无独立仓库，因此具体的使用方法目前需参考文档字符串。

### `mf` 模块:

`mf`代表`matrix factorization`（`矩阵分解`）。矩阵分解是一类降维操作。该模块暂时仅包含**用NumPy实现的`Non-negative Matrix Factorization`（`非负矩阵分解`）**。

非负矩阵分解通过两个低秩的矩阵 ($m \times k$ 和 $k \times n$, $k << m, n$) 来表示原矩阵 ($m \times n$) 以达到降维的目的，同时提供更稀疏的特征表达。非负矩阵分解可以用作图像修复、主题建模等。在原任务中，我们通过非负矩阵分解的方式，对一系列受噪声污染的人脸进行重建，效果显著。

具体来说，我们实现了8种高效的非负矩阵分解，包括基于：$L_2$ 范数, KL散度, IS散度, $L_{2, 1}$ 范数, 超表面损失, $L_1$ 范数正则, Capped范数, 柯西等的非负矩阵分解。在数值稳定条件下，均可通过测试且有效。

此外，该模块的最大特点在于为学习者/研究者提供了自定义的非负矩阵分解开发框架，并且可以原地测试和对比效果，极大程度地简化开发流程。

详细的效果和使用方法可以参照独立仓库：[非负矩阵分解](https://github.com/XavierSpycy/NumPyNMF)。

考虑到为今后整合更多的模块提供便利，在使用该模块时，目前仍然需要：
```python
from mlforce.mf.nmf import BasicNMF
```
使得导入该类后，可以完全按照独立仓库的方式运行代码，例如开发新的算法、测试新算法的性能等。

```python
class ExampleNMF(BasicNMF):
    # 为了自定义一个不同的NMF算法，继承BasicNMF，并重新定义 matrix_init 和 update 方法。
    def matrix_init(self, X, n_components, random_state=None):
        # 在此实现您的初始化逻辑。
        # 尽管我们提供了内置的方法，但是动手实现一种定制的初始化可以显著提升性能。
        # D, R = <your_initialization_logic>
        # D, R = np.array(D), np.array(R)
        return D, R  # 确保D, R被返回

    def update(self, X, kwargs):
        # 在此实现迭代的更新逻辑。
        # 在您算法逻辑需要的地方修改self.D, self.R
        # flag = <convergence_criterion>
        return flag  # 如果收敛，则返回True，否则返回False。
```

### `nn` 模块:

`nn`代表`neural networks`（`神经网络`）。这个模块主要是**基于NumPy从零实现Keras风格的多层感知机**。我们实现了：

  - `激活函数`：PyTorch中几乎所有的激活函数。
  - `隐藏层`：全连接层、批量归一化层、丢弃层、激活层；其中，全连接层实现了包括Xavier均匀/正态，Kaiming均匀/正态等多种初始化策略。
  - `优化器`：SGD（含带动量的、Nesterov版本的）、Adagrad、Adadelta、Adam。
  - `学习率调度器`：步长、常数、多步学习率调度器。
  - `回调`：早停法等。
  - `多层感知机`：前馈-反向传播、回归 & 分类损失函数、小批量训练等多种技巧；此外，为提供便捷的交互界面，我们还实现了训练进度条、指标记录与绘制、保存和加载权重等。

我们的从零实现的多层感知机：
- 在经典数据集`MNIST手写数字`多分类任务上，在相同超参数下，可以达到与Keras框架所需时间接近，同时准确度接近；
- 在经典数据集`加利福尼亚州房价预测`回归任务上，也可以在短时间内达到令人满意的结果。

该模块的独立仓库在此：[多层感知机](https://github.com/XavierSpycy/NumPyMultilayerPerceptron)。

需要注意的是，为了方便起见，您可以通过：

```python
from mlforce import nn
```

或者

```python
import mlforce.nn as nn
```

从而确保代码可以无缝衔接独立仓库中的代码示例。

仅需几行代码，即可完成一个手写数字识别的神经网络的搭建。

```python
from mlforce.nn.layers import Input, Dense
from mlforce.nn.optim import Adam
from mlforce.nn.mlp import MultilayerPerceptron

layers = [
    Input(input_dim=784),
    Dense(128, activation='relu', init='kaiming_uniform'),
    Dense(16, activation='relu', init='kaiming_uniform'),
    Dense(10, activation='softmax')
]

mlp = MultilayerPerceptron(layers)
mlp.compile(optimizer=Adam(),
            metrics=['CrossEntropy', 'Accuracy'])
mlp.fit(X_train, y_train, epochs=10, batch_size=32, use_progress_bar=True)
```

### 注意

由于 GitHub 对于 LFS（大文件储存）存在限额，因此`mf`和`nn`模块的数据集可在各自的独立仓库中查看。

## 主要依赖
* Python
* NumPy
* Pandas
* SciPy

## 文档
需要更详细的用法指导，可以查看文档字符串（`文档`也将会尽快发布）。

补充材料可以在有独立仓库的模块的`README.md`文件中查询。

## 贡献
欢迎贡献！如果您发现任何问题或者对于提升有建议，请[开启一个issue](https://github.com/XavierSpycy/MLForce/issues)或者提交一个pull请求。

## 证书
- MIT证书

## 版本历史
- v1.0.0 (2024-01-)
  - 对各个模块进行结构化重构
  - 优化多层感知机模块的实现
  - 新增非负矩阵分解模块
  
- v0.1.0 (2023-07-28)
  - 首次发布

## 作者:
- Jiarui Xu / GitHub用户名: [XavierSpycy](https://github.com/XavierSpycy)