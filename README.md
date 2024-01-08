English | [简体中文](README.zh-CN.md)

# Machine Learning Force
![PyPI](https://img.shields.io/pypi/v/mlforce)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XavierSpycy/MLForce/blob/main/basics_test.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
Our library, named `MLForce`, stands for `Machine Learning Force`. It is a Python toolkit designed to assist beginners in machine learning to understand and implement a variety of algorithms from scratch.

Every module in `MLForce` serves a distinct purpose, ranging from fundamental machine learning algorithms to neural networks and non-negative matrix factorization. You'll embark on a thorough and comprehensive journey of various machine learning algorithms, exploring and understanding them from multiple perspectives including supervised learning, deep learning, unsupervised learning, and more.

By combining ease of use with effectiveness, this library aims to solidify your theoretical understanding and practical skills in machine learning. It opens the door to the world of machine learning for you. So, embrace `MLForce` and set off on a rewarding journey into the art of machine learning!

## Installation
You can install `MLForce` using `pip`:

```
$ pip install mlforce
```

## Modules
### `basics` module:

In this module, as indicated by its name, we have implemented a variety of fundamental machine learning algorithms. These include K-nearest neighbors (KNN), linear regression, decision tree feature selection, Hidden Markov Models, KMeans, hierarchical clustering, DBSCAN, and others.

Some of these algorithms are specifically effective in particular scenarios with certain types of data inputs. Therefore, for this module, we have developed an in-built `StandardDataset` to standardize input data and labels, and to validate our algorithms.

Since this module does not have a separate repository, detailed usage instructions are currently available in the form of documentation strings.

### `mf` module:

The `mf `in our module stands for `matrix factorization`, a type of dimensionality reduction operation. Currently, this module exclusively includes **`Non-negative Matrix Factorization` implemented using NumPy**.

Non-negative Matrix Factorization (NMF) achieves dimensionality reduction by representing the original matrix ($m \times n$) through two lower-rank matrices ($m \times k$ and $k \times n$, where $k << m, n$), providing sparser feature representation. NMF is applicable in areas such as image reconstruction and topic modeling. In our original task, we successfully used NMF to reconstruct a series of noise-affected facial images, yielding significant results.

Specifically, we have implemented eight efficient variations of NMF, including those based on: $L_2$ Norm, KL Divergence, IS Divergence, $L_{2, 1}$ norm, Hypersurface Cost, $L_1$ Norm Regularization, Capped Norm, and Cauchy. These variations have been tested for numerical stability and effectiveness.

A notable feature of this module is its custom development framework for Non-negative Matrix Factorization, allowing learners and researchers to test and compare effects in-place, greatly simplifying the development process.

For detailed effects and usage methods, please refer to the independent repository: [Non-negative Matrix Factorization](https://github.com/XavierSpycy/NumPyNMF)。

Considering the convenience of integrating more modules in the future, the current usage of this module still requires:

```python
from mlforce.mf.nmf import BasicNMF
```

This allows the class to be imported and used exactly as in the independent repository, such as developing new algorithms or testing the performance of new algorithms.

```python
class ExampleNMF(BasicNMF):
    # To tailor a unique NMF algorithm, subclass BasicNMF and redefine matrix_init and update methods.
    def matrix_init(self, X, n_components, random_state=None):
        # Implement your initialization logic here.
        # Although we provide built-in methods, crafting a bespoke initialization can markedly boost performance.
        # D, R = <your_initialization_logic>
        # D, R = np.array(D), np.array(R)
        return D, R  # Ensure D, R are returned.

    def update(self, X, kwargs):
        # Implement the logic for iterative updates here.
        # Modify self.D, self.R as per your algorithm's logic.
        # flag = <convergence_criterion>
        return flag  # Return True if converged, else False.
```

### `nn` module:

The `nn` in our module stands for `neural networks`. This module primarily focuses on implementing **a Keras-style `multilayer perceptron from scratch` using NumPy**. We have implemented:

- `Activation Functions`: Nearly all activation functions found in PyTorch.
- `Hidden Layers`: Fully connected layers, batch normalization layers, dropout layers, activation layers; including various initialization strategies for fully connected layers such as Xavier uniform/normal and Kaiming uniform/normal.
- `Optimizers`: SGD (with and without momentum, Nesterov version), Adagrad, Adadelta, Adam.
- `Learning Rate Schedulers`: Step, constant, multi-step learning rate schedulers.
- `Callbacks`: Techniques like early stopping.
- `Multilayer Perceptron`: Feed-forward & backward propagation, regression & classification loss functions, batch training, and other techniques; additionally, for convenient interaction, we implemented training progress bars, metric recording and plotting, saving and loading weights, etc.

Our from-scratch multilayer perceptron:

- On the classic `MNIST handwritten digits` multi-classification task, achieves a similar accuracy and requires a comparable amount of time as the Keras framework with the same hyperparameters.

- On the `California Housing Price Prediction` regression task, it also achieves satisfactory results in a short amount of time.

The independent repository for this module can be found here: [Multilayer Perceptron](https://github.com/XavierSpycy/NumPyMultilayerPerceptron)。

For convenience, you can use:
```python
from mlforce import nn
```
or
```python
import mlforce.nn as nn
```

to ensure seamless integration with the code examples in the independent repository.

With just a few lines of code, you can build a neural network for handwriting digit recognition.

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

### Note

Due to GitHub's limitations on LFS (Large File Storage), the datasets for the mf and nn modules can be found in their respective independent repositories.

## Main Dependencies
* Python
* NumPy
* Pandas
* SciPy

## Documentation
For more detailed usage instructions, refer to the documentation strings (comprehensive `Documentation` will also be published shortly).

Supplementary materials can be found in the `README.md` files of modules with independent repositories.

## Contributions
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/XavierSpycy/MLForce/issues) or submit a pull request.


## License
- MIT License

## Version History
- v1.0.0 (2024-01-)
  - Structural refactoring of various modules
  - Optimization of the multilayer perceptron module implementation
  - Addition of the non-negative matrix factorization module
  
- v0.1.0 (2023-07-28)
  - Initial release

## Author:
- Jiarui Xu / GitHub Username: [XavierSpycy](https://github.com/XavierSpycy)