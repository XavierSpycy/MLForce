# Machine Learning Force -- MLForce
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/m/XavierSpycy/mlsolutions/main)
![PyPI - Downloads](https://img.shields.io/pypi/:period/:mlforce)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/XavierSpycy/mlsolutions/main?labpath=test.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
My library, named "MLForce" which stands for Machine Learning Force, is a comprehensive Python toolkit meticulously crafted to cater to the needs of machine learning beginners. With a user-friendly design, it aims to empower aspiring data enthusiasts and learners in the fields of machine learning and data analysis. Each module in "MLStarter" serves a distinct purpose, providing a seamless and efficient experience for users throughout the entire machine learning journey.

By combining ease of use with powerful functionality, this library aims to nurture your understanding and practical skills in machine learning, unlocking the doors to endless possibilities in the realm of data-driven solutions. So, embrace "MaLL" and embark on a rewarding journey of mastering the art of machine learning!

## Installation
You can install MLForce using pip:
```
pip install mlforce
```

## Quick Start
Here's a quick example of how to use MLStarter Library:

```python
# Import modules
from mlforce.datasets import StandardDataset
from mlforce.mlsolutions import KNearestNeighbor

# Load the datasets
ds = StandardDataset()
ds.load_knn('numeric')
X, y = ds.data, ds.target
new_example = ds.new_example 

# Call a particular algorithm
knn = KNearestNeighbor(k=3)
knn.fit(X, y)
knn.predict(new_example)
print(knn)

# Ouputs:
# The closest nearest neighbors are ex.1, 0, 2. The majority of Label is yes; hence, 3-Nearest Neighbor predicts Label = yes.
```
## Modules
* **datasets** module:
  The "datasets" module serves as a reliable data-loading component that efficiently handles various datasets and seamlessly feeds them to the   models.
* **mlsolutions**
  The "mlsolutions" module is a comprehensive collection of essential algorithms in the fields of machine learning and data mining. It provides users with a powerful toolkit to tackle a wide range of challenges in these domains.
* **mlperceptron**
  The "mlperceptron" module is dedicated to implementing multi-layer perceptrons using the popular numerical computing library, NumPy.
* **digitrecognizer**
  The "digitrecognizer" module specializes in digit recognition tasks, such as the well-known MNIST dataset. It enables seamless training and evaluation of models for accurate digit classification.

## Dependencies
* Python
* numpy
* pandas
* scipy
* torch

## Documentation
For more detailed usage instructions, check out the Documentation.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please [Open an issue](https://github.com/XavierSpycy/MLForce/issues) or submit a pull request.


## License
MaLL is licensed under the MIT License.

## Version History
- v0.1.0 (2023-07-28)
  * Initial release of MaLL.

## Author and contact:
Author: Jiarui Xu      
Email: xujiarui98@foxmail.com
