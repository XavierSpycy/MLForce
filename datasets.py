import numpy as np
import pandas as pd
from mlsolutions import KNearestNeighbor

class StandardDataset(object):
    """
        datasets class which is used to implement dataset read operations.
    """
    """
        datasets类, 用于实现数据集读取操作。
    """

    def reset(self):
        """
            This method will reset all attributes that were set during the previous data loading process, 
            thus preventing any potential confusion for users.
        """
        """
            这个方法将重置在上一次数据加载过程中设置的所有属性, 
            从而避免可能对用户产生的任何潜在困惑。
        """
        
        self.dataset = None
        self.matrix = None
        self.data = None
        self.target = None
        self.new_example = None
        self.cluster_labels = None
        self.initial_probability = None
        self.transition = None
        self.emission = None
        
    def overview(self):
        """
            When this method is called, it will print the data loaded in the current dataset. 
            The specific content to be printed will be determined by the actual situation of the current dataset.
        """
        """
            调用该方法时, 将会打印当前数据集中加载的数据。
            具体的打印内容将会由当前数据集的实际情况而决定。
        """
        
        if self.option == 'train':
            print("The training set is:")
            print(self.dataset)
            if self.new_example is not None:
                print()
                print("The new example is:")
                print(self.new_example)
        elif self.option == 'matrix':
            if self.matrix == 'distance':
                print("The distance matrix is:")
                print(self.dataset)
                if self.cluster_labels is not None:
                    print()
                    print("The cluster labels are:")
                    print(self.cluster_labels)
            elif self.matrix == 'similarity':
                print("The similarity matrix is:")
                print(self.dataset)
            elif self.matrix == 'markov':
                print("The transition matrix is:")
                print(self.dataset)
            elif self.matrix == 'hiddenmarkov':
                print("The initial probabilities are:")
                print(self.initial_probability.to_string(dtype=False))
                print()
                print("The transition matrix is:")
                print(self.transition)
                print()
                print("The emission matrix is:")
                print(self.emission)

    def load_knn(self, mode):
        """
            Load the K-nearest neighbors dataset.
        """
        """
            加载K最近邻数据集
        """
        
        self.reset()
        self.option = "train"
        if mode == 'numeric':
            self.dataset = pd.DataFrame({
                "Feature 1": [1, 3, 3, 5],
                "Feature 2": [3, 5, 2, 2],
                "Feature 3": [1, 2, 2, 3],
                "Label": ["yes", "yes", "no", "no"]}
            )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]
            self.new_example = pd.DataFrame({'Feature 1': 2, 'Feature 2': 4, 'Feature 3': 2}, index=[0])
        elif mode == 'nominal':
            self.dataset = pd.DataFrame({
                "Feature 1": ["<=30", "<=30", "[31,40]", ">40", ">40", "[31,40]", "<=30", "[31,40]", ">40"],
                "Feature 2": ["high", "high", "high", "medium", "low", "low", "medium", "medium", "medium"],
                "Feature 3": ["no", "no", "no", "no", "yes", "yes", "no", "no", "no"],
                "Feature 4": ["fair", "excellent", "fair", "fair", "excellent", "excellent", "fair", "excellent", "excellent"],
                "Label": ["no", "no", "yes", "yes", "no", "yes", "no", "yes", "no"]}
            )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]
            self.new_example = pd.DataFrame({'Feature 1': '<=30', 'Feature 2': 'medium', 'Feature 3': 'yes', 'Feature 4': 'fair'}, index=[0])
    
    def load_onerule(self):
        self.reset()
        self.option = "train"
        self.dataset = pd.DataFrame({
            "Feature 1": ["bad", "unknown", "unknown", "unknown", "unknown", "unknown", "bad", "bad", "good", "good", "good", "good", "good", "bad"],
            "Feature 2": ["high", "high", "low", "low", "low", "low", "low", "low", "low", "high", "high", "high", "high", "high"],
            "Feature 3": ["none", "none", "none", "none", "none", "adequate", "none", "adequate", "none", "adequate", "none", "none", "none", "none"],
            "Feature 4": ["low", "average", "average", "low", "high", "high", "low", "high", "high", "high", "low", "average", "high", "average"],
            "Label": ["high", "high", "moderate", "high", "low", "low", "high", "moderate", "low", "low", "high", "moderate", "low", "high"]}
        )
        self.data = self.dataset.iloc[:, :-1]
        self.target = self.dataset.iloc[:, -1]
        self.new_example = pd.DataFrame({'Feature 1': 'unknown', 'Feature 2': 'low', 'Feature 3': 'none', 'Feature 4': 'average'}, index=[0])

    def load_prism(self):
        self.reset()
        self.option = "train"
        self.dataset = pd.DataFrame({
            "Feature 1": ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
            "Feature 2": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "hot", "mild"],
            "Feature 3": ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
            "Feature 4": ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"],
            "Label": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]}
        )
        self.data = self.dataset.iloc[:, :-1]
        self.target = self.dataset.iloc[:, -1]
    
    def load_naivebayes(self, code):
        self.reset()
        self.option = "train"
        if code == 'nb1':
            self.dataset = pd.DataFrame({
                "Feature 1": ["yes", "no", "no", "yes", "yes", "no", "yes", "no", "no", "no"],
                "Feature 2": ["single", "married", "single", "married", "divorced", "married", "divorced", "single", "married", "single"],
                "Feature 3": ["very high", "high", "medium", "very high", "high", "low", "very high", "high", "medium", "low"],
                "Label": ["yes", "yes", "no", "no", "yes", "no", "no", "yes", "no", "yes"]}
            )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]
            self.new_example = pd.DataFrame({'Feature 1': 'no', 'Feature 2': 'married', 'Feature 3': 'very high'}, index=[0])
        elif code == 'nb2':
            self.dataset = pd.DataFrame({
                "Feature 1": ["yes", "no", "no", "yes", "yes", "no", "yes", "no", "no", "no"],
                "Feature 2": ["single", "married", "single", "married", "divorced", "married", "divorced", "single", "married", "single"],
                "Feature 3": [125, 100, 70, 120, 95, 60, 220, 85, 75, 90],
                "Label": ["yes", "yes", "no", "no", "yes", "no", "no", "yes", "no", "yes"]}
            )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]
            self.new_example = pd.DataFrame({'Feature 1': 'no', 'Feature 2': 'married', 'Feature 3': 120}, index=[0])
        elif code == 'nb3':
            self.dataset = pd.DataFrame({
                "Feature 1": ["nice", "nice", "boring", "boring", "nice", "boring", "boring"],
                "Feature 2": ["sunny", "sunny", "rainy", "sunny", "rainy", "rainy", "rainy"],
                "Feature 3": ["yes", "no", "yes", "yes", "yes", "no", "no"],
                "Feature 4": ["annoying", "annoying", "great", "great", "great", "annoying", "great"],
                "Label": ["good", "bad", "good", "bad", "good", "good", "good"]}
            )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]
            self.new_example = pd.DataFrame({'Feature 1': 'boring', 'Feature 2': 'sunny', 'Feature 3': 'yes', 'Feature 4': 'annoying'}, index=[0])

    def load_decisiontree(self, code):
        self.reset()
        self.option = "train"
        if code == 'dt1':
            self.dataset = pd.DataFrame({
            "Feature 1": ["circle", "circle", "square", "triangle", "square", "square", "square", "circle"],
            "Feature 2": ["blue", "blue", "blue", "blue", "red", "blue", "red", "red"],
            "Label": ["+", "+", "-", "-", "+", "-", "+", "+"]}
        )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]
        elif code == 'dt2':
            self.dataset = pd.DataFrame({
            "Feature 1": ["nice", "nice", "boring", "boring", "nice", "boring", "boring"],
            "Feature 2": ["sunny", "sunny", "rainy", "sunny", "rainy", "rainy", "rainy"],
            "Feature 3": ["yes", "no", "yes", "no", "yes", "no", "no"],
            "Label": ["good", "bad", "good", "bad", "good", "good", "good"]}
        )
            self.data = self.dataset.iloc[:, :-1]
            self.target = self.dataset.iloc[:, -1]

    def load_perceptron(self):
        self.reset()
        self.option = "train"
        self.dataset = pd.DataFrame({
            "Feature 1": [1, 0, 1, 1, 0],
            "Feature 2": [0, 1, 1, 1, 0],
            "Feature 3": [0, 1, 0, 1, 1],
            "Label": [1, 0, 1, 0, 0]}
        )
        self.data = self.dataset.iloc[:, :-1]
        self.target = self.dataset.iloc[:, -1]

    def load_kmeans(self, code):
        self.reset()
        self.option = "matrix"
        if code == 'kmeans1':
            self.dataset = pd.DataFrame({
                "A": [0, 2, 7, 10, 1],
                "B": [2, 0, 3, 4, 6],
                "C": [7, 3, 0, 5, 9],
                "D": [10, 4, 5, 0, 8],
                "E": [1, 6, 9, 8, 0]},
                index = ["A", "B", "C", "D", "E"]
            )
        elif code == 'kmeans2':
            self.dataset = pd.DataFrame({
                "A": [0, 10, 2, 1, 12, 5, 4],
                "B": [10, 0, 4, 3, 6, 23, 7],
                "C": [2, 4, 0, 5, 9, 14, 19],
                "D": [1, 3, 5, 0, 1, 7, 4],
                "E": [12, 6, 9, 1, 0, 2, 18],
                "F": [5, 23, 14, 7, 2, 0, 6],
                "G": [4, 7 ,19, 4, 18, 6, 0]},
                index = ["A", "B", "C", "D", "E", "F", "G"]
            )
        self.matrix = 'distance'

    def load_hierarchical(self):
        self.load_kmeans('kmeans1')
    
    def load_dbscan(self, code):
        self.reset()
        self.option = "matrix"
        if code == 'dbscan1':
            self.dataset = pd.DataFrame({
                "A": [0, 1, 4, 5, 6],
                "B": [1, 0, 2, 6, 7],
                "C": [4, 2, 0, 3, 4],
                "D": [5, 6, 3, 0, 1],
                "E": [6, 7, 4, 1, 0]},
                index = ["A", "B", "C", "D", "E"]
            )
        elif code == 'dbscan2':
            self.dataset = pd.DataFrame({
                "A": [0.0, 5.0, 6.0, 3.6, 7.0, 7.2, 8.0, 2.2],
                "B": [5.0, 0.0, 6.1, 4.2, 5.0, 4.1, 3.2, 4.5],
                "C": [6.0, 6.1, 0.0, 5.0, 1.5, 1.5, 7.5, 6.5],
                "D": [3.6, 4.2, 5.0, 0.0, 3.6, 4.1, 7.2, 1.5],
                "E": [7.0, 5.0, 1.5, 3.6, 0.0, 1.4, 6.7, 5.0],
                "F": [7.2, 4.1, 1.5, 4.1, 1.4, 0.0, 5.4, 5.5],
                "G": [8.0, 3.2, 7.5, 7.2, 6.7, 5.4, 0.0, 7.5],
                "H": [2.2, 4.5, 6.5, 1.5, 5.0, 5.5, 7.5, 0.0]},
                index = ["A", "B", "C", "D", "E", "F", "G", "H"]
            )
        self.matrix = 'distance'

    def load_cluster_evaluate(self, method):
        self.reset()
        self.option = "matrix"
        if method == 'sihouette_coefficient':
            self.dataset = pd.DataFrame({
                "A": [.00, .10, .65, .55],
                "B": [.10, .00, .70, .60],
                "C": [.65, .70, .00, .30],
                "D": [.55, .60, .30, .00]},
                index = ["A", "B", "C", "D"]
            )
            self.matrix = 'distance'
        elif method == 'correlation':
            self.dataset = pd.DataFrame({
                "A": [1.00, 1.00, 0.08, 0.25],
                "B": [1.00, 1.00, 0.00, 0.17],
                "C": [0.08, 0.00, 1.00, 0.67],
                "D": [0.25, 0.17, 0.67, 1.00]},
                index = ["A", "B", "C", "D"]
            )
            self.matrix = 'similarity'
        self.cluster_labels = pd.Series([1, 1, 2, 2], index=["A", "B", "C", "D"])
    
    def load_markov(self):
        self.reset()
        self.option = "matrix"
        self.dataset = pd.DataFrame({
            "State A": [0.8, 0.2, 0.2],
            "State B": [0.05, 0.6, 0.3],
            "State C": [0.15, 0.2, 0.5]},
            index = ["State A", "State B", "State C"]
        )
        self.matrix = 'markov'
    
    def load_hidden_markov(self, code):
        self.reset()
        if code == 'hm1':
            self.initial_probability = pd.Series({'State A': 0.4, 'State B': 0.3, 'State C': 0.3})
            self.transition = pd.DataFrame({
                "State A": [0.8, 0.2, 0.2],
                "State B": [0.05, 0.6, 0.3],
                "State C": [0.15, 0.2, 0.5]},
                index = ["State A", "State B", "State C"]
            )
            self.emission = pd.DataFrame({
                "Observation 1": [.1, .8, .3],
                "Observation 2": [.9, .2, .7]},
                index = ["State A", "State B", "State C"]
            )
        elif code == 'hm2':
            self.initial_probability = pd.Series({'State A': 0.5, 'State B': 0.5})
            self.transition = pd.DataFrame({
                "State A": [0.6, 0.5],
                "State B": [0.4, 0.5]},
                index = ["State A", "State B"]
            )
            self.emission = pd.DataFrame({
                "Observation 1": [.6, .2],
                "Observation 2": [.3, .3],
                "Observation 3": [.1, .5]},
                index = ["State A", "State B"]
            )
        self.matrix = 'hiddenmarkov'

class MLPDataset(object):
  def __init__(self, 
              proportion=0.1, random_state=None,
              norm_method='MinMax',
              na_method='mean', fillna_params={'fixed': 0, 'axis': 0},
              distribution='gaussian', param1=0, param2=1e-2):
    """"Load the dataset and pre-processing"""
    """Parameters:
        proportion, corresponds to the proportion of validation set in the input, default: 0.1;
        random_state, corresponds to the random seed, default: None;
        norm_method, corresponds to the normalization method, default: 'MinMax';
        na_method, corresponds to the strategy to deal with NA values, default: 'mean';
        fillna_params, corresponds to the parameters of some specific methods, default: {'fixed': 0, 'axis': 0};
        distribution, corresponds to the noise distribution when adding the noise, default: 'gaussian';
        params1, params2, corresponds to the paramters of a specific distribution, default: param1 - 0, param2 - 1e-2.
    """
    self.dir = "./data/"
    self.X = np.load(self.dir + "train_data.npy")
    self.y = np.load(self.dir + "train_label.npy")
    self.X_test = np.load(self.dir + "test_data.npy")
    self.y_test = np.load(self.dir + "test_label.npy")
    
    # Parameters of train_validation_split() method
    self.proportion = proportion
    self.random_state = random_state
    
    # Parameters of normalize_() method
    self.norm_method = norm_method
    
    # Parameters of fillna_() method
    self.na_method = na_method
    self.fillna_params = fillna_params
    
    # Paramters of add_noise(), including the distribution of the noise
    self.distribution = distribution
    self.param1 = param1
    self.param2 = param2

  def train_validation_split(self):
    """Split traing set into new traing set and validation set, by default 90:10"""
    # Set random seed
    np.random.seed(self.random_state)
  
    # Shuffle indices based on number of smaples
    n_samples = self.X.shape[0]
    indices = np.random.permutation(n_samples)
  
    # Split indices into training and validation indices
    validation_size = int(n_samples * self.proportion)
    train_indices, valid_indices = indices[validation_size:], indices[:validation_size]
  
    # Split data by indices
    self.X_train, self.X_valid = self.X[train_indices], self.X[valid_indices]
    self.y_train, self.y_valid = self.y[train_indices], self.y[valid_indices]
  
  def center_(self):
    """Data centering"""
    mean = np.mean(self.X, axis=0)
    self.X -= mean
    self.X_test -= mean

  def normalize_(self):
    """Data normalization of the input"""
    if self.norm_method == "MinMax":
      param1 = np.min(self.X, axis=0)
      param2 = np.max(self.X, axis=0) - param1
    elif self.method == "Standard":
      param1 = np.mean(self.X, axis=0)
      param2 = np.std(self.X, axis=0)
    self.X = (self.X - param1) / param2
    self.X_test = (self.X_test - param1) / param2

  def fillna_(self):
    """ Deal with missing values of the input"""
    axis = self.fillna_params["axis"]
    
    if self.na_method != "drop":
      if self.na_method == "fixed":
        # Fill NA values with a fixed value
        value = self.fillna_params["fixed"]
      elif self.na_method == "mean":
        # Fill NA values with their mean
        value = np.nanmean(self.X, axis=axis)
        # Fill NA values with their median
      elif self.na_method == "median":
        value = np.nanmedian(self.X, axis=axis)
      
      if axis == 0:
        self.X = np.where(np.isnan(self.X), np.tile(value, (self.X.shape[0], 1)), self.X)
      else:
        self.X = np.where(np.isnan(self.X), value, self.X)

    # Delete all rows with NA values
    else:
      self.X = self.X[~np.isnan(self.X).any(axis=axis)]   
  
  def add_noise_(self):
    """Add noise to the input"""

    """Note: If distribution equals to gaussian, 
              param1, corresponds to mean, 
              param2, corresponds to standard deviation;

              If distribution equals to uniform,
              param1, corresponds to lower bound, 
              param2, corresponds to upper bound.
    """
    if self.distribution == "gaussian":
      noise = np.random.normal(self.param1, self.param2, self.X.shape)
      self.X += noise
    elif self.distribution == "uniform":
      noise = np.random.uniform(self.param1, self.param2, self.X.shape)
      self.X += noise

  def one_hot_encode(self, labels):
    rows = labels.shape[0]
    num_classes = len(np.unique(labels))
    output = np.zeros((rows, num_classes))
    for row in range(rows):
      label = labels[row, 0]
      output[row, label] = 1
    return output
  
  def load(self, split=True, fillna=False, center=False, normalize=False, add_noise=False, encode=False):
    """Load the datasets"""
    """Parameters: All parameters here are boolean parameters
        split, corresponds to whether we split datasets to training set and validation set, default: True;
        fillna, corresponds to whether we deal with NA values, default: False;
        center, corresponds to whether we center the features, default: False;
        normalizae, corresponds to whether we normalize the features, default: False;
        add_noise, corresponds to whether we add some noises in training set, default: False.
    """
    # If we decide to deal with NA values:
    if fillna:
      self.fillna_()

    # If we decide to center the data:
    if center:
      self.center_()

    # If we decide to normalize the data:
    if normalize:
      self.normalize_()

    # If we decide to add some random noise to the data:
    if add_noise:
      self.add_noise_()
    
    # If we decide to encode lables using one-hot, but we will automatically use one-hot encoding in the training process;
    if encode:
      self.y = self.one_hot_encode(self.y)
      self.y_test = self.one_hot_encode(self.y_test)

    # If we decide to split the dataset to training set, validation set, and test set:
    # this will be used by default.
    if split:
      self.train_validation_split()
      return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test
    else:
      return self.X, self.y, self.X_test, self.y_test