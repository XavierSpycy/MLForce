import pandas as pd

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