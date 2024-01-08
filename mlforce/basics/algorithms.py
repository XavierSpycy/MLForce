"""The code comments will include bilingual content, including both English and Chinese."""
"""代码注释将包含双语内容, 包括英文和中文。"""

"""由于我们使用了一些Python的内置函数, 在选择最大值或最小值时, 
如果存在两值相等, 将会按照Python的默认行为选择最终值。
所以, 在我们构造的函数中, 当两值相等时, 我们并没有进行随机选择。
这里, 我们对此进行说明, 注意我们可能会在今后的更新中对这一点进行逐步优化。"""
"""Due to our use of certain built-in functions in Python, when selecting the maximum or minimum value, 
if there is a tie between two values, the final value will be chosen according to Python's default behavior. 
Therefore, in our custom function, we did not perform a random selection when there is a tie between two values. 
Here, we would like to clarify this and note that we may gradually improve this aspect in future updates."""

"""At the same time, since we aim to align the output results of objects with our teaching objectives, 
we have not pursued the optimal implementation for specific algorithmic components. 
In future updates, we will strive to enhance the efficiency of various algorithms and meet the diverse needs of our users."""
"""同时, 因为我们希望对象的输出结果符合我们的教学目的, 
因此我们并未追求具体的算法部分的最优实现。
在今后的更新中, 我们将会力求完善不同算法的效率, 并满足用户的多种需求。"""

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from functools import reduce

class KNearestNeighbor(object):
    """
        Implement K-nearest neighbors
    """
    """
        实现K最近邻
    """
    
    """Example/示例:
        Suppose X, y, and new_example have been already defined (X: training inputs, y: training labels, new_examples: new examples).
        假设X, y, new_examples均已被定义(X: 训练输入, y: 训练标签, new_examples: 新样例)

        >>> knn = KNearestNeighbor(k=3)
        >>> knn.fit(X, y)
        >>> prediction = knn.predict(new_examples)
        >>> print(knn)
        >>> The closest nearest neighbors are ex.5, 15, 12. The majority of Movie is B; hence, 3-Nearest Neighbor predicts Movie = B.
            The closest nearest neighbors are ex.5, 10, 2. The majority of Movie is A; hence, 3-Nearest Neighbor predicts Movie = A.
            The closest nearest neighbors are ex.3, 12, 13. The majority of Movie is B; hence, 3-Nearest Neighbor predicts Movie = B.
    """

    def __init__(self, k: int) -> None:
        """
            Initialize the object KNearestNeighbor
        """
        """
            初始化对象KNearestNeighbor
        """
        
        '''Parameter:
            k, corresponds to the value of k in k-Nearest Neighbors.
        '''
        '''参数:
            k, 对应k最近邻中的k值。
        '''

        """
            Very sensitive to the value of k / 对k值敏感
            Rule of thumb: k <= sqrt(number of training examples) / 经验法则: k <= sqrt(训练样例的数量)
            Commercial packages typically use k = 10 / 商业包通常用k = 10
        """

        # Check the data type of the inputs / 检查输入的数据类型
        if not isinstance(k, int):
            raise TypeError("The k of K-Nearest Neighbor must be an integer.")
        # Verify the validity of the value of k in k-Nearest Neighbor / 验证k最近邻中k值的有效性
        if k < 1:
            raise ValueError("The k of K-Nearest Neighbor must be larger than 1.")
        self.k = k
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
            Fit the model to the training data
        """
        """
            使模型拟合训练数据
        """
        
        '''Parameters:
            X_train, corresponds to the training inputs;
            y_train, corresponds to the training labels.
        '''
        '''参数:
            X_train, 对应训练输入;
            y_train, 对应训练标签。
        '''
        
        # Detect the data type of attributes, including 'numeric' and 'nominal'
        self.feature_type = None
        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("Labels must be a pandas Series.")
        if len(X_train) != len(y_train):
            raise RuntimeError("Length of inputs and labels must match.")
        if len(y_train) < self.k:
            raise ValueError("Insufficient number of the training data.")
        for column in X_train.columns:
            if self.feature_type is None:
                if pd.api.types.is_numeric_dtype(X_train[column]):
                    self.feature_type = 'numeric'
                else:
                    self.feature_type = 'nominal'
            else:
                if pd.api.types.is_numeric_dtype(X_train[column]):
                    feature_type = 'numeric'
                else:
                    feature_type = 'nominal'
                if feature_type != self.feature_type:
                    raise TypeError("Data type must be normalized.")
        
        # Store the training examples / 储存训练样例
        self.features = set(X_train.columns.tolist())
        self.category = y_train.name
        self.train_examples = X_train.index
        self.train_inputs = X_train.to_numpy()
        self.train_labels = y_train
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
            Predict labels for the test inputs
        """
        """
            为测试输入预测标签
        """

        '''Parameter:
            X_test, corresponds to the test inputs.
        '''
        '''参数:
            X_test, 对应测试输入。
        '''
        
        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame.")
        test_features = set(X_test.columns.tolist())
        if test_features != self.features:
            raise ValueError("Training and test features must match.")
            
        self.predictions = []
        self.evidences = []
        if self.feature_type == 'numeric':
            for index, data in X_test.iterrows():
                # Compute the Euclidean distance(also called L2 norm) using broadcasting mechanism
                # 用广播机制计算欧几里得距离(也被称为L2范数)
                distance = np.sqrt(np.sum(np.power(self.train_inputs - data.to_numpy(), 2), axis=1))
                # Retrieve the indices the top-k closest example(s) to a given new example (with no guarantee of being sorted in order)
                # 取回前k个距离给定的新样例最近的样例的切片(不确保按顺序排序)
                indices = np.argpartition(distance, self.k)[:self.k]
                # Sort / 排序
                sorted_indices = indices[np.argsort(distance[indices])]
                self.predictions.append(self.train_labels[sorted_indices].value_counts().idxmax())
                self.evidences.append(self.train_examples[sorted_indices].tolist())
        elif self.feature_type == 'nominal':
            for index, data in X_test.iterrows():
                # Compute the distance for nominal attributes (also using broadcasting mechanism) / 为分类属性计算距离(同样适用广播机制)
                # In our implementation, we define the distance for nominal attributes as follows / 在我们的实现中, 我们定义分类属性的距离如下: 
                # Distance = 0 if attribute values are the same / 距离 = 0, 若属性值相同;
                # Distance = 1 if attribute values are not the same / 距离 = 1, 若属性值不同.
                distance = np.sqrt(np.sum(self.train_inputs != data.to_numpy(), axis=1))
                # The same operations with numeric attributes / 与数值属性相同的操作
                indices = np.argpartition(distance, self.k)[:self.k]
                sorted_indices = indices[np.argsort(distance[indices])]
                self.predictions.append(self.train_labels[sorted_indices].value_counts().idxmax())
                self.evidences.append(self.train_examples[sorted_indices].tolist())
        return pd.Series(self.predictions)
        
    def __str__(self) -> str:
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """
        
        # Distinguish a subtle variation when "k" is singular or plural / 区别当k为单数或复数时细微的不同
        if not hasattr(self, 'evidences'):
            return "To reach your expectation, please ensure that you have called the 'predict' method first."
        if self.k == 1:
            string_evidence = ["The closest nearest neighbor is ex." + ", ".join(map(str, evidence)) for evidence in self.evidences]
            string_prediction = [f"Hence, {self.k}-Nearest Neighbor predicts {self.category} = {prediction}." for prediction in self.predictions]
        else:
            string_evidence = ["The closest nearest neighbors are ex." + ", ".join(map(str, evidence)) for evidence in self.evidences]
            string_prediction = [f"The majority of {self.category} is {prediction}; hence, {self.k}-Nearest Neighbor predicts {self.category} = {prediction}." for prediction in self.predictions]
        string_outputs = [str_evi + ". " + str_pred for str_evi, str_pred in zip(string_evidence, string_prediction)]
        return "\n".join(string_outputs)
    
class OneRule(object):
    """
        Implement 1R, which stands for 1-rule. 
        The rule can be represented as 1-level decision tree (or decision stump).
    """
    """
        实现1R算法, 其中1R代表一条规则。
        该规则可以表示为一级决策树(或决策桩)。
    """
    
    """Examples / 示例:
        Suppose X, y, and new_example have been already defined (X: training inputs, y: training labels, new_examples: new examples).
        假设X, y, new_example均已被定义(X: 训练输入, y: 训练标签, new_examples: 新样例)
        
        **
        # Example / 示例:
        # Initialize an OneRule object / 实例化一个OneRule对象
        >>> oneR = OneRule()
        # Fit the model to the data / 使模型拟合数据
        >>> oneR.fit(X, y)
        # Generate the rule / 生成规则
        >>> oneR.generate()
        # Predict the output(s) / 基于规则预测数据
        # The variable "prediction" will store the model's prediction(s) based on the test inputs.
        # 变量"prediction"将会储存模型基于测试输入的的输出。
        >>> prediction = oneR.predict(new_examples)
        # 打印模型输出
        >>> print(oneR)
        
        # The output will be in the following format / 输出形如: 
        >>> The rule based on Age has the minimum number of errors, whose error rate is 0.500. Hence, 1R produces the following rule:
            if Age = <18 then Vote = Neutral; else if Age = >60 then Vote = No; else if Age = [35, 60] then Vote = Yes; else if Age = [18, 35) then Vote = Yes.

            The new example 0 has Age = <18 and hence will be classified as Vote = Neutral.
            The new example 1 has Age = >60 and hence will be classified as Vote = No.
        
    """

    def __init__(self, bins:int = 5) -> None:
        """
            Initialize the object OneRule
        """
        """
            初始化对象OneRule
        """

        '''Parameter:
            bins, corresponds to the number of bins when fitting the model to numerical attributes; Default: 5.
        '''
        '''参数:
            bins, 对应当使模型拟合数值属性时的区间数量。
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(bins, int):
            raise TypeError("Discretization parameter must be an integer.")
        self.bins = bins

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
            Fit the model to the training data
        """
        """
            使模型拟合数据
        """
        
        '''Parameters:
            X, corresponds to the training inputs;
            y, corresponds to the training labels.
        '''
        '''参数:
            X, 对应训练输入;
            y, 对应训练标签。
        '''
        
        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("Labels must be a pandas Series.")
        if len(X) != len(y):
            raise RuntimeError("Length of inputs and labels must match.")
        
        # Discretize the numerical attributes / 离散化数值属性
        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                X[column] = pd.cut(X[column], bins=self.bins)
            
        # Convert the column index into a list object / 将列索引转化为列表对象
        self.features = X.columns.tolist()
        # Store the class name of the labels / 储存标签的类名
        self.labelname = y.name
        # Conpute the total number of the training examples / 计算训练样例的总数
        num_total = len(y)
        # Intialize the best correct number / 初始化最优正确数量
        best_correct = 0
        
        # Iterate all features / 遍历所有特征
        for feature in self.features:
            # Initialize the correct number of each feature / 初始化每个元素的正确数量
            num_correct = 0
            feature_values = X[feature].unique().tolist()
            # Iterate all values of each feature / 遍历每个特征的所有值
            for feature_value in feature_values:
                # Accumulate the count of the maximum coverage of correct labels / 累计最大覆盖的正确标签的数量
                num_correct += y[X[feature] == feature_value].value_counts().max()
            # Compare the current correct count with the best correct count / 比较当前的正确数量与最佳正确数量
            if num_correct > best_correct:
                best_correct = num_correct
                self.best_feature = feature
        # Compute the best error rate / 计算最佳的错误率
        self.best_error_rate = 1- num_correct / num_total
        # Store all values of the best feature in a list object / 将最佳特征的所有值储存至列表对象
        best_feature_values = X[self.best_feature].unique().tolist()
        # Store the prediction rules in a dictionary object / 将预测规则储存至字典对象
        self.feature_prediction_pairs = {}
        for best_feature_value in best_feature_values:
            prediction = y[X[self.best_feature] == best_feature_value].value_counts().idxmax()
            self.feature_prediction_pairs[best_feature_value] = prediction
            
    def generate(self) -> None:
        """
            Generate the rule in the string format
        """
        """
            生成字符串形式的规则
        """

        # Merge the rules stored in the list into a string / 合并存放在列表中的规则为一个字符串
        self.generated_rule_list = []
        for feature, prediction in self.feature_prediction_pairs.items():
            self.generated_rule_list.append(f"if {self.best_feature} = {feature} then {self.labelname} = {prediction}")
        self.generated_rule = "; else ".join(self.generated_rule_list) + "."
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
            Predict the labels for the test inputs
        """
        """
            为测试输入预测标签
        """

        # # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame..")
        if self.best_feature not in X_test:
            raise ValueError("The best attribute is not in present in the test data.")
        
        # Predict labels for the test inputs using the generated rule / 用生成的规则为测试输入预测标签
        self.new_examples = X_test.index.tolist()
        self.predictions = []
        self.rule_based = []
        rule_based_feature = X_test[self.best_feature]
        for feature_value in rule_based_feature:
            prediction = self.feature_prediction_pairs[feature_value]
            self.rule_based.append(feature_value)
            self.predictions.append(prediction)
        return pd.Series(self.predictions)
    
    def __str__(self):
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        rule_result = ""
        prediction_result = ""
        if hasattr(self, 'generated_rule'):
            rule_result = f"The rule based on {self.best_feature} has the minimum number of errors, whose error rate is {self.best_error_rate:.3f}. Hence, 1R produces the following rule:\n{self.generated_rule}\n"
        else:
            return "To reach your expectation, please ensure that you have called the 'generate' method first."
        if hasattr(self, 'predictions'):
            prediction_string = [f"The new example {new_example} has {self.best_feature} = {rule} and hence will be classified as {self.labelname} = {prediction}." for new_example, rule, prediction in zip(self.new_examples, self.rule_based, self.predictions)]
            prediction_result = "\n".join(prediction_string)
        return rule_result + "\n" + prediction_result

class PRISM(object):
    """
        Implement PRISM, a rule-based covering algorithm;
        It constructs a set of if-then rulses that cover all examples from a certain class 
        and do not cover any examples from the other classes.
    """
    """
        实现PRISM, 一种基于规则的覆盖算法;
        它构建一个if-then规则的集合, 覆盖来自特定类别的所有样例
        并且不覆盖任何来自其他类别的样例。
    """

    """Examples / 示例:
        Suppose X, y, and new_example have been already defined (X: training inputs, y: training labels, new_examples: new examples).
        假设X, y, new_example均已被定义(X: 训练输入, y: 训练标签, new_examples: 新样例)
        
        **
        # Example / 示例:
        
        
    """

    def best_condition(self, inputs, labels):
        """
            Compute the optimal feature and its optimal value which should be added into the rules in the recent feature space
        """
        """
        """
        best_correct = 0
        best_accuracy = 0.0
        best_feature = None
        best_feature_value = None
        
        for feature in inputs.columns:
            feature_Series = inputs[feature]
            feature_values = feature_Series.unique()
            for feature_value in feature_values:
                condition = feature_Series == feature_value
                conditional_labels = labels[condition]
                num_total = len(conditional_labels)
                correct_condition = conditional_labels == self.target_class
                num_correct = len(conditional_labels[correct_condition])
                curr_accuracy = num_correct / num_total
                if curr_accuracy > best_accuracy or (curr_accuracy == best_accuracy and num_correct > best_correct):
                    best_accuracy = curr_accuracy
                    best_correct = num_correct
                    best_feature = feature
                    best_feature_value = feature_value
        return best_feature, best_feature_value
    
    def best_condition_combination(self, inputs, labels):
        best_features = []
        best_feature_values = []
        while len(labels.unique()) > 1:
            best_feature, best_feature_value = self.best_condition(inputs, labels)
            best_features.append(best_feature)
            best_feature_values.append(best_feature_value)
            update_condition = inputs[best_feature] == best_feature_value
            inputs, labels = inputs[update_condition], labels[update_condition]
        return best_features, best_feature_values
    
    def generate_rule(self, inputs, labels):
        while len(labels.unique()) > 1:
            conditions = []
            str_conditions_list = []
            best_features, best_feature_values = self.best_condition_combination(inputs, labels)
            for best_feature, best_feature_value in zip(best_features, best_feature_values):
                conditions.append(inputs[best_feature] == best_feature_value)
                str_conditions_list.append(f"{best_feature} = {best_feature_value}")
            str_conditions = " & ".join(str_conditions_list)
            self.output_string.append(f"if {str_conditions} then {self.labelname} = {self.target_class}")
            combined_conditions = reduce(lambda x, y: x & y, conditions)
            inputs = inputs[~combined_conditions]
            labels = labels[~combined_conditions]
            if len(labels.unique()) <= 1:
                break
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
            Fit the model to the data
        """
        """
            使模型拟合数据
        """

        '''Parameter:
            X, corresponds to the training inputs;
            y, corresponds to the training labels.
        '''
        '''参数:
            X, 对应训练输入;
            y, 对应训练标签。
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("Labels must be a pandas Series.")
        if len(X) != len(y):
            raise RuntimeError("Length of inputs and labels must match.")

        # Since we might have to generate rules for different class,
        # we need to record the original data.
        # 由于我们可能想要为不同的类别生成规则,
        # 我们需要记录原始数据。
        self.original_inputs = X
        self.original_labels = y
        self.labelname = y.name
        
    def generate(self, target_class):
        """
        """
        """
        """

        '''
        '''
        '''
        '''
        
        # Reject the unexpected inputs / 拒绝意外输入
        if target_class not in self.original_labels.unique().tolist():
            raise ValueError("The target class is not present in labels.")
        self.target_class = target_class
        self.output_string = []
        self.temporary_inputs = self.original_inputs
        self.temporary_labels = self.original_labels
        self.generate_rule(self.temporary_inputs, self.temporary_labels)
        return self.output_string
    
    def __str__(self):
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        output_string = ";\n".join(self.output_string)
        return f"For {self.labelname} = {self.target_class}, PRISM generates the following rule(s):\n{output_string}."

class SimpleLinearRegression(object):
    """
        Implement a simple linear regression
    """
    """
        实现一个一元线性回归
    """

    def fit(self, x, y):
        n = len(x)

        self.b_1 = (np.sum(x * y) - np.sum(x) * np.sum(y) / n) / (np.sum(np.power(x, 2)) - np.power(np.sum(x), 2) / n)
        self.b_0 = np.mean(y) - self.b_1 * np.mean(x)

    def predict(self, x):
        return self.b_0 + self.b_1 * x
    
    def R_square(self, x, y):
        y_hat = self.predict(x)
        y_bar = np.mean(y)
        SSR = np.sum(np.power(y_hat - y_bar, 2))
        SST = np.sum(np.power(y - y_bar, 2))
        return SSR / SST
    
    def MAE(self, x, y):
        """
            Compute the Mean Absolute Error
        """
        n = len(x)
        y_hat = self.predict(x)
        return 1 / n * np.sum(np.abs(y_hat - y))
    
    def MSE(self, x, y):
        """
            Compute the Mean Squared Error
        """
        n = len(x)
        y_hat = self.predict(x)
        return 1 / n * np.sum(np.power(y_hat - y, 2))
    
    def RMSE(self, x, y):
        """
            Compute the Root Mean Squared Error
        """
        return np.sqrt(self.MSE(x, y))
    
class LinearRegression(object):
    """
        Implement a linear regression, suitable for simple and multiple linear regression
    """
    """
        实现一个线性回归, 适用于一元和多元线性回归
    """

    def fit(self, X, y):
        n = len(y)
        X = np.column_stack((np.ones(n), X))  # Add a column of ones for the intercept term / 为截距项增加一列1

        # Calculate the coefficient vector using the normal equation
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        n = len(X)
        X = np.column_stack((np.ones(n), X))  # Add a column of ones for the intercept term / 为截距项增加一列1
        return X.dot(self.coefficients[1:]) + self.coefficients[0]

class NaiveBayes(object):
    def probability_hypothesis(self, y_train):
        self.P_hypothesis = []
        total_num = len(y_train)
        for category in self.categories:
            self.P_hypothesis.append(len(y_train[y_train == category])/total_num)
        return self.P_hypothesis
    
    def probability_evidence(self, featureSeries, condition):
        if pd.api.types.is_numeric_dtype(featureSeries):
            condition = float(condition)
            mean = featureSeries.mean()
            std = featureSeries.std()
            return norm.pdf(condition, mean, std)
        else:
            return len(featureSeries[featureSeries == condition])/ len(featureSeries)
    
    def fit(self, X_train, y_train):
        self.categories = y_train.unique().tolist()
        self.probability_hypothesis(y_train)
        self.train_inputs = X_train
        self.train_labels = y_train
        self.label_name = y_train.name
        
    def predict(self, X_test):
        self.test_inputs = X_test
        features = X_test.columns
        self.predictions = []
        for index, data in X_test.iterrows():
            P_highest = 0
            prediction = None
            for i, category in enumerate(self.categories):
                P_evidence = self.P_hypothesis[i]
                category_inputs = self.train_inputs[self.train_labels == category]
                for j, feature in enumerate(features):
                    featureSeries = category_inputs[feature]
                    P_Ej_category = self.probability_evidence(featureSeries, data[j])
                    P_evidence *= P_Ej_category
                if P_evidence > P_highest:
                    P_highest = P_evidence
                    prediction = category
            self.predictions.append(prediction)
        return pd.Series(self.predictions)
    
    def __str__(self):
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """
        string_ouputs = [f"Naive Bayes predicts {self.label_name} = {prediction}." for prediction in self.predictions]
        return "\n".join(string_ouputs)
        
class DecisionTreeRootSelection(object):
    """
        Implement 
    """
    """"""
    def entropy(self, mylist):
        """
            Compute the entropy based on a proportion list of exmaples.
        """
        
        '''Parameter:
            mylist, .'''
        
        # Convert the input into a list object
        myarray = np.array(mylist)
        if 0 in mylist:
            return 0
        # 
        
        myarray = myarray / np.sum(myarray)
        # Formula 
        entropy = np.sum(- myarray * np.log2(myarray))
        return entropy
    
    def information_gain(self, entropy):
        # 
        return self.general_entropy - entropy
        
    def dataset_entropy(self, y):
        counts = []
        self.categories = y.unique().tolist()
        for category in self.categories:
            num_category = len(y[y == category])
            counts.append(num_category)
        self.general_entropy = self.entropy(counts)
        return self.general_entropy
    
    def feature_entropy(self, frequency, distribution):
        frequency_array = np.array(frequency)
        probability_array = frequency_array / np.sum(frequency_array)
        entropy_array = np.array([self.entropy(dist) for dist in distribution])
        feature_entropy = np.sum(probability_array * entropy_array)
        return feature_entropy
    
    def distribution(self, mySeries):
        counts = []
        total_num = len(mySeries)
        for category in self.categories:
            num_category = len(mySeries[mySeries == category])
            counts.append(num_category)
        return total_num, counts

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:

        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("Labels must be a pandas Series.")
        best_feature = None
        best_information_gain = 0
        self.dataset_entropy(y)
        self.features = X.columns
        self.categories = y.unique().tolist()
        for ftr in self.features:
            frequency = []
            distribution = []
            feature = X[ftr]
            feature_uniq = feature.unique()
            for uniq in feature_uniq:
                con_category = y[feature == uniq]
                total, dist = self.distribution(con_category)
                frequency.append(total)
                distribution.append(dist)
            feature_entropy = self.feature_entropy(frequency, distribution)
            information_gain = self.information_gain(feature_entropy)
            if information_gain >= best_information_gain:
                best_information_gain = information_gain
                best_feature = ftr
        self.selected_feature = best_feature
                
    def select(self):
        return self.selected_feature
    
    def __str__(self):
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """
        return f"As {self.selected_feature} has the highest information gain, it will be selected as the root of the tree."

class Perceptron(object):
    """
        Implement Perceptron
    """
    """
        实现感知机
    """

    def __init__(self, n_in, default=True) -> None:
        """
            Initialize the object Perceptron
        """
        """
            初始化对象Perceptron
        """

        '''Parameters:
            n_in, corresponds to the number of input features;
            default, corresponds to whether the parameters will be assigned default values.
        '''
        '''参数:
            n_in, 对应输入特征的数量;
            default, 对应参数是否赋默认值。
        '''

        self.n_in = n_in
        if default:
            # Initialize the weights and the bias using default values (all zeros)
            self.weights = np.zeros(n_in)
            self.bias = 0
            
    def initialize_weights(self, weights, bias):
        """
            weights,
            bias, 
        """
        """
        """

        # Check the data type of the inputs / 检查输入的数据类型
        if not isinstance(weights, np.ndarray):
            raise TypeError("The input of the weights must be a numpy ndarray.")
        if not isinstance(bias, (int, float)):
            raise TypeError("The input of the bias must be an integer or a float.")
        self.weights = weights
        self.bias = bias
        
    def step(self, x) -> int:
        """
            Define the step function
        """
        """
            定义阶梯函数
        """

        '''Parameter:
            x, corresponds to 
        '''
        if x >= 0:
            return 1
        else:
            return 0
    
    def fit(self, X, y, epochs=1):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Inputs must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("Labels must be a pandas Series.")
        self.epochs = epochs
        self.inputs = X.to_numpy()
        self.target = y.to_numpy()
        for ep in range(epochs):
            for i, t in zip(self.inputs, self.target):
                a = self.step(self.weights.T @ i + self.bias)
                e = t - a
                self.weights += e * i
                self.bias += e
    
    def __str__(self):
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        ep = self.epochs
        w = self.weights.tolist()
        b = self.bias
        return f"After {ep} epoch(s), weight vector and bias: w = {w}, b = {b}."
    
class Kmeans(object):
    """
        Implement Kmeans
    """
    """
        实现Kmeans
    """

    def __init__(self, centroids: list) -> None:
        """
            Initialize the object Kmeans
        """
        """
            初始化对象Kmeans
        """
        """Parameter:
            centroids, 
        """

        self.centroids = centroids
        self.k = len(centroids)
        self.clusters = {}
        for i in range(self.k):
            self.clusters[i] = []
        
    def fit(self, matrix: pd.DataFrame) -> None:
        '''Parameter:
            matrix, correponds to the distance matrix.
        '''
        '''参数:
            matrix, 对应距离矩阵。
        '''
        if not isinstance(matrix, pd.DataFrame):
            raise TypeError("Distance matrix must be a pandas DataFrame.")
        self.points = matrix.columns.tolist()
        distance = matrix.loc[self.centroids, :].to_numpy().T
        for i, dist in enumerate(distance):
            self.clusters[np.argmin(dist)].append(self.points[i])
    
    def __str__(self) -> str:
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        output_list = []
        for cluster in self.clusters.values():
            output_list.append(str(set(cluster)))
        output_string = ", ".join(output_list)
        return f"After the first epoch, the clusters are: " + output_string + "."

class HierarchicalClustering(object):
    def __init__(self) -> None:
        """
            Intialize the object HierarchicalClustering
        """
        """
            初始化对象HierarchicalClustering
        """
        # Record the agglomerative procedure
        self.agglomerative = []
        
    def min_value_index(self, matrix: pd.DataFrame) -> tuple:
        """
            Locate the row and column index of the minimum value
        """
        """
            定位最小值的行索引和列索引
        """

        '''Parameter:
            matrix, correponds to the distance matrix.
        '''
        '''参数:
            matrix, 对应距离矩阵。
        '''

        num_cluster = len(matrix.columns)
        min_value = float('inf')
        min_value_row = None
        min_value_column = None
        for i in range(0, num_cluster-1):
            for j in range(i + 1, num_cluster):
                matrix_element = matrix.iloc[i, j]
                if matrix_element < min_value:
                    min_value = matrix_element
                    min_value_row = i
                    min_value_column = j
        return min_value_row, min_value_column
    
    def update(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
            Operations during the iterations
        """
        """
            迭代过程中的操作
        """

        '''Parameter:
            matrix, correponds to the distance matrix.
        '''
        '''参数:
            matrix, 对应距离矩阵。
        '''

        # Merge the two closest clusters / 合并两个最近的簇
        i, j = self.min_value_index(matrix)
        cluster_i, cluster_j = matrix.index[i], matrix.columns[j]
        cluster_new = cluster_i + cluster_j
        self.agglomerative.append(set(cluster_new))
        # Update the proximity matrix / 更新接近度矩阵
        # Update the distance between clusters after merged / 更新合并后簇间距离
        distance_i = matrix[cluster_i]
        distance_j = matrix[cluster_j]
        distance_new = pd.concat([distance_i, distance_j], axis=1).min(axis=1)
        # Delete the merged cluster from the index / 从索引中删除已合并的簇
        index_list = matrix.index.tolist()
        index_list[i] = cluster_new
        del index_list[j]
        # Delete the merged cluster from the distance matrix i.e. its corresponding row and column / 从距离矩阵中删除已合并的簇, 即其行和列
        matrix[cluster_i] = distance_new
        matrix.loc[cluster_i] = distance_new
        matrix = matrix.drop(cluster_j, axis=0)
        matrix = matrix.drop(cluster_j, axis=1)
        matrix = matrix.set_axis(index_list, axis=0)
        matrix = matrix.set_axis(index_list, axis=1)
        return matrix
     
    def fit(self, matrix: pd.DataFrame) -> None:
        """
            Fit the model to the distance matrix
        """
        """
            使模型拟合距离矩阵
        """

        '''Parameter:
            matrix, correponds to the distance matrix.
        '''
        '''参数:
            matrix, 对应距离矩阵。
        '''

        if not isinstance(matrix, pd.DataFrame):
            raise TypeError("Distance matrix must be a pandas Series.")
        if not matrix.equals(matrix.T):
            raise ValueError("Distance matrix must be symmetric.")
        # Iterate until only a single cluster remains
        while len(matrix) > 1:
            matrix = self.update(matrix)
    
    def __str__(self) -> str:
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        output_list = [f"After step {it+1}, the latest merged cluster is: {cluster_result}" for it, cluster_result in enumerate(self.agglomerative)]
        output_string = ";\n".join(output_list)
        return output_string + "."
    
class DBSCAN(object):
    def __init__(self, eps, minpts):
        self.eps = eps
        self.minpts = minpts
    
    def label_core_points(self):
        matrix = self.input_matrix.to_numpy()
        num_neighbors = np.sum(matrix <= self.eps, axis=0)
        self.cores = self.points[num_neighbors >= self.minpts]
        self.cores = self.cores.tolist()
    
    def label_border_points(self):
        self.borders = []
        self.core_neighbors = {}
        for core in self.cores:
            self.core_neighbors[core] = []
            for point in self.points:
                distance = self.input_matrix.loc[core, point]
                if distance <= self.eps:
                    self.borders.append(point)
                    self.core_neighbors[core].append(point)
    
    def label_noise_points(self):
        self.noises = []
        for point in self.points:
            if point not in self.cores and point not in self.borders:
                self.noises.append(point)
        
    def fit(self, matrix) -> None:

        '''Parameter:
            matrix, correponds to the distance matrix.
        '''
        '''参数:
            matrix, 对应距离矩阵。
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        self.clusters = []
        self.input_matrix = matrix
        self.points = matrix.columns
        self.label_core_points()
        self.label_border_points()
        self.label_noise_points()
        for core, cluster in self.core_neighbors.items():
            potential_cluster = set(cluster)
            if potential_cluster not in self.clusters:
                self.clusters.append(potential_cluster)
    
    def __str__(self):
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        output_list = [f"K{k+1} = {cluster}" for k, cluster in enumerate(self.clusters)]
        output_string = ", ".join(output_list)
        return f"Final clustering: {output_string}."

class ClusteringEvaluator(object):
    """
        Implement a clustering evaluator, 
        which can evaluate the clustering results using correlation or sihouette coefficient
    """
    """
        实现一个聚类评价器,
        可以用相关性或者轮廓系数评价聚类结果
    """

    def __init__(self, method: str) -> None:
        """
            Initialize the object ClusteringEvaluator
        """
        """
            初始化对象ClusteringEvaluator
        """

        '''Parameter:
            method, corresponds to the method specified to evaluate the clusetering results; Options: 'correlation', 'sihouette_coefficient'.
        '''
        '''参数:
            method, 对应明确的方法以评估聚类结果; 选项: 'correlation', 'sihouette_coefficient'。
        '''

        if method not in {'correlation', 'sihouette_coefficient'}:
            raise ValueError("Method {method} does not exist.")
        self.method = method
    
    def average_distance(self, mySeries: pd.Series) -> float:
        """
            Compute the average distance between points
        """
        """
            计算点间平均距离
        """
        
        '''Parameter:
            mySeries, corresponds to a pandas Series, which contains distances between a point and other points
        '''
        '''参数:
            mySeries, 对应一个pandas Series, 包含了一个点和其他点间的距离
        '''

        newSeries = mySeries.copy()
        newSeries = newSeries[mySeries != 0]
        return newSeries.mean()
        
    def fit(self, cluster_labels: pd.Series, 
            distance_matrix: pd.DataFrame = None, 
            similarity_matrix_distance: pd.DataFrame = None) -> None:
        """
            Fit the ClusteringEvaluator to the clustering results and distance matrix, 
            or similarity matrix derived from distance matrix.
            (It depends on the evaluation method we specify.)
        """
        """
            使聚类评价器拟合聚类结果和距离矩阵,
            或者距离矩阵衍生的相似性矩阵。
            (它将取决于我们明确的评价方法。)
        """

        '''Parameters:
            cluster_labels, corresponds to the cluster results.
            distance_matrix, corresponds to the distance matrix; Default: None.
            similarity_matrix_distance, corresponds to the similarity matrix derived from the distance matrix; Default: None.
        '''
        '''参数:
            cluster_labels,
            distance_matrix,
            similarity_matrix_distance, 
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(cluster_labels, pd.Series):
            raise TypeError("Cluster labels must be a pandas Series.")
        if self.method == 'correlation':
            if distance_matrix is None:
                if similarity_matrix_distance is None:
                    raise ValueError("At least one of distance matrix and similarity matrix must not be None.")
                if not isinstance(similarity_matrix_distance, pd.DataFrame):
                    raise TypeError("Similarity matrix must be a pandas DataFrame.")
            if similarity_matrix_distance is None:
                if distance_matrix is None:
                    raise ValueError("At least one of distance matrix and similarity matrix must not be None.")
                if not isinstance(distance_matrix, pd.DataFrame):
                    raise TypeError("Distance matrix must be a pandas DataFrame.")
                distance_matrix = distance_matrix.to_numpy()
                d_max = np.max(distance_matrix)
                d_min = np.min(distance_matrix)
                similarity_matrix_distance = 1 - (distance_matrix - d_min) / (d_max - d_min)
            num_points = len(similarity_matrix_distance)
            if not isinstance(similarity_matrix_distance, np.ndarray):
                similarity_matrix_distance = similarity_matrix_distance.to_numpy()
            similarity_matrix_results_list = []
            similarity_matrix_distance_list = []
            for i in range(0, num_points):
                for j in range(i, num_points):
                    similarity_matrix_results_list.append(int(cluster_labels[i] == cluster_labels[j]))
                    similarity_matrix_distance_list.append(similarity_matrix_distance[i][j])
            corr, p_value = pearsonr(similarity_matrix_distance_list, similarity_matrix_results_list)
            self.clustering_quality = corr
        elif self.method == 'sihouette_coefficient':
            if not isinstance(distance_matrix, pd.DataFrame):
                raise TypeError("Distance matrix must be a pandas DataFrame.")
            self.method_string = 'sihouette coefficient'
            clusters = cluster_labels.unique()
            cohesion = cluster_labels.copy()
            separation = cluster_labels.copy()
            for point in cohesion.index:
                current_matrix = distance_matrix.copy()
                cluster_label = cluster_labels[point]
                cohesion_matrix = current_matrix[cluster_labels == cluster_label]
                separation_matrix = current_matrix[cluster_labels != cluster_label]
                cohesion_Series = cohesion_matrix[point]
                separation_Series = separation_matrix[point]
                cohesion[point] = self.average_distance(cohesion_Series)
                separation[point] = self.average_distance(separation_Series)
            point_sihouette_coefficient = (separation - cohesion) / np.maximum(cohesion, separation)
            cluster_sihouette_coefficient = []
            for cluster in clusters:
                cluster_sihouette_coefficient.append(point_sihouette_coefficient[cluster_labels == cluster].mean())
            clustering_sihouette_coefficient = np.mean(cluster_sihouette_coefficient)
            self.clustering_quality = clustering_sihouette_coefficient
        
    def __str__(self) -> str:
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        if not hasattr(self, 'method_string'):
            self.method_string = self.method
        return f"The evaluation results of the clustering quality using {self.method_string} is {self.clustering_quality:.3f}."

class MarkovChain(object):
    """
        Implement a simple Markov Chain
    """
    """
        实现一个简单的马尔可夫链
    """

    def __init__(self, transition_matrix: pd.DataFrame, 
                 time_sequence: list =['yesterday', 'today', 'tomorrow', 'the day after tomorrow']) -> None:
        """
            Initialize the object MarkovChain
        """
        """
            初始化对象MarkovChain
        """
        
        '''Parameters:
            transition_matrix, corresponds to the transition probability matrix,
            time_sequence, corresponds to the four time steps. Default: ['yesterday', 'today', 'tomorrow', 'the day after tomorrow'].
        '''
        '''
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        if len(time_sequence) != 4:
            raise ValueError("Time sequence must have a length of 4.")
        if transition_matrix.index.tolist() != transition_matrix.columns.tolist():
            raise ValueError("The states in transition matrix must be consistent.")
        self.transition_matrix = transition_matrix
        self.time_sequence = time_sequence
        
    def fit(self, state_sequence: list) -> None:
        if len(state_sequence) != 4:
            raise ValueError("State sequence must have a length of 4.")
        self.state_sequence = state_sequence
        
    def state_after_next_state(self) -> float:
        """
            Compute the probability of a specific state after the next state given a current state and a next state
        """
        """
            计算给定当前状态下的下一特定状态的概率
        """

        # Reject the unexpected inputs / 拒绝意外输入
        if None in self.state_sequence[1:]:
            raise ValueError("Sequence contains None after the first element.")
        self.state_sequence[0] = None
        current_state = self.state_sequence[1]
        next_state = self.state_sequence[2]
        state_after_next_state = self.state_sequence[3]
        self.probability = self.transition_matrix.loc[current_state, next_state] * self.transition_matrix.loc[next_state, state_after_next_state]
        return self.probability
        
    def next_state(self) -> float:
        """
            Compute the probability of a specific next state given a current state
        """
        """
            计算给定当前状态下的下一特定状态的概率
        """

        # Reject the unexpected inputs / 拒绝意外输入
        if None in self.state_sequence[:-1]:
            raise ValueError("Sequence contains None before the last element.")
        self.state_sequence[3] = None
        current_state = self.state_sequence[1]
        next_state = self.state_sequence[2]
        self.probability = self.transition_matrix.loc[current_state, next_state]
        return self.probability
    
    def __str__(self) -> str:
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        if self.state_sequence[0] == None:
            output_string = f"{self.state_sequence[2]} {self.time_sequence[2]} and {self.state_sequence[3]} {self.time_sequence[3]}"
        elif self.state_sequence[3] == None:
            output_string = f"{self.state_sequence[2]} {self.time_sequence[2]}"
        return f"Given that {self.time_sequence[1]} is {self.state_sequence[1]}, the probability that it will be {output_string} is {self.probability:.2f}."
    
class HiddenMarkovModel(object):
    """
        Implement Hidden Markov Model"""
    """
        实现隐马尔可夫模型"""

    """Examples / 示例:
    Suppose initial_probability, transition, emission have been already defined 
    (initial_probability: intial probabilities, transition: transition matrix, emission: emission matrix).
    假设initial_probability, transition, emission均已被定义
    (initial_probability: 初始概率, transition: 转移矩阵, emission: 发射矩阵)

    ** 
    # Example / 示例 1:
    # Initialize an HiddenMarkovModel object / 实例化一个OneRule对象
    >>> observations = ['No Umbrella', 'Umbrella']
    >>> hmm = HiddenMarkovModel(initial_probability, transition, emission)
    >>> hmm.fit(observations)
    >>> print(hmm)


    """
    def __init__(self, initial_probability: pd.Series, transition: pd.DataFrame, emission: pd.DataFrame) -> None:
        """
            In our implementation, we do not consider HMM Problem 3 (also called Learning problem). 
            Therefore, we initialize a Hidden Markov Model using the given initial probabilities, transition matrix, and emission matrix.
        """
        """
            在我们的实现中, 我们不考虑HMM问题3(也称为学习问题)。
            因此, 我们将用给定的初始概率, 转移矩阵, 发射矩阵来初始化一个隐马尔可夫模型。
        """

        '''Parameters:
            initial_probability, corresponds to the initial probabilities of states A_0;
            transition, corresponds to the transition probability matrix A;
            emission, corresponds to the emission probability matrix E.
        '''
        '''参数:
            initial_probability, 对应状态A_0的初始概率;
            transition, 对应转移矩阵A;
            emission, 对应发射矩阵E。
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(initial_probability, pd.Series):
            raise TypeError("The initial probabilities of states must be a pandas Series.")
        if not isinstance(transition, pd.DataFrame):
            raise TypeError("The transition probability matrix A must be a pandas DataFrame.")
        if not isinstance(emission, pd.DataFrame):
            raise TypeError("The emission probability matrix E must be a pandas DataFrame.")
        if transition.index.tolist() != transition.columns.tolist():
            raise ValueError("The states in transition probability matrix must be consistent.")
        if  transition.index.tolist() != initial_probability.index.tolist():
            raise ValueError("The states of transition probability matrix must match those of initial probabilities.")
        if transition.index.tolist() != emission.index.tolist():
            raise ValueError("The states of transition probability matrix must match those of emission matrix.")
        if not np.isclose(initial_probability.sum(), 1, rtol=1e-6): # To avoid the precision problem
            raise ValueError("The sum of initial probability must be 1.")
        for index, data in transition.iterrows():
            if not np.isclose(data.sum(), 1, rtol=1e-6):
                raise ValueError("The sum of transition probability matrix must be 1.")
        for index, data in emission.iterrows():
            if not np.isclose(data.sum(), 1, rtol=1e-6):
                raise ValueError("The sum of emission probability matrix must be 1.")
        # Intialize the object HiddenMarkovModel
        self.initial_probability = initial_probability
        self.transition = transition
        self.emission = emission
        self.hidden_states = transition.columns.tolist()
    
    def forward(self) -> None:
        """
            Solve HMM problem 1 (also called Evaluation problem).
            i.e. Given an observation sequence, what is the probability of this sequence?
        """
        """
            这种方法将解决HMM问题1 (也称为评估问题)。
            即, 给定一个观测序列, 该序列的概率是多少？
        """

        # Implement the forward algorithm based on matrix operations / 基于矩阵运算实现前向算法
        self.forward_probability = []
        for index, observation in enumerate(self.observations):
            # Initialization / 初始化: f_k(1) = A_0(k)e_k(x_1)
            if index == 0:
                self.forward_probability.append(self.emission[observation] * self.initial_probability)
            # Iteration / 迭代: f_k(i) = e_k(x_i) \Sigma_j f_j(i - 1)a_{jk}
            # Compute forward probability of state k at time step i / 计算i时刻k状态的前向概率
            else:
                forward_probability_new = self.emission[observation] * (self.forward_probability[-1] @ self.transition)
                self.forward_probability.append(forward_probability_new)
        # Termination / 终止:  P(X) = \Sigma_k f_k(m)
        self.probability = self.forward_probability[-1].sum()
        
    def viterbi(self) -> None:
        """
            Solve HMM problem 2 (also called Decoding problem).
            i.e. Given an observation sequence, what is the most likely sequence of hidden states?
        """
        """
            这个方法将解决HMM问题2(也称为解码问题)。
            即, 给定一个观测序列, 最可能的隐藏状态序列是什么？
        """
        
        # Implement the viterbi algorithm based on matrix operations / 基于矩阵运算实现维比特算法
        self.viterbi_score = []
        self.back_pointers = []
        self.hidden_state_sequence = []
        for index, observation in enumerate(self.observations):
            # Initialization / 初始化: V_k(1) = A_0(k)e_k(x_1)
            if index == 0:
                self.viterbi_score.append(self.emission[observation] * self.initial_probability)
            # Iteration / 迭代
            #
            else:
                intermediate = self.viterbi_score[-1] * self.transition.T
                # Compute Viterbi score of state k at time i / 计算状态k在i时刻的维比特得分
                # V_k(i) = e_k(x_i)max_j(V_j)a_{jk}
                viterbi_score_new = self.emission[observation] * intermediate.max(axis=1)
                # Back-pointer of the state k at time i / 状态k在i时刻的回指指针
                # Ptr_k(i) = argmax_j V_j(i - 1) a_{jk}
                back_pointer = intermediate.idxmax(axis=1).tolist()
                self.viterbi_score.append(viterbi_score_new)
                self.back_pointers.append(back_pointer)
        # Termination and trace-back / 终止和回溯
        best_last_state = self.viterbi_score[-1].idxmax()
        self.hidden_state_sequence.append(best_last_state)
        for back_pointer in reversed(self.back_pointers):
            best_last_state = back_pointer[self.hidden_states.index(best_last_state)]
            self.hidden_state_sequence.insert(0, best_last_state)
    
    def fit(self, observations: list) -> None:
        """
            Fit the model to the observation sequence
        """
        """
            使模型拟合观测序列
        """

        '''Parameter:
            observations, corresponds to a list of observation sequence.
        '''
        '''参数:
            observations, 对应一个观测序列的列表。
        '''

        # Reject the unexpected inputs / 拒绝意外输入
        if not isinstance(observations, (list, np.ndarray, pd.Series)):
            raise TypeError("The observation sequence must be a list, numpy ndarray or pandas Series.")
        # If there is any observation in the observation sequence does not match the existing observations in the emission probability matrix, this error will be raised.
        # 如果有任何在观测序列中的观测与在发射概率矩阵中已知观测不匹配, 这个错误将被抛出。
        if not set(observations).issubset(set(self.emission.columns.tolist())):
            raise ValueError("Some of observations do not match the emission probability matrix.")
        # If the parameter 'observations' passed is an empty list, this error will be raised.
        # 如果传入参数'observations'是一个空列表, 这个错误将被抛。
        if len(observations) == 0:
            raise IndexError("The observation sequence must have at least one observation.")
        self.observations = observations
        # Apply the Forward algorithm / 应用前向算法
        self.forward()
        # Apply the Viterbi algorithm / 应用维比特算法
        self.viterbi()
    
    def __str__(self) -> str:
        """
            Defines the string representation
        """
        """
            定义字符串表示形式
        """

        observation_sequence = ", ".join(self.observations)
        most_likely_sequence = ", ".join(self.hidden_state_sequence)
        observation_result = f"The probability of the observation sequence {observation_sequence} is {self.probability:.3f}"
        most_likely_result = f"The most likely sequence of hidden states is {most_likely_sequence}"
        return f"Conclusion: a) {observation_result}, b) {most_likely_result}."