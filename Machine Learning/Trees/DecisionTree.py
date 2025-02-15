import numpy as np

class DecisionTree():
    def __init__(self , type , max_depth=None , criterion='gini' , min_samples_split=2 , min_samples_leaf=1 , splitter='best' , metric=None ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.tree = None
        self.type = type
        self.metric = metric
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        number_of_samples , number_of_features = X.shape

        if len(y) < self.min_samples_split or len(np.unique(y)) == 1 or depth == self.max_depth:
            value = (np.mean(y) if len(y) > 0 else 0) if self.type == 'regression' else np.bincount(y).argmax()
            return Node(left=None , right=None , feature=None , threshold=None , value=value)
        
        features = np.random.choice(X.shape[1] , X.shape[1] , replace=False)
        feature , threshold , impurity = self._split(X, y , features)
        left_indices = X[:,feature] < threshold
        right_indices = X[:,feature] >= threshold
        left_tree = self._grow_tree(X[left_indices] , y[left_indices] , depth+1)
        right_tree = self._grow_tree(X[right_indices] , y[right_indices] , depth+1)
        return Node(left=left_tree , right=right_tree , feature=feature , threshold=threshold , value=None)
    
    def _split(self, X, y , features):
        if self.splitter == 'best':
            return self._best_splitter(X, y , features)
        else:
            return self._random_splitter(X, y , features)
        
    def _best_splitter(self, X, y , features):
        best_feature = None
        best_threshold = None
        best_impurity = np.inf
        for feature in features:
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                left_indices = X[:,feature] < threshold
                right_indices = X[:,feature] >= threshold
                left_y = y[left_indices]
                right_y = y[right_indices]
                impurity = self._impurity(left_y , right_y)
                if impurity < best_impurity:
                    best_feature = feature
                    best_threshold = threshold
                    best_impurity = impurity
        return best_feature , best_threshold , best_impurity
    
    def _random_splitter(self, X, y , features):
        feature = np.random.choice(features)
        threshold = np.random.choice(np.unique(X[:,feature]))
        impurity = self._impurity(y)
        return feature , threshold , impurity
        

    def _impurity(self, left_values , right_values):
        left_impurity = self.criterion(left_values) if self.type == 'classification' else self.criterion(left_values  , np.mean(left_values) if len(left_values) > 0 else 0)
        right_impurity = self.criterion(right_values) if self.type == 'classification' else self.criterion(right_values , np.mean(right_values) if len(right_values) > 0 else 0)

        n = len(left_values) + len(right_values)

        # Weighted sum of the impurities
        weighted_left_impurity = (len(left_values)/n)*left_impurity
        weighted_right_impurity = (len(right_values)/n)*right_impurity
        return weighted_left_impurity + weighted_right_impurity
    
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])
    
    def _predict(self, inputs):
        node = self.tree
        while node.value is None:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def score(self, X, y):
        predictions = self.predict(X)
        return self.metric(y , predictions)
    
    # Write a function to print the tree in a readable format
    def _show_tree(self, node, feature_names ,  depth=0):
        if node.value is not None:
            print(f"{'|   ' * depth}Leaf: {node.value}")
        else:
            print(f"{'|   ' * depth}Feature {feature_names[node.feature]} < {node.threshold}")
        
            new_depth = depth + 1
            self._show_tree(node.left, feature_names, depth=new_depth)
            self._show_tree(node.right,feature_names, depth=new_depth)

        

    def get_leaf_nodes(self):
        return self._get_leaf_nodes(self.tree)
    
    def _get_leaf_nodes(self, node):
        if node.is_leaf():
            return [node]
        left_nodes = self._get_leaf_nodes(node.left)
        right_nodes = self._get_leaf_nodes(node.right)
        return left_nodes + right_nodes
    
class Node():
    def __init__(self , left , right , feature , threshold , value):
        self.left = left
        self.right = right
        self.feature = feature  
        self.threshold = threshold
        self.value = value

    def is_leaf(self):
        return self.value is not None