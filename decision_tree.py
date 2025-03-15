import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time

class Node:
    """Node class for the Decision Tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Index of the feature to split on
        self.threshold = threshold    # Threshold value for the feature
        self.left = left              # Left subtree (feature <= threshold)
        self.right = right            # Right subtree (feature > threshold)
        self.value = value            # Predicted class (leaf node)
    
    def is_leaf(self):
        """Check if the node is a leaf node"""
        return self.value is not None
    
class DecisionTree:
    """Decision Tree Classifier implemented from scratch"""
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity=1e-7):
        self.max_depth = max_depth                # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.min_impurity = min_impurity          # Minimum impurity required to split a node
        self.root = None                          # Root node of the tree
        self.feature_importances_ = None          # Feature importances
        
    def _calculate_entropy(self, y):
        """Calculate entropy of a dataset"""
        class_counts = Counter(y)
        entropy = 0
        for count in class_counts.values():
            probability = count / len(y)
            entropy -= probability * np.log2(probability)
        return entropy
    
    def _calculate_information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split"""
        # Calculate parent entropy
        parent_entropy = self._calculate_entropy(y)
        
        # Calculate weighted entropy of children
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Weighted entropy
        child_entropy = (n_left / n) * self._calculate_entropy(y_left) + \
                        (n_right / n) * self._calculate_entropy(y_right)
        
        # Information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _best_split(self, X, y):
        """Find the best feature and threshold for splitting the data"""
        n_samples, n_features = X.shape
        
        # If not enough samples, don't split
        if n_samples < self.min_samples_split:
            return None, None
        
        # Find the best feature and threshold
        best_info_gain = -float("inf")
        best_feature, best_threshold = None, None
        
        for feature_idx in range(n_features):
            # Get unique values in the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # For each possible threshold
            for threshold in thresholds:
                # Split the data
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx
                
                # If split is too small, skip
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                # Calculate information gain
                info_gain = self._calculate_information_gain(
                    y, y[left_idx], y[right_idx]
                )
                
                # Update if better split is found
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # If minimum impurity is not met
        if best_info_gain < self.min_impurity:
            return None, None
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # If max_depth is reached or only one class or not enough samples, create leaf node
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # If no valid split, create leaf node
        if best_feature is None:
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class)
        
        # Split the data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        # Update feature importances
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(n_features)
        
        # Return internal node
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        """Build the decision tree"""
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._build_tree(X, y)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)
        
        return self
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample"""
        # If leaf node, return the value
        if node.is_leaf():
            return node.value
        
        # Determine which branch to follow
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict classes for samples in X"""
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array(predictions)
    
    def visualize(self, feature_names=None, class_names=None, max_depth=None):
        """Visualize the decision tree"""
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(100)]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(100)]
            
        def _get_tree_info(node, depth=0, parent=None, is_left=None):
            if max_depth is not None and depth > max_depth:
                return []
            
            if node.is_leaf():
                return [{
                    'type': 'leaf',
                    'depth': depth,
                    'parent': parent,
                    'is_left': is_left,
                    'value': class_names[node.value] if node.value < len(class_names) else str(node.value)
                }]
            
            info = [{
                'type': 'node',
                'depth': depth,
                'parent': parent,
                'is_left': is_left,
                'feature': feature_names[node.feature] if node.feature < len(feature_names) else f"Feature {node.feature}",
                'threshold': node.threshold
            }]
            
            info.extend(_get_tree_info(node.left, depth + 1, len(info) - 1, True))
            info.extend(_get_tree_info(node.right, depth + 1, len(info) - 1, False))
            
            return info
        
        tree_info = _get_tree_info(self.root)
        
        # Calculate positions
        max_depth_tree = max(info['depth'] for info in tree_info) + 1
        width, height = max_depth_tree * 3, max_depth_tree * 1.5
        
        fig, ax = plt.figure(figsize=(width, height)), plt.gca()
        
        # Positions for each depth
        ys = {depth: height - (depth + 0.5) * height / max_depth_tree for depth in range(max_depth_tree)}
        
        # Assign x positions
        def assign_x_positions(tree_info):
            # Count nodes at each depth
            depth_counts = Counter(info['depth'] for info in tree_info)
            depth_positions = {depth: {} for depth in range(max_depth_tree)}
            
            # Calculate position for the root
            root_idx = next(i for i, info in enumerate(tree_info) if info['depth'] == 0)
            tree_info[root_idx]['x_pos'] = width / 2
            
            # Recursively assign positions based on parent
            for info in sorted(tree_info, key=lambda x: x['depth']):
                depth, is_left, parent_idx = info['depth'], info['is_left'], info['parent']
                
                if depth == 0:  # Root already assigned
                    continue
                
                parent_x = tree_info[parent_idx]['x_pos']
                
                # Calculate horizontal spacing at this depth
                h_space = width / (depth_counts[depth] + 1)
                
                if parent_idx not in depth_positions[depth-1]:
                    depth_positions[depth-1][parent_idx] = {'left_idx': 0, 'right_idx': 0}
                
                if is_left:
                    pos_idx = depth_positions[depth-1][parent_idx]['left_idx'] + 1
                    depth_positions[depth-1][parent_idx]['left_idx'] += 1
                    offset = -pos_idx * h_space / 2
                else:
                    pos_idx = depth_positions[depth-1][parent_idx]['right_idx'] + 1
                    depth_positions[depth-1][parent_idx]['right_idx'] += 1
                    offset = pos_idx * h_space / 2
                
                info['x_pos'] = parent_x + offset
        
        assign_x_positions(tree_info)
        
        # Draw nodes and connections
        for info in tree_info:
            x, y = info['x_pos'], ys[info['depth']]
            
            if info['type'] == 'leaf':
                # Draw leaf node
                circle = plt.Circle((x, y), 0.1, color='lightblue', fill=True)
                ax.add_patch(circle)
                plt.text(x, y, info['value'], ha='center', va='center', fontsize=10)
            else:
                # Draw decision node
                circle = plt.Circle((x, y), 0.1, color='lightgreen', fill=True)
                ax.add_patch(circle)
                plt.text(x, y + 0.2, f"{info['feature']} <= {info['threshold']:.3f}", ha='center', va='center', fontsize=10)
            
            # Draw connection to parent
            if info['depth'] > 0:
                parent_info = tree_info[info['parent']]
                parent_x, parent_y = parent_info['x_pos'], ys[parent_info['depth']]
                plt.plot([parent_x, x], [parent_y, y], 'k-')
        
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        return fig

class DecisionTreeEvaluator:
    """Class for evaluating and testing the decision tree classifier"""
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None):
        """Split the data into training and testing sets"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Shuffle the indices
        indices = np.random.permutation(len(X))
        
        # Calculate split point
        split = int(len(X) * (1 - test_size))
        
        # Split the data
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        """Calculate accuracy score"""
        return np.sum(y_true == y_pred) / len(y_true)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, normalize=False):
        """Calculate confusion matrix"""
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(classes)
        
        # Initialize confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # Fill confusion matrix
        for i in range(len(y_true)):
            true_class_idx = np.where(classes == y_true[i])[0][0]
            pred_class_idx = np.where(classes == y_pred[i])[0][0]
            cm[true_class_idx, pred_class_idx] += 1
        
        # Normalize if required
        if normalize:
            cm = cm.astype(float)
            row_sums = cm.sum(axis=1)
            cm = cm / row_sums[:, np.newaxis]
        
        return cm, classes
    
    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if np.any(cm < 1) else 'd'
        thresh = cm.max() / 2.
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return plt
    
    @staticmethod
    def generate_synthetic_data(n_samples=100, n_features=2, n_classes=2, random_state=None):
        """Generate synthetic data for testing the decision tree"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate random data
        X = np.random.randn(n_samples, n_features)
        
        # Generate random classes
        y = np.random.randint(0, n_classes, size=n_samples)
        
        return X, y
    
    @staticmethod
    def benchmark(model, X_train, y_train, X_test, y_test):
        """Benchmark the model training and prediction times"""
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = DecisionTreeEvaluator.accuracy_score(y_test, y_pred)
        
        return {
            'train_time': train_time,
            'predict_time': predict_time,
            'accuracy': accuracy
        }

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    evaluator = DecisionTreeEvaluator()
    X_train, X_test, y_train, y_test = evaluator.train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Evaluate
    accuracy = evaluator.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    cm, classes = evaluator.confusion_matrix(y_test, y_pred)
    evaluator.plot_confusion_matrix(cm, classes, title='Confusion Matrix')
    
    # Visualize the tree
    fig = tree.visualize(feature_names=iris.feature_names, 
                         class_names=iris.target_names)
    plt.show()
