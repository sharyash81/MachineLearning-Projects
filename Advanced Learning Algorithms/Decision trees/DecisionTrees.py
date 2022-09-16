import numpy as np
import matplotlib.pyplot as plt
from utils import *

def compute_entropy(y):
    if len(y) == 0:
        return 0
    entropy = 0 
    p1 = len(y[y==1]) / len(y)
    if p1 == 0 or p1 == 1:
        return 0
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    return entropy

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    
    for index in node_indices:
        if X[index][feature] == 1:
            left_indices.append(index)
        else:
            right_indices.append(index)

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    information_gain = 0
    w_left = len(left_indices) / len(node_indices)
    w_right = 1 - w_left
    information_gain = compute_entropy(y_node) - ( w_left*compute_entropy(y_left) + w_right*compute_entropy(y_right) )
    
    return information_gain

def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = -1
    max_information_gain = 0

    for i in range(num_features):
        new_inf = compute_information_gain(X,y,node_indices,i)
        if new_inf > max_information_gain:
            max_information_gain = new_inf
            best_feature = i

    return best_feature
    

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    global tree
    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)

    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

if __name__ == "__main__":

    X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
    y_train = np.array([1,1,0,0,1,0,0,1,1,0])
    root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    global tree
    tree = []
    build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
    generate_tree_viz(root_indices, y_train, tree)




