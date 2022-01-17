#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def init():
    while True:
        # ask for .csv filename
        filename = input("Enter CSV filename (leave empty for diabetes.csv): ") or "diabetes.csv"
    
        if(filename != "diabetes.csv"):
            if(filename[-4:]) != ".csv":
                filename = filename + ".csv"
    
        # open dataframe (using provided filename)
        try:
            diabetes_df = pd.read_csv(filename)
        except:
            print("File not found, please try again.")
            continue
    
        target_feature = input("Enter target feature name (leave empty for 'Outcome'): ") or "Outcome"
    
        if(target_feature != "Outcome"):
            # check for target feature in dataframe
            if(target_feature not in diabetes_df.columns.array):
                print("Target feature not found, please try again.")
                continue
            
        max_depth = input("Enter maximum depth of DT (leave empty for 4): ") or 4
            
        if(not str(max_depth).isnumeric() or not 1 <= int(max_depth) <= 10):
            print("Invalid depth, please try again.")
            continue
        
        train_size = input("Enter percentage of training data (leave empty for 60%): ") or 60
            
        if(not str(train_size).isnumeric() or not 1 <= int(train_size) <= 99):
            print("Invalid training data percentage, please try again.")
            continue
        
        folds = input("Enter number of k-folds (leave empty for 5): ") or 5
        
        if(not str(folds).isnumeric() or not 1 <= int(folds) <= 10):
            print("Invalid number of folds, please try again.")
            continue
        
        print("Please wait...")
        
        ind_features = [x for x in diabetes_df.columns.array if x != target_feature]
        
        # input received, start computing
        training_df, testing_df = split_train_test(diabetes_df, train_size)
        
        trained_tree = build_tree(training_df, ind_features, max_depth)
        
        print_tree(trained_tree)
        
        cross_validate(diabetes_df, trained_tree, target_feature, folds)
        
        input("Program complete, press ENTER to close...")
        break 

class AttributeTest:
    def __init__(self, node_col, node_val):
        self.node_col = node_col
        self.node_val = node_val
    
    def split(self, dataframe):
        left_split = dataframe[dataframe[self.node_col] < self.node_val]
        left_split.reset_index(drop=True, inplace=True)
        
        right_split = dataframe[dataframe[self.node_col] >= self.node_val]
        right_split.reset_index(drop=True, inplace=True)
        
        return left_split, right_split
    
    def test_instance(self, instance):
        if(instance[self.node_col] >= self.node_val):
            return True
        else:
            return False
    
    def __repr__(self):
        return("Is " + self.node_col + " >= " + str(self.node_val) + "?")
    
class Leaf:
    def __init__(self, dataframe):
        percentages = {}
        for outcome in set(dataframe["Outcome"]):
            label = "Not Diabetic"
            
            if(outcome == 1):
                label = "Diabetic"
            
            percentages[label] = round(dataframe[dataframe["Outcome"] == outcome].shape[0] / dataframe.shape[0] * 100, 2)
            
        self.predictions = percentages
    
    def return_prediction(self):
        max_pred = max(self.predictions, key=lambda key: self.predictions[key])
        return max_pred
        
        
class Decision_Node:
    def __init__(self, attr_test, left, right):
        self.attr_test = attr_test
        self.left = left
        self.right = right
    
    def walk_node(self, instance):
        if(self.attr_test.test_instance(instance)):
            return self.right
        else:
            return self.left
        
def gini_index(dataframe):
    impurity = 1
    for outcome in set(dataframe["Outcome"]):
        outcome_prob = dataframe[dataframe["Outcome"] == outcome].shape[0] / dataframe.shape[0]
        impurity -= outcome_prob ** 2
    
    return impurity


def information_gain(left, right, current_gini):
    p_left = left.shape[0] / (left.shape[0] + right.shape[0])
    p_right = 1 - p_left
    return current_gini - (p_left * gini_index(left) + p_right * gini_index(right))


def find_best_split(dataframe, features):
    curr_gini = gini_index(dataframe)
    max_inf_gain = 0
    best_attr_test = None
    
    for feature in features:
        unique_vals = set(dataframe[feature])
        
        for val in unique_vals:
            attr_test = AttributeTest(feature, val)
            
            left, right = attr_test.split(dataframe)
            
            if(left.shape[0] == 0 or right.shape[0] == 0):
                continue
            
            inf_gain = information_gain(left, right, curr_gini)
            
            if(inf_gain >= max_inf_gain):
                max_inf_gain = inf_gain
                best_attr_test = attr_test
    
    return max_inf_gain, best_attr_test



def build_tree(dataframe, features, max_depth=4, depth=0):
    inf_gain, attr_test = find_best_split(dataframe, features)
    
    if(inf_gain == 0.0 or depth == max_depth):
        return Leaf(dataframe)
    
    left, right = attr_test.split(dataframe)
    
    left_tree = build_tree(left, features, max_depth, depth + 1)
    right_tree = build_tree(right, features, max_depth, depth + 1)
    
    return Decision_Node(attr_test, left_tree, right_tree)

def print_tree(node, spacing=''):
    if(isinstance(node, Leaf)):
        print(spacing + "Leaf predict: " + str(node.predictions))
    else:    
        print(spacing + str(node.attr_test))
        print(spacing + "--> False: ")
        print_tree(node.left, spacing=spacing+ '  ')
        print(spacing + "--> True: ")
        print_tree(node.right, spacing=spacing+ '  ')

def walk_tree(tree, instance):
    if(isinstance(tree, Leaf)):
        return tree.return_prediction()
    else:
        next_tree = tree.walk_node(instance)
        return walk_tree(next_tree, instance)
    
# metric calculation functions

def calculate_accuracy(true_pos, false_pos, true_neg, false_neg):
    total = true_pos + false_pos + true_neg + false_neg
    return(((true_pos + true_neg) * 100) / total)

def calculate_precision(true_pos, false_pos):
    total = false_pos + true_pos
    return ((true_pos * 100) / total)

def calculate_recall(true_pos, false_neg):
    total = true_pos + false_neg
    return (true_pos * 100) / total

def calculate_f_measure(true_pos, false_pos, true_neg, false_neg):
    precision = calculate_precision(true_pos, false_pos)
    recall = calculate_recall(true_pos, false_neg)
    f_measure = (2 * precision * recall) / (precision + recall)
    return f_measure

def measure_tree_performance(tree, testing_df, target_feature):
    y_actual = testing_df[target_feature]
    y_pred = []
    
    for instance_index in range(testing_df.shape[0]):
        result = walk_tree(tree, testing_df.iloc[instance_index])
        y_pred.append(result)
        
    y_pred = [0 if y == "Not Diabetic" else 1 for y in y_pred]
    y_pred = pd.Series(data=y_pred)
    
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    for ind in range(y_pred.shape[0]):
        pred = y_pred[ind]
        actual = y_actual[ind]
        
        if(pred == 0 and actual == 0):
            true_negatives += 1
        elif(pred == 0 and actual == 1):
            false_negatives += 1
        elif(pred == 1 and actual == 1):
            true_positives += 1
        else:
            false_positives += 1
    
    accuracy = calculate_accuracy(true_positives, false_positives, true_negatives, false_negatives)
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f_measure = calculate_f_measure(true_positives, false_positives, true_negatives, false_negatives)
    
    return accuracy, precision, recall, f_measure

def split_train_test(dataframe, train_percentage):
    training_df = dataframe.sample(frac = train_percentage / 100)
    training_indexes = list(training_df.index.values)
    testing_df = dataframe.drop(index=training_indexes)
    testing_df.reset_index(drop=True, inplace=True)

    return training_df, testing_df

def cross_val_split(dataframe, folds):
    dataframe_splits = []
    df_copy = dataframe.copy()
    fold_size = int(df_copy.shape[0] / folds)
    
    for i in range(folds):
        current_fold = df_copy.sample(n = fold_size)
        df_copy = df_copy.drop(index=list(current_fold.index.values))
        current_fold = current_fold.reset_index(drop=True)
        dataframe_splits.append(current_fold)
        
    return dataframe_splits

def cross_validate(dataframe, tree, target_feature, folds):
    avg_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    avg_f_measure = 0
    
    dataframe_folds = cross_val_split(dataframe, folds)
    
    for fold in range(folds):
        data = dataframe_folds[fold]
        acc, prec, rec, f = measure_tree_performance(tree, data, target_feature)
        avg_accuracy += acc
        avg_precision += prec
        avg_recall += rec
        avg_f_measure += f

    avg_accuracy /= folds
    avg_precision /= folds
    avg_recall /= folds
    avg_f_measure /= folds
    
    print("Avg. cross-validated accuracy: " + str(avg_accuracy))
    print("Avg. cross-validated precision: " + str(avg_precision))
    print("Avg. cross-validated recall: " + str(avg_recall))
    print("Avg. cross-validated f-measure: " + str(avg_f_measure))
    
if __name__ == "__main__":
    init()
