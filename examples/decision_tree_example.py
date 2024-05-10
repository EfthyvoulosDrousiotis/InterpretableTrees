# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import statistics

from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
import pandas as pd
from mpi4py import MPI
import numpy as np
from minepy import MINE
import random
from tabulate import tabulate
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import warnings
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")




df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\nature_paper\hepatitis.csv")
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()



def generate_uniform_list(n):
    uniform_list = np.linspace(0, 1, n)
    return uniform_list

iterations = 1
accuracies = []

# Split the dataset into training and testing sets

trees_similarities_many = []
weird_acc = []
collection_of_weird_acc = []
average_acc = []
for i in range(iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    a = 10
    #b = 5
    target = dt.TreeTarget(a)
    cumulative_sum_list = generate_uniform_list(X.shape[1])
    initialProposal = dt.TreeInitialProposal(X_train, y_train, cumulative_sum_list)
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
    try:
        treeSMCSamples, weights,leaf_possibilities = dtSMC.sample(20, 100, cumulative_sum_list) #(iterations, particles)
        smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
        smcAccuracy = [dt.accuracy(y_test, smcLabels)]
        SMC_diagnostics = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=False)
        smcAccuracy_diagnostics = [dt.accuracy(y_test, i)for i in SMC_diagnostics]
        average_acc.append(smcAccuracy_diagnostics)
        accuracies.append(smcAccuracy)
     
    
        print("dataset random state:", i, "accuracy is:", smcAccuracy)
        accuracy_per_dataset_iteration = []
        av_weight = np.sum(weights)/len(weights)
        median_weight = statistics.median(weights)
        sample_for_acc = []


        sample_for_acc = []
        for i in range(len(weights)):#select only the trees where its weights are above the median value
            if weights[i]>median_weight:
                sample_for_acc.append(treeSMCSamples[i])

        smcLabels_filtered = dt.stats(sample_for_acc, X_test).predict(X_test, use_majority=True)
        smcAccuracy_filtered = [dt.accuracy(y_test, smcLabels_filtered)]#majority voting for the filtered values
        SMC_diagnostics_filtered = dt.stats(sample_for_acc, X_test).predict(X_test, use_majority=False)
        smcAccuracy_diagnostics_filtered = [dt.accuracy(y_test, i)for i in SMC_diagnostics_filtered]#average accuracy for the 
        print("filtered majority accuracy: ", smcAccuracy_filtered)
        print("filtered accuracy:", [np.sum(smcAccuracy_diagnostics_filtered)/len(smcAccuracy_diagnostics_filtered)])
        '''
        weird acc gets the best accuracy between the majority voting, filtered majority voting and average filtered accuracy
        '''
        weird_acc.append(smcAccuracy_filtered) 
        weird_acc.append(np.sum(smcAccuracy_diagnostics_filtered)/len(smcAccuracy_diagnostics_filtered)) 
        weird_acc.append(smcAccuracy)
        max_acc = max(weird_acc)
        weird_acc= []
        collection_of_weird_acc.append(max_acc)
        print("-----------------------------")
        
        k_performing_trees = 5
        top_k_performing_trees, acc, top_tree_leaf_poss = dt.stats.print_top_k_performing_trees(smcAccuracy_diagnostics, treeSMCSamples, k_performing_trees, np.exp(weights), leaf_possibilities)
        
        trees_similarities = []
        for i in range(k_performing_trees):
            #trees_similarities.append("tree{i}")
            for k in range(k_performing_trees):
                similarity = dt.trees_similarity(top_k_performing_trees[i].tree, top_k_performing_trees[k].tree)
                trees_similarities.append(similarity)
                #print("tree ", i ,"and tree",k, "are ",similarity,"similar" )
            trees_similarities_many.append(trees_similarities)
            trees_similarities= []
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
        
#plt.plot(smcAccuracy_diagnostics, weights)
plt.scatter(smcAccuracy_diagnostics, (weights))

print("majority voting accuracy(normal):", np.sum(accuracies)/iterations)
print("average  accuracy: ", np.mean(average_acc))

s=0
for elem in collection_of_weird_acc:
    if isinstance(elem, list):
        s+=elem[0]
    else:
        s+=elem

print("weird collection accuracy: ", np.sum(s)/iterations)#the accuracy of the weird collection of accuracies
'''
this section is for the trees similarity
'''
k_performing_trees = 5
top_k_performing_trees, acc, leafs_poss = dt.stats.print_top_k_performing_trees(smcAccuracy_diagnostics, treeSMCSamples, k_performing_trees, np.exp(weights), leaf_possibilities)
trees_similarities_many = []
trees_similarities = []
for i in range(k_performing_trees):
    #trees_similarities.append("tree{i}")
    for k in range(k_performing_trees):
        similarity = dt.trees_similarity(top_k_performing_trees[i].tree, top_k_performing_trees[k].tree)
        trees_similarities.append(similarity)
        #print("tree ", i ,"and tree",k, "are ",similarity,"similar" )
    trees_similarities_many.append(trees_similarities)
    trees_similarities= []
    
# def calculate_complexity(tree):
#     #print("tree: ", tree.tree, "num of nodes is: ", len(tree.tree),"and num of leafs is: ",len(tree.leafs))
#     return (len(tree.tree)+len(tree.leafs))
    
# trees_complexity = []
# for tree in top_k_performing_trees:
#     trees_complexity.append(calculate_complexity(tree))

# #plt.scatter(trees_complexity, acc)
# print("average accuracy k top trees: ", np.sum(acc)/k_performing_trees)
# smcLabels_k_top = dt.stats(top_k_performing_trees, X_test).predict(X_test, use_majority=True)#labels for the top k performing trees
# smcAccuracy_k_top = [dt.accuracy(y_test, smcLabels)]#majority voting accuracy for the top k performing trees
# print("majority voting for k top trees:", smcAccuracy_k_top)

# col_names = ["Trees", "Tree0", "Tree1", "Tree2","Tree3", "Tree4","Tree5","Tree6","Tree7","Tree8","Tree9"]

# print(tabulate(trees_similarities_many, headers=col_names, tablefmt="fancy_grid", showindex="always"))


# from collections import Counter
# # count the occurrences of each point
# c = Counter(zip(trees_complexity,acc))
# # create a list of the sizes, here multiplied by 10 for scale
# s = [10*c[(xx,yy)] for xx,yy in zip(trees_complexity,acc)]
# plt.scatter(trees_complexity, acc, s=s)
# plt.xlabel("Number of nodes")
# plt.ylabel("Accuracy")
# #
# plt.show()

# '''
# plot the i_th you want compared with the rest to see their similarities_vs_accuracies 
# '''
# from collections import Counter
# # count the occurrences of each point
# c = Counter(zip(trees_similarities_many[2][1:],acc))
# # create a list of the sizes, here multiplied by 10 for scale
# s = [10*c[(xx,yy)] for xx,yy in zip(trees_similarities_many[2][1:],acc)]
# plt.scatter(trees_similarities_many[2][1:], acc, s=s)
# plt.title("Least similar tree similarities and accuracies")
# plt.xlabel("similarity")
# plt.ylabel("accuracy")
# #
# plt.show()

# depth = []
# for tree in top_k_performing_trees:
#     depth.append(max(tree.tree, key=lambda x: x[-1])[-1])
    
# from collections import Counter
# # count the occurrences of each point
# c = Counter(zip(depth,acc))
# # create a list of the sizes, here multiplied by 10 for scale
# s = [10*c[(xx,yy)] for xx,yy in zip(depth,acc)]
# plt.scatter(depth, acc, s=s)
# plt.xlabel("Depth")
# plt.ylabel("Accuracy")
# #
# plt.show()


# ax = df.plot.bar(rot=0)
# plt.xlabel("Trees")
# plt.ylabel("Accuracy")
# plt.savefig('Accuracy.pdf', dpi=100)

    