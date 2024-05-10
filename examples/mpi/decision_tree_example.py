import numpy as np
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn import datasets
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.domain import decision_tree as dt
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.util import gather_all
from minepy import MINE
import pandas as pd
import statistics
from tabulate import tabulate
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")


#df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\nature paper\Contraceptive.csv")
#for dan joyce work
# # #df=df.drop(["Date"], axis = 1)
# df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\datasets_smc_mcmc_CART\StudentsMH.csv")
# # # df.Target=df.Target.astype('category').cat.codes

# df = df.dropna()
# df.gender=df.gender.astype('category').cat.codes
# df.course=df.course.astype('category').cat.codes
# df.year=df.year.astype('category').cat.codes
# df.CGPA=df.CGPA.astype('category').cat.codes
# df.MaritalStatus=df.MaritalStatus.astype('category').cat.codes
# df.Anxiety=df.Anxiety.astype('category').cat.codes
# df.PanicAttack=df.PanicAttack.astype('category').cat.codes
# df.treatment=df.treatment.astype('category').cat.codes
# df.Target=df.Target.astype('category').cat.codes

# df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\students.csv",sep=";")

#df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\ACHAdata\killing_yourself_in_the_past_year.csv")
df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\ACHAdata\killing_yourself_in_the_past_year_non_uni_features.csv")
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)


def mic_cum_list(X_train, y_train): 
    single_dimensional_array = y_train.reshape(-1, 1)
    Data = np.concatenate((X_train, single_dimensional_array), axis=1)
    results = []
    mine = MINE(alpha=0.5, c=15) #MIC correlation
    for column in  Data.T:
        mine.compute_score(column,Data[:,-1]) 
        results.append(mine.mic())
    
    results=results[:-1]#remove the coeefficient of the target with the target
    corr_matrix = df.corr()#perasons correlation
    div = sum(results)
    res = []
    for elemenent in results:
        res.append(elemenent/div)#normalise the weights
    cumulative_sum_list = np.cumsum(res)#cumulative sum
    return cumulative_sum_list


cumulative_sum_list = mic_cum_list(X_train, y_train)



N = 16 #1<< 4
T = 50
seed = 0
num_of_iter= 10
weird_acc = []
collection_of_weird_acc = []
accuracies = []
filtered_acc = []
top_k_trees_average_acc = []
top_k_trees_majority_acc = []
confusion_matrices = []

for i in range(num_of_iter):
    weird_acc= []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i, stratify=df.REGION)
    a = 10
    target = dt.TreeTarget(a)
    cumulative_sum_list = mic_cum_list(X_train, y_train)
    initialProposal = dt.TreeInitialProposal(X_train, y_train, cumulative_sum_list)
    exec = Executor_MPI()
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal, False, exec=exec)
    try:
        MPI.COMM_WORLD.Barrier()
        start = MPI.Wtime()
        comm = MPI.COMM_WORLD
        weights_gather = np.zeros(N, dtype='d') 
        treeSMCSamples, weights = dtSMC.sample(T, N, cumulative_sum_list)
    
        MPI.COMM_WORLD.Barrier()
        end = MPI.Wtime()
    
        if MPI.COMM_WORLD.Get_size() > 1:
            treeSMCSamples = gather_all(treeSMCSamples, exec)
            comm.Allgather([np.array(weights, dtype = 'd'),  MPI.DOUBLE], [weights_gather, MPI.DOUBLE])
        
        weights = np.copy(weights_gather)
        
        smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test)
        smcAccuracy = balanced_accuracy_score(y_test, smcLabels)
        SMC_diagnostics = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=False)
        Individual_Trees_acc = [balanced_accuracy_score(y_test, i)for i in SMC_diagnostics]#find the accuracy of the individual trees
        accuracies.append(smcAccuracy)
        
        
        accuracy_per_dataset_iteration = []
        median_weight = statistics.median(weights)#find the median weights
        sample_for_acc = []
        
        
        sample_for_acc = []
        for k in range(len(weights)):#select only the trees where its weights are above the median value
            if weights[k]>=median_weight:
                sample_for_acc.append(treeSMCSamples[k])

        smcLabels_filtered = dt.stats(sample_for_acc, X_test).predict(X_test, use_majority=True)
        smcAccuracy_filtered = [balanced_accuracy_score(y_test, smcLabels_filtered)]#majority voting for the filtered trees which are better than the median
        SMC_diagnostics_filtered = dt.stats(sample_for_acc, X_test).predict(X_test, use_majority=False)
        smcAccuracy_diagnostics_filtered = [balanced_accuracy_score(y_test, m)for m in SMC_diagnostics_filtered]#average accuracy for the filtered trees which are better than the median
        filtered_acc.append(smcAccuracy_filtered)
        '''
        weird acc gets the best accuracy between the majority voting, filtered majority voting and average filtered accuracy
        '''
        weird_acc.append(smcAccuracy_filtered[0]) 
        weird_acc.append(np.sum(smcAccuracy_diagnostics_filtered)/len(smcAccuracy_diagnostics_filtered)) 
        weird_acc.append(smcAccuracy)
        max_acc = max(weird_acc)
        
        
        
        k_performing_trees = 8
        top_k_performing_trees, acc = dt.stats.print_top_k_performing_trees(Individual_Trees_acc, treeSMCSamples, k_performing_trees, weights)
        smcLabels_k_top = dt.stats(top_k_performing_trees, X_test).predict(X_test, use_majority=True)#labels for the top k performing trees
        smcAccuracy_k_top = [balanced_accuracy_score(y_test, smcLabels_k_top)]#majority voting accuracy for the top k performing trees
        top_k_trees_majority_acc.append(smcAccuracy_k_top)
        top_k_trees_average_acc.append(np.sum(acc)/k_performing_trees)
        collection_of_weird_acc.append(max_acc)
        confusion_matrices.append(confusion_matrix(y_test, smcLabels_k_top))
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("dataset random state:", i)
            print("accuracy is:", smcAccuracy)
            print("Majority voting accuracy from trees trees with more than the median loglike: ", smcAccuracy_filtered)
            print("Average accuracy from trees trees with more than the median loglike:", [np.sum(smcAccuracy_diagnostics_filtered)/len(smcAccuracy_diagnostics_filtered)])
            print("majority accuracy of k_top:", smcAccuracy_k_top)
            print("average accuracy of k_top: ", np.sum(acc)/k_performing_trees)
            print("SMC run-time: ", end-start)
            print("===============================================")
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
        

s=0
for elem in collection_of_weird_acc:
    if isinstance(elem, list):
        s+=elem[0]
    else:
        s+=elem


if MPI.COMM_WORLD.Get_rank() == 0:
    print("majority voting accuracy(normal):", np.sum(accuracies)/num_of_iter)
    print("Filtered majority voting accuracy(normal): ", np.sum(filtered_acc)/num_of_iter)#the accuracy of the weird collection of accuracies
    print("average accuracy k top trees: ", np.sum(top_k_trees_average_acc)/num_of_iter)
    print("majority voting for k top trees:", np.sum(top_k_trees_majority_acc)/num_of_iter) 
    print("weird collection accuracy: ", np.sum(s)/num_of_iter)#the accuracy of the weird collection of accuracies
    print(classification_report(y_test, smcLabels_k_top))
    max_shape = np.array(max(confusion_matrices, key=lambda x: x.size)).shape

    # Pad matrices to the maximum shape to enable broadcasting
    padded_matrices = [np.pad(matrix, pad_width=tuple((0, m - s) for s, m in zip(matrix.shape, max_shape)), mode='constant') for matrix in confusion_matrices]

    confusion_m_sum = np.sum(padded_matrices, axis=0)
    print(confusion_m_sum)






'''
this section is for the trees similarity
'''

# k_performing_trees = 5
# top_k_performing_trees, acc = dt.stats.print_top_k_performing_trees(Individual_Trees_acc, treeSMCSamples, k_performing_trees, np.exp(weights))
# trees_similarities_many = []
# trees_similarities = []
# for i in range(k_performing_trees):
#     trees_similarities.append("tree{i}")
#     for k in range(k_performing_trees):
#         similarity = dt.trees_similarity(top_k_performing_trees[i].tree, top_k_performing_trees[k].tree)
#         trees_similarities.append(similarity) 
#         #print("tree ", i ,"and tree",k, "are ",similarity,"similar" )
#     trees_similarities_many.append(trees_similarities)
#     trees_similarities= []
    
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

# col_names = ["Trees", "Tree1", "Tree2", "Tree3","Tree4", "Tree5"]#,"Tree6","Tree7","Tree8","Tree9","Tree10"]

# print(tabulate(trees_similarities_many, headers=col_names, tablefmt="fancy_grid", showindex="always"))




