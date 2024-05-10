from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
import pandas as pd
from minepy import MINE




df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\LION18\movies.csv")
# #df=df.drop(["Date"], axis = 1)
df = df.dropna()
# df.month=df.month.astype('category').cat.codes
# df.day=df.day.astype('category').cat.codes
iterations = [1000]#,10,20,30,40,50,60,70,80,90,100]
# data = datasets.load_diabetes()
# X = data.data
# y = data.target

def mic_cum_list(X_train, y_train): 
    single_dimensional_array = y_train.reshape(-1, 1)
    Data = np.concatenate((X_train, single_dimensional_array), axis=1)
    results = []
    mine = MINE(alpha=0.1, c=15) #MIC correlation
    for column in  Data.T:
        mine.compute_score(column,Data[:,-1]) 
        results.append(mine.mic())
    
    results=results[:-1]#remove the coefficient of the target with the target
    #corr_matrix = df.corr()#perasons correlation
    div = sum(results)
    res = []
    for elemenent in results:
        res.append(elemenent/div)#normalise the weights
    cumulative_sum_list = np.cumsum(res)#cumulative sum
    return cumulative_sum_list

    
    



for k in iterations:
    iterations = k
    trees = 4
    y = df.Target
    X = df.drop(['Target'], axis=1)
    X = X.to_numpy()
    y = y.to_numpy()
    accuracies = []
    top_trees_majority = []
    top_trees_average = []
    for i in range (10):   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
        cumulative_sum_list = mic_cum_list(X_train, y_train)
        a = 100
        target = dt.RegressionTreeTarget(a)
        initialProposal = dt.TreeInitialProposal(X_train, y_train, cumulative_sum_list)
        dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
        try:
            treeSMCSamples,weights, t = dtSMC.sample(iterations, trees, cumulative_sum_list) #(num of iteration, numf of particles)
            
            smcLabels = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
            smcAccuracy = [dt.accuracy_mse(y_test, smcLabels)]
            accuracies.append(smcAccuracy)
            
            smcLabels_average = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=False)
            smcAccuracy_average = [dt.accuracy_mse(y_test, label) for label in smcLabels_average]
            
            
            k_performing_trees = 4
            
            
            
            
            top_k_performing_trees, acc = dt.stats.print_top_k_performing_trees(smcAccuracy_average, treeSMCSamples, k_performing_trees, np.exp(weights))
            
            smcLabels_top_k_average_labels = dt.RegressionStats(top_k_performing_trees, X_test).predict(X_test, use_majority=False)
            smcAccuracy_top_k_average = [dt.accuracy_mse(y_test, label) for label in smcLabels_top_k_average_labels]
            top_trees_average.append(np.mean(smcAccuracy_top_k_average))
            
            smcLabels_top_k_majority_labels = dt.RegressionStats(top_k_performing_trees, X_test).predict(X_test, use_majority=True)
            smcAccuracy_top_k_majority = [dt.accuracy_mse(y_test, smcLabels_top_k_majority_labels)]
            top_trees_majority.append(smcAccuracy_top_k_majority)
            
            
            
            
            
            print("iterations:  ", i ,"majority accuracy: ", smcAccuracy)
            print("average accuracy: ", np.mean(smcAccuracy_average))
            print("top k trees average: ", np.mean(smcAccuracy_top_k_average))
            print("majority top k accuracy ", smcAccuracy_top_k_majority)
            print("----------- ")
        except ZeroDivisionError:
            print("SMC sampling failed due to division by zero")
print("------------Final Results----------")
print("average majority voting accuracy is: ", np.mean(accuracies))
print("average top k accuracy is: ", np.mean(top_trees_average))
print("majority top k accuracy is: ", np.mean(top_trees_majority))
    


