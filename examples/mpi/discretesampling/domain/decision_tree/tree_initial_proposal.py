import math
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.base.random import RNG
from discretesampling.domain.decision_tree import Tree
import bisect


class TreeInitialProposal(DiscreteVariableInitialProposal):
    def __init__(self, X_train, y_train, cumulative_sum_list):
        self.X_train = X_train
        self.y_train = y_train
        self.cumulative_sum_list = cumulative_sum_list
        # self.rng = rng

    # def sample(self, rng):
    #     leafs = [1, 2]

    #     feature = rng.randomInt(0, len(self.X_train[0])-1)
    #     threshold = rng.randomInt(0, len(self.X_train)-1)
    #     tree = [[0, 1, 2, feature, self.X_train[threshold, feature],0]]
    #     return Tree(self.X_train, self.y_train, tree, leafs)
    
    def find_interval(self, uniform_num, cumulative_sum_list):
        # Use binary search to find the right interval index
        index = bisect.bisect_left(cumulative_sum_list, uniform_num)
        
        if index < len(cumulative_sum_list):
            if index == 0:
                return 0, (0, cumulative_sum_list[index])
            else:
                return index, (cumulative_sum_list[index - 1], cumulative_sum_list[index])
        else:
            return None, None 

    def sample(self, rng=RNG(), target=None):
        leafs = [1, 2]
        u = rng.uniform()
        feature, interval = self.find_interval(u, self.cumulative_sum_list)
        #print(index)
        #feature = rng.randomInt(0, len(self.X_train[0])-1)
        threshold = rng.randomInt(0, len(self.X_train)-1)
        tree = [[0, 1, 2, feature, self.X_train[threshold, feature], 0]]
        init_tree = Tree(self.X_train, self.y_train, tree, leafs)

        if target is None:
            return init_tree

        i = 0
        while i < len(leafs):
            u = rng.uniform()
            prior = math.exp(target.evaluatePrior(init_tree))
            # print("tree before: ", init_tree)
            if u < prior:
                init_tree = init_tree.grow_leaf(leafs.index(leafs[i]), rng)
                leafs = init_tree.leafs
            else:
                i += 1
            # print("tree after: ", init_tree)
        return init_tree

    def eval(self, x, target=None):
        num_features = len(self.X_train[0])
        num_thresholds = len(self.X_train)
        if target is None:
            return -math.log(num_features) - math.log(num_thresholds)
        else:
            return -math.log(num_features) - math.log(num_thresholds) + target.evaluatePrior(x)
