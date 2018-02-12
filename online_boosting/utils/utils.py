import warnings
from sklearn.datasets import load_svmlight_file
warnings.filterwarnings("ignore", module="sklearn")

from online_boosting.ensemblers.adaboost import AdaBooster
from online_boosting.ensemblers.ogboost import OGBooster
from online_boosting.ensemblers.ocpboost import OCPBooster
from online_boosting.ensemblers.expboost import EXPBooster
from online_boosting.ensemblers.osboost import OSBooster
from online_boosting.ensemblers.smoothboost import SmoothBooster

from online_boosting.learners.naive_bayes_gaussian import NaiveBayes as GaussianNB
from online_boosting.learners.naive_bayes_binary import NaiveBayes as BinaryNB
from online_boosting.learners.perceptron import Perceptron
from online_boosting.learners.random_stump import RandomStump
#from online_boosting.learners.decision_stump import DecisionStump
from online_boosting.learners.decision_tree import DecisionTree
from online_boosting.learners.knn import kNN
from online_boosting.learners.histogram import RNB
from online_boosting.learners.winnow import Winnow
from online_boosting.learners.mlp import MLP
from online_boosting.learners.mlp2 import MLP as MLP2


def load_data(filename):
    X, y = load_svmlight_file(filename)

    data = zip(X, y)
    return data


ensemblers = {
    "AdaBooster": AdaBooster,
    "OCPBooster": OCPBooster,
    "EXPBooster": EXPBooster,
    "OGBooster": OGBooster,
    "OSBooster": OSBooster,
    "SmoothBooster": SmoothBooster
}


def get_ensembler(ensembler_name):
    return ensemblers[ensembler_name]


weak_learners = {
    "GaussianNB": GaussianNB,
    "BinaryNB": BinaryNB,
    "kNN": kNN,
    #"MLP": MLP,
    #"MLP2": MLP2,
    #"DecisionStump": DecisionStump,
    "DecisionTree": DecisionTree,
    "Perceptron": Perceptron,
    "RandomStump": RandomStump,
    #"Winnow": Winnow,
    #"Histogram": RNB
}


def get_weak_learner(weak_learner_name):
    return weak_learners[weak_learner_name]
