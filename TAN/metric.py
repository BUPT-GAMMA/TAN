import numpy as np
import collections
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata

PAD = 0
EOS = 1

def _retype(y_prob, y):
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    y_prob = np.array(y_prob)
    y = np.array(y)
    return y_prob, y

def _binarize(y, n_classes=None):
    return label_binarize(y, classes=range(n_classes))

def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(y_prob, y, k=10):
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]
    return sum(ranks) / float(len(ranks))

def MRR(y_prob,y):
    ranks = []
    n_classes = y_prob[0].shape[0]
    for p_, y_ in zip(y_prob, y):
        ranks += [1 / (n_classes - rankdata(p_, method='max')[y_] + 1)]
    return sum(ranks) / float(len(ranks))

def hits_k(y_prob, y, k=10):
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)

def portfolio(pred, gold, k_list=[1,5,10,20]):
    scores_len = 0
    y_prob=[]
    y=[]
    for i in range(gold.shape[0]): # predict counts
        if gold[i]!=PAD and gold[i]!=EOS:
            scores_len+=1.0
            y_prob.append(pred[i])
            y.append(gold[i])
    scores = {}
    scores['MRR'] = MRR(y_prob, y)
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob, y, k=k)
        scores['map@' + str(k)] = mapk(y_prob, y, k=k)
    return scores, scores_len