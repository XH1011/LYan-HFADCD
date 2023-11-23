from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
import pandas as pd
# np.random.seed(130)
for run_number in range(10):
    dir = 'D:/Desktop/paper1/diffusion model/Results/data_encoded/en_'
    # dir='D:/Desktop/paper1/diffusion model/data/'
    # 读取正常数据
    path_train_0 = dir + 'x0_train.pkl'
    with open(path_train_0, 'rb') as f0:
        X_train_0 = pickle.load(f0)
        # X_train_0 = pickle.load(f0)[0]

    # # 标准化
    des = X_train_0.std(axis=0)
    media = X_train_0.mean(axis=0)
    X_train_0 = (X_train_0 - media) / des

    clf = IsolationForest()
    clf.fit(X_train_0)

    # 测试阶段
    faults = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    accum_percent = []
    X_test_all = []
    for fault in faults:
        # path_test = dir + fault + '_test.pkl'
        # path_test = dir + 'generate_' + fault + '_test.pkl'
        path_test = dir + fault + '_test.pkl'
        with open(path_test, 'rb') as f:
            X_test = pickle.load(f)
            # X_test = pickle.load(f)[0]
        X_test = (X_test - media) / des
        X_test_all.append(X_test)
    X_test_all = np.concatenate(X_test_all, axis=0)
    score = clf.decision_function(X_test_all)
    min_score = np.min(score)
    max_score = np.max(score)
    prob = (score - min_score) / (max_score - min_score)
    ab_prob = 1 - prob

    labels = np.squeeze(np.concatenate((np.zeros((300, 1), dtype=int), np.full((1800, 1), -1, dtype=int))))
    pre_labels = np.where(prob > ab_prob, 0, -1)

    n_error = 0
    for i in range(len(labels)):
        if labels[i] != pre_labels[i]:
            n_error += 1
    accuracy = 1 - (n_error / len(labels))
    print('accuracy', accuracy)
    TP = np.sum((labels == 0) & (pre_labels == 0))
    TN = np.sum((labels == -1) & (pre_labels == -1))
    FP = np.sum((labels == -1) & (pre_labels == 0))
    FN = np.sum((labels == 0) & (pre_labels == -1))
    precision = TP / (TP + FP)
    print('precision', precision)
    recall = TP / (TP + FN)
    print('recall', recall)
    F1_score = 2 * (precision * recall) / (precision + recall)
    print('F1_score', F1_score)

    data = {'Normal_prob': prob, 'Abnormal_prob': ab_prob}
    df = pd.DataFrame(data)
    accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})
    precision_df = pd.DataFrame({'precision': [precision]})
    recall_df = pd.DataFrame({'recall': [recall]})
    F1_df = pd.DataFrame({'F1_score': [F1_score]})
    df = pd.concat([df, accuracy_df, precision_df, recall_df, F1_df], axis=1)
    with pd.ExcelWriter('D:/Desktop/paper1/experiment/Prob_iforest.xlsx', engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=f'Sheet{run_number + 1}', float_format='%.4f')




