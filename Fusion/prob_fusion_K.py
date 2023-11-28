# version 1
# from sklearn.ensemble import IsolationForest
# import pickle
# import numpy as np
# import pandas as pd
#
# file_name='C:/Users/Administrator/Desktop/diffusion model/Results/data_encoded/en_test_all.pkl'
# with open(file_name, "rb") as f:
#      labels = np.squeeze(pickle.load(f)[1])
# # labels = np.squeeze(np.concatenate((np.zeros((300, 1),dtype= int), np.full((1800, 1), -1,dtype=int))))
# #iforest
# df = pd.read_excel('C:/Users/Administrator/Desktop/diffusion model/Results/Prob_iforest_original.xlsx', sheet_name='sheet1')
# prob_n_ifo = df.iloc[:, 1].values # 如果 prob 是第一列
# prob_ab_ifo = df.iloc[:, 2] .values # 如果 ab_prob 是第二列
# accuracy_ifo = df['Accuracy'].iloc[0]
# print('异常识别：', accuracy_ifo)
#
# #clustering
# # df = pd.read_excel('C:/Users/Administrator/Desktop/DSClustering/Results/Result10/reg_1_1_1.xlsx', sheet_name='Sheet1')
# df = pd.read_excel('C:/Users/Administrator/Desktop/DSClustering/Results_original/Result1/reg_1_1_1.xlsx', sheet_name='Sheet1')
# prob_C0_cls = df.iloc[:, 4].values # 如果 prob 是第一列
# prob_C1_cls = df.iloc[:, 5].values
# prob_C2_cls = df.iloc[:, 6].values
# prob_C3_cls = df.iloc[:, 7].values
# prob_C4_cls = df.iloc[:, 8].values
# prob_C5_cls = df.iloc[:, 9].values
# prob_C6_cls = df.iloc[:, 10].values
# accuracy_cls = round(df['acc_x'].iloc[0], 4)
# print('聚类', accuracy_cls)
#
# for i in range(11):
#     print(i)
#     w_ifo = i/10
#     w_cls = 1-w_ifo
#     prob_n = w_ifo*prob_n_ifo +w_cls*prob_C0_cls
#     prob_f = 1-prob_n
#     prob_C0 = prob_n
#     prob_C1 = np.zeros(len(labels))
#     prob_C2 = np.zeros(len(labels))
#     prob_C3 = np.zeros(len(labels))
#     prob_C4 = np.zeros(len(labels))
#     prob_C5 = np.zeros(len(labels))
#     prob_C6 = np.zeros(len(labels))
#     pre_labels = np.squeeze(np.zeros(labels.shape, dtype=int))
#     for i in range(len(labels)):
#         prob_C1[i]= (prob_C1_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
#         prob_C2[i]= (prob_C2_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
#         prob_C3[i]= (prob_C3_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
#         prob_C4[i]= (prob_C4_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
#         prob_C5[i]= (prob_C5_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
#         prob_C6[i]= (prob_C6_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
#         pre_labels[i] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}[np.argmax([prob_C0[i], prob_C1[i], prob_C2[i],prob_C3[i],prob_C4[i],prob_C5[i],prob_C6[i]])]
#     n_error=0
#     for i in range(len(labels)):
#         if labels[i] != pre_labels[i]:
#             n_error+=1
#     accuracy = 1- (n_error/len(labels))
#     accuracy = round(accuracy,4)
#     print("fusion accuracy",accuracy)
#
#version 2
import pickle
import numpy as np
import pandas as pd

file_name='D:/Desktop/paper1/diffusion model/Results/data_encoded/en_test_all.pkl'
with open(file_name, "rb") as f:
     labels = np.squeeze(pickle.load(f)[1])
df_result = pd.DataFrame()
for j in range(1, 11):
    print('实验次数:',j)
    # iforest
    # #original
    df = pd.read_excel('D:/Desktop/paper1/experiment/Prob_iforest_original_4metrics.xlsx', sheet_name='Sheet' + str(j))
    # #encoded
    # df = pd.read_excel('D:/Desktop/paper1/experiment/Prob_iforest_4metrics.xlsx', sheet_name='Sheet' + str(j))
    prob_n_ifo = df.iloc[:, 1].values
    prob_ab_ifo = df.iloc[:, 2] .values
    accuracy_ifo = df['Accuracy'].iloc[0]
    print('Anomaly detection：', accuracy_ifo)

    #clustering
    #original
    df = pd.read_excel('D:/Desktop/paper1/DSClustering/Results_original/Result'+str(j)+'/reg_1_1_1.xlsx', sheet_name='Sheet1')
    # #encoded
    # df = pd.read_excel('D:/Desktop/paper1/DSClustering/Results/Result'+str(j)+'/reg_1_1_1.xlsx', sheet_name='Sheet1')

    prob_C0_cls = df.iloc[:, 4].values # 如果 prob 是第一列
    prob_C1_cls = df.iloc[:, 5].values
    prob_C2_cls = df.iloc[:, 6].values
    prob_C3_cls = df.iloc[:, 7].values
    prob_C4_cls = df.iloc[:, 8].values
    prob_C5_cls = df.iloc[:, 9].values
    prob_C6_cls = df.iloc[:, 10].values
    accuracy_cls = round(df['acc_x'].iloc[0], 4)
    print('Fault clustering', accuracy_cls)

    for k in range(11):
        print(k)
        w_ifo = k/10
        w_cls = 1-w_ifo
        prob_n = w_ifo*prob_n_ifo +w_cls*prob_C0_cls
        prob_f = 1-prob_n
        prob_C0 = prob_n
        prob_C1 = np.zeros(len(labels))
        prob_C2 = np.zeros(len(labels))
        prob_C3 = np.zeros(len(labels))
        prob_C4 = np.zeros(len(labels))
        prob_C5 = np.zeros(len(labels))
        prob_C6 = np.zeros(len(labels))
        pre_labels = np.squeeze(np.zeros(labels.shape, dtype=int))
        for i in range(len(labels)):
            prob_C1[i]= (prob_C1_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
            prob_C2[i]= (prob_C2_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
            prob_C3[i]= (prob_C3_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
            prob_C4[i]= (prob_C4_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
            prob_C5[i]= (prob_C5_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
            prob_C6[i]= (prob_C6_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
            pre_labels[i] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}[np.argmax([prob_C0[i], prob_C1[i], prob_C2[i],prob_C3[i],prob_C4[i],prob_C5[i],prob_C6[i]])]
        n_error=0
        for i in range(len(labels)):
            if labels[i] != pre_labels[i]:
                n_error+=1
        accuracy = 1- (n_error/len(labels))
        accuracy = round(accuracy,4)
        print("fusion accuracy",accuracy)

        # 将结果DataFrame写入Excel文件
        data = {f'{k}': [accuracy]}
        accuracy_df = pd.DataFrame(data)
        df_result = pd.concat([df_result, accuracy_df], axis=1)
        with pd.ExcelWriter('D:/Desktop/paper1/experiment/fusion.xlsx', engine='openpyxl', mode='w') as writer:
            df_result.to_excel(writer, sheet_name='Sheet1',  float_format='%.4f')
