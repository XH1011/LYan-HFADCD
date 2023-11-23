import numpy as np
from munkres import Munkres
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize,MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import neighbors
from PIL import Image
import tensorflow as tf
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import keras.backend as K
from collections import Counter
from operator import itemgetter

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    #L1 真实标签 L2 预测
    Label1 = np.unique(L1) #保存数组L1的唯一元素，删除任何重复的值
    nClass1 = len(Label1) #Label1数据的数量
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i] #ind_cla1布尔数组，值为 True 的位置表示 L1 中与当前 Label1[i] 相等的元素位置
        ind_cla1 = ind_cla1.astype(float) #将ind_cla1转换为浮点型数组，将 True 转换为 1.0，False 转换为 0.0
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1) #?
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index) #index中存储的是什么？
    c = index[:,1] #第二列的所有元素
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C,ro):
    if ro < 1: #小于1，则执行稀疏化操作，否则直接返回原始矩阵 C
        N = C.shape[1] #列数
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0)) ##对矩阵 C 的绝对值按列进行降序排序，每列的元素按照从大到小的顺序排列
        Ind = np.argsort(-np.abs(C),axis=0) #对绝对值矩阵 C 按列进行索引排序
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float) #该列绝对值的总和
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i] #累计和
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def post_proC(C, K, d, alpha, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = thrC(C,ro)
    # drawC(C)
    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0])) #执行奇异值分解，并获取矩阵 C 的左奇异向量、前 r 个奇异值和右奇异向量
    # print(S)
    U = U[:,::-1] #将矩阵U按列进行逆序排序
    S = np.sqrt(S[::-1]) #将矩阵S按列进行逆序排序后，再取其平方根
    S = np.diag(S) #按照对角线形式创建为一个对角矩阵，一维数组S将返回为二维矩阵
    U = U.dot(S) #U 与对角矩阵 S 进行点乘运算（矩阵乘法)
    U = normalize(U, norm='l2', axis = 1) #对矩阵U进行L2范式归一化，使其在第二个轴上的向量长度为1
    Z = U.dot(U.T)
    Z = Z * (Z>0) #将小于等于零的元素变为零，大于零的元素保持不变(True被当作1，False为0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize',random_state=22)
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1 #获得聚类结果，fit_predict 方法获得，结果中的类别标签从 0 开始，通过加 1 进行偏移
    return grp, L
    #返回聚类结果和对称矩阵L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:]) #真实标签与聚类结果不一致的样本数量
    missrate = err_x.astype(float) / (gt_s.shape[0]) #聚类错误样本数量占总样本数量的比例
    NMI = metrics.normalized_mutual_info_score(gt_s, c_x) #衡量真实标签和聚类结果的相似性

    purity = 0
    N = gt_s.shape[0]
    Label1 = np.unique(gt_s) #获取真实标签中的唯一值
    nClass1 = len(Label1) #计算真实标签中的类别数量
    Label2 = np.unique(c_x)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    for label in Label2:
        tempc = [i for i in range(N) if s[i] == label]
        hist,bin_edges = np.histogram(gt_s[tempc],Label1)
        purity += max([np.max(hist),len(tempc)-np.sum(hist)])
    purity /= N #计算平均纯度？
    return missrate,NMI,purity
    #评估聚类结果的质量，其中聚类错误率越低、标准化互信息越高、纯度越高表示聚类结果越好

def display(Coef, subjs, d, alpha, ro,numc = None,label = None):
    if numc is None:
        numc = np.unique(subjs).shape[0]
    if label is None:
        label = subjs #如果未提供标签数组 label，则使用 subjs 数组作为标签(subjs 数组是通过输入得到的)
    y_x, L = post_proC(Coef, numc, d, alpha, ro) #聚类结果 y_x 和相关的相似性矩阵 L
    missrate_x, NMI, purity = err_rate(label, y_x)
    acc_x = 1 - missrate_x #聚类的准确率
    print("our accuracy: %.4f" % acc_x)
    print("our NMI: %.4f" % NMI, "our purity: %.4f" % purity)
    return acc_x,L,y_x
    #聚类准确率、相似形矩阵和聚类结果

def display1(Coef, subjs, d, alpha, ro,numc = None,label = None):
    if numc is None:
        numc = np.unique(subjs).shape[0]
    if label is None:
        label = subjs
    y_x, L = post_proC(Coef, numc, d, alpha, ro)
    y_x=best_map(label,y_x)

    return L,y_x
    #相似度矩阵 L 和映射后的聚类结果 y_x

def prob(L, grp):
    # num_clusters = len(np.unique(grp))
    # probabilities = np.zeros((len(grp), num_clusters))
    #
    # for i in range(num_clusters):
    #     cluster_indices = np.where(grp == i + 1)[0]
    #     cluster_similarity_values = L[cluster_indices, :]
    #     # 计算每个簇的相似性值之和
    #     cluster_similarity_sum = np.sum(cluster_similarity_values, axis=0)
    #     # 计算每个数据点属于该簇的概率
    #     probabilities[:, i] = cluster_similarity_sum
    # # 对每个数据点的概率进行归一化
    # probabilities = probabilities / np.sum(probabilities, axis=1)[:, np.newaxis]
    #
    # # 将标签分成7组，每组300个标签
    # label_groups = [grp[i:i + 300] for i in range(0, len(grp), 300)]
    # probabilities_groups = [probabilities[i:i + 300, :] for i in range(0, len(probabilities), 300)]
    # # 找出每组中数量最多的标签作为总标签
    # total_labels = []
    # for group in label_groups:
    #     label_counts = Counter(group)
    #     most_common_label = label_counts.most_common(1)[0][0]
    #     total_labels.append(most_common_label)
    # sorted_grp = sorted(zip(total_labels, label_groups))
    # sorted_probabilities = sorted(zip(total_labels, probabilities_groups))
    # re_grp = [group for _, group in sorted_grp]
    # re_grp = np.concatenate(re_grp)
    # re_probabilities = [group for _, group in sorted_probabilities]
    # re_probabilities = np.concatenate(re_probabilities)
    num_clusters = len(np.unique(grp))
    probabilities = np.zeros((len(grp), num_clusters))

    for i in range(num_clusters):
        cluster_indices = np.where(grp == i + 1)[0]
        cluster_similarity_values = L[cluster_indices, :]
        cluster_similarity_sum = np.sum(cluster_similarity_values, axis=0)
        probabilities[:, i] = cluster_similarity_sum

    # Normalize the probabilities
    probabilities = probabilities / np.sum(probabilities, axis=1)[:, np.newaxis]

    label_groups = [grp[i:i + 300] for i in range(0, len(grp), 300)]
    probabilities_groups = [probabilities[i:i + 300, :] for i in range(0, len(probabilities), 300)]

    total_labels = []
    sorted_probabilities = []

    for group, probs in zip(label_groups, probabilities_groups):
        label_counts = Counter(group)
        most_common_label = label_counts.most_common(1)[0][0]
        total_labels.append(most_common_label)
        sorted_probabilities.append(probs)

    # Combine the sorted probabilities and labels
    sorted_probabilities, _ = zip(*sorted(zip(sorted_probabilities, total_labels), key=lambda x: x[1]))

    re_probabilities = np.concatenate(sorted_probabilities)
    sorted_grp, _ = zip(*sorted(zip(label_groups, total_labels), key=lambda x: x[1]))
    re_grp = np.concatenate(sorted_grp)
    return re_probabilities,re_grp

def cal_acc(label_10_subjs,re_probabilities):
    n_error=0
    labels = np.argmax(re_probabilities, axis=1) + 1
    for i in range(len(labels)):
        if labels[i]!=label_10_subjs[i]:
            n_error+=1
    accuracy =1-(n_error/len(labels))
    return accuracy


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orangered','greenyellow','darkviolet']
marks = ['o','+','.']
#用于可视化的颜色和标记列表

def visualize(Img,Label,CAE=None,filep=None):
#filep=None时，不会保存图像，只会在窗口显示；要保存的话def visualize(Img,Label,CAE=None,filep='./image/visualize.png')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    if CAE is not None:
        bs = CAE.batch_size
        Z = CAE.transform(Img[:bs,:])
        Z = np.zeros([Img.shape[0], Z.shape[1]])
        for i in range(Z.shape[0] // bs):
            Z[i * bs:(i + 1) * bs, :] = CAE.transform(Img[i * bs:(i + 1) * bs, :])
        if Z.shape[0] % bs > 0:
            Z[-bs:, :] = CAE.transform(Img[-bs:, :])
    else:
        Z = Img
    Z_emb = TSNE(n_components=2).fit_transform(Z, Label) #使用t-SNE算法对Z进行降维
    # print(Z_emb)
    lbs = np.unique(Label)
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()
        # print(Z_embi)
        class_label = "Class " + str(ii + 1)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii % 10], marker=marks[ii // 10], label=class_label,s=3)
    ax1.legend(loc='upper right')
    # ax1.tick_params(direction='in')
    if filep is not None:
        plt.savefig(filep,bbox_inches='tight', pad_inches=0.1)
    # plt.figure(0)
    # plt.show()
#使用不同的颜色和标记符号，在子图上绘制降维后的数据点。添加图例（legend）以表示不同类别或标签

def visualize2(Img,Label):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    lbs = np.unique(Label)
    if Img.shape[1]>2:
        Z_emb = TSNE(n_components=2).fit_transform(Img, Label)
    else:
        Z_emb = Img
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()
        # print(Z_embi)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii % 10], marker=marks[ii // 10], label=str(ii),s=6)
    ax1.legend()
    # plt.figure(2)
    # plt.show()
#将数据的聚类结果可视化，并以不同的颜色和标记表示不同的聚类簇

#输入的数据Img的维度大于2，则使用TSNE算法对数据进行降维，将其转换为2维进行可视化；如果数据的维度已经是2维，则直接使用原始数据。进行可视化仅在窗口显示，没有提供保存选择
#首先检查是否传入了一个CAE（Contractive Autoencoder）对象，如果有，则利用CAE对数据进行降维，否则直接使用原始数据；然后再进行TSNE进行降维。可以进行保存选择

def visualize3(Img,Label,CAE=None,filep=None):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    # if CAE is not None:
    #     bs = CAE.batch_size
    #     Z = CAE.transform(Img[:bs,:])
    #     Z = np.zeros([Img.shape[0], Z.shape[1]])
    #     for i in range(Z.shape[0] // bs):
    #         Z[i * bs:(i + 1) * bs, :] = CAE.transform(Img[i * bs:(i + 1) * bs, :])
    #     if Z.shape[0] % bs > 0:
    #         Z[-bs:, :] = CAE.transform(Img[-bs:, :])
    # else:
    #     Z = Img
    Z_emb = TSNE(n_components=2, metric='precomputed').fit_transform(Img, Label)
    #metric='precomputed' 参数表示输入数据为预计算的距离矩阵，而不是原始特征数据。
    # print(Z_emb)
    lbs = np.unique(Label)
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()
        # print(Z_embi)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii % 10], marker=marks[ii // 10], label=str(ii),s=3)
    ax1.legend()
    if filep is not None:
        plt.savefig(filep)
    # plt.figure(3)
    # plt.show()


def NNtest(Img,Label,teImg,teLabel,CAE=None,n_neigh=1,km=False):
    Label = np.ravel(Label)
    teLabel = np.ravel(teLabel)
    n = Img.shape[0]
    if CAE is not None:
        Z = CAE.transform(Img)
        teZ = np.zeros([teImg.shape[0],Z.shape[1]])
        bs = CAE.batch_size
        for i in range(teZ.shape[0]//bs):
            teZ[i*bs:(i+1)*bs,:] = CAE.transform(teImg[i*bs:(i+1)*bs,:])
        if teZ.shape[0]%bs>0:
            teZ[-bs:,:] = CAE.transform(teImg[-bs:, :])
    else:
        print('raw')
        Z = np.reshape(Img,[Img.shape[0],-1])
        teZ = np.reshape(teImg,[teImg.shape[0],-1])
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neigh)
    clf.fit(Z, Label)
    pre = clf.predict(teZ)
    missrate_x, NMI, purity = err_rate(teLabel, pre)
    acc_x = 1 - missrate_x
    print("NN accuracy: %.4f" % acc_x)
    print("NN NMI: %.4f" % NMI, "NN purity: %.4f" % purity)
    if km:
        acc_km = KMtest(teZ,teLabel,None)
        return acc_x,acc_km
    return acc_x
#使用最近邻分类器对样本进行分类，并评估分类结果的准确率

def KMtest(Img,Label,CAE=None):
    Label = np.ravel(Label)
    n = Img.shape[0]
    if CAE is not None:
        Z = CAE.transform(Img)
    else:
        print('raw')
        Z = np.reshape(Img,[Img.shape[0],-1])
    clf = cluster.KMeans(n_clusters=np.unique(Label).shape[0])
    lb = clf.fit_predict(Z)
    missrate_x, NMI, purity = err_rate(Label, lb)
    acc_x = 1 - missrate_x
    print("KM accuracy: %.4f" % acc_x)
    print("KM NMI: %.4f" % NMI, "KM purity: %.4f" % purity)
    return acc_x
#使用 K-Means 算法对样本进行聚类，并评估聚类结果的准确率

def drawC(C,name='C-L2.png',norm=False):
    C = np.abs(C)
    C = C * (np.ones_like(C)-np.eye(C.shape[0]))
    if norm:
        C = C / np.sum(C,axis=1,keepdims=True)
    min_max_scaler = MinMaxScaler(feature_range=[0,255]) #后续看需要改吗？不是图像数据映射到什么范围？
    CN = min_max_scaler.fit_transform(C)
    CN = CN + 255*np.eye(C.shape[0])
    IC = Image.fromarray(CN).convert('L')
    IC.save(name)
    # IC.show()
#将给定的相关性矩阵绘制成图像，并保存为图片文件


def cosine(x1,x2):
    dot1 = tf.reduce_sum(tf.multiply(x1,x2))
    dot2 = tf.sqrt(tf.reduce_sum(tf.square(x1)))
    dot3 = tf.sqrt(tf.reduce_sum(tf.square(x2)))
    max_ = K.maximum(dot2*dot3, K.epsilon())
    return dot1 / max_
#计算两个向量的余弦相似度，量化它们之间的相似性或相关性

def cosinea(X,batch_size):
    b=[]
    #t=[]
    for i in range (batch_size):
        a = []
        #n = []
        for j in range (batch_size):
            f=cosine(X[i],X[j])
            #onel=np.ones(shape=f.shape)
            #zerol=np.zeros(shape=f.shape)
            a.append(f)
            #n.append(tf.where(tf.reduce_max(f)>0.9,onel,zerol))
        b.append(a)
        #t.append(n)
    c=tf.reshape(b,[batch_size,batch_size])
    #d=tf.reshape(t,[batch_size,batch_size])

    return c
    #余弦相似度矩阵

def euclidean_distance(x,y):#欧氏距离，两个向量之间的距离度量，表示向量之间的几何距离
    '''
    Computes the euclidean distances between x and y
    '''
    return K.sqrt(K.maximum(K.sum(K.square(x - y), keepdims=True), K.epsilon()))

def contrastive_loss(y,d): #对比损失函数:通过最小化对比损失来优化模型，使得同类样本在嵌入空间中更加接近，不同类样本更加分散
    tmp = y*tf.square(d)#同类样本，y=1 这部分损失希望同类样本的距离较小
    tmp2= (1-y)*tf.square(tf.maximum((1-d),0)) #不同类样本，y=0 这部分损失希望不同类样本的距离较大
    return 0.5*tf.reduce_sum(tmp+tmp2)

