#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install similaritymeasures
#pip install tslearn


# In[ ]:


import glob
import numpy as np
import time
import math
import random
from scipy import linalg as LA
import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import similaritymeasures
import tslearn
from tslearn.metrics import dtw
from google.colab import files
from sklearn.decomposition import PCA


# # Generating Trajectories (50-50 by reversed direction)

# # Generating Trajectories 

# In[ ]:


n = 100
B = [0] * (2*n)

for i in range(n):
    B[i] = [[random.uniform(-1,1), random.uniform(-1, 1)]]
    
for i in range(n):
    for j in range(49):
        B[i].append([random.uniform(j,j+1), random.uniform(0, 5)])
    for j in range(49,n-2):
        B[i].append([random.uniform(n-j-3,n-j-2), random.uniform(-5, 0)])
    B[i].append([random.uniform(-1,1), random.uniform(-1, 1)])
    B[i] = np.array(B[i])
    
for i in range(n, 2*n):
    B[i] = [[random.uniform(-1,1), random.uniform(-1, 1)]]
    
for i in range(n, 2*n):
    for j in range(49):
        B[i].append([random.uniform(j,j+1), random.uniform(-5, 0)])
    for j in range(49,n-2):
        B[i].append([random.uniform(n-j-3,n-j-2), random.uniform(5, 0)])
    B[i].append([random.uniform(-1,1), random.uniform(-1, 1)])
    B[i] = np.array(B[i])
    
B = np.array(B)
data = B


# # Metric

# In[ ]:


def dist_signed_point_closed(Q, gamma, sigma): 
    
    p1 = gamma[:-1]
    p2 = gamma[1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1)) + 10e-6
    
    w = (p1-p2)*(-1,1)/(L * np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
# signed distance to the extended lines of segments
    dist_signed = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    x = abs(dist_signed.copy())
    R = (L**2).reshape(-1,1)
# u = argmin points on the extended lines of segments
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)

    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
# d1 = distance to start points
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
# d2 = distance to end points
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))
    d = np.where(d1 < d2, d1, d2)
    dist_segment = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-6), dist_signed, d)
    
    J2 = [0] * len(Q)
    for i in range(len(Q)): 
        J2[i] = np.where(abs(G + H - L)[i] > 10e-6)[0]
    J2 = np.array(J2)

    dist_segment_copy = dist_segment.copy()
    dist = abs(dist_segment_copy)


    j = np.argmin(dist, axis =1)

    sign = np.ones(len(Q))
    for k in range(len(Q)): 
        if j[k] in J2[k]:
            if j[k] == 0 and LA.norm(Q[k] - gamma[0]) < LA.norm(Q[k] - gamma[1]):
                
                y = LA.norm(gamma[0]-gamma[1]) - LA.norm(gamma[-1] - gamma[-2])
                if y < 0:
                    x = gamma[0] + 0.1 * LA.norm(gamma[0]-gamma[1])*(gamma[-2]-gamma[-1])/LA.norm(gamma[-2]-gamma[-1])
                    z = gamma[0] + 0.1 * LA.norm(gamma[0]-gamma[1])*(gamma[1]-gamma[0])/LA.norm(gamma[1]-gamma[0])
                    q = 2 * gamma[0] - (x + z)/2
                else: 
                    x = gamma[0] + 0.1 * LA.norm(gamma[-1]-gamma[-2])*(gamma[1]-gamma[0])
                    z = gamma[0] + 0.1 * LA.norm(gamma[-1]-gamma[-2])*(gamma[-2]-gamma[-1])
                    q = 2 * gamma[0] - (x + z)/2
                sign[k] = np.sign((q-gamma[-1]).dot(w[-1] + w[0]))
                
            elif j[k] == len(gamma)-2 and LA.norm(Q[k] - gamma[-1]) < LA.norm(Q[k] - gamma[-2]):
                s = w[-1].dot((Q[k] - gamma[-1])/ LA.norm(Q[k] - gamma[-1]) + 10e-6)
                sign[k] = np.sign(s)
            
            elif LA.norm(Q[k] - gamma[j[k]]) < LA.norm(Q[k] - gamma[j[k]+1]):  
                q = 2 * gamma[j[k]] - (gamma[j[k]-1] + gamma[j[k]+1])/2
                sign[k] = np.sign((q-gamma[j[k]]).dot(w[j[k]-1] + w[j[k]]))
                    
            elif LA.norm(Q[k] - gamma[j[k]+1]) <= LA.norm(Q[k] - gamma[j[k]]):
                q = 2 * gamma[j[k]+1] - (gamma[j[k]] + gamma[j[k]+2])/2
                sign[k] = np.sign((q-gamma[j[k]+1]).dot(w[j[k]] + w[j[k]+1]))

    E = dist_segment[np.arange(len(dist_segment)),j] 
    F = dist[np.arange(len(dist)),j] 
    dist_weighted = sign * (1/sigma) * (E.reshape(-1,1) * np.exp(-(F/sigma)**2).reshape(-1,1)).reshape(1,-1)

    return dist_weighted.reshape(len(Q))


# In[ ]:


def dist_signed_point_unclosed(Q, gamma, sigma): 
    
    p1 = gamma[:-1]
    p2 = gamma[1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1)) + 10e-6
    w = (p1-p2)*(-1,1)/(L * np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
# signed distance to the extended lines of segments
    dist_signed = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    x = abs(dist_signed.copy())
    R = (L**2).reshape(-1,1)
# u = argmin points on the extended lines of segments
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)

    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
# d1 = distance to start points
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
# d2 = distance to end points
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))
    d = np.where(d1 < d2, d1, d2)
    dist_segment = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-6), dist_signed, d)
    
    J2 = [0] * len(Q)
    for i in range(len(Q)): 
        J2[i] = np.where(abs(G + H - L)[i] > 10e-6)[0]
    J2 = np.array(J2)

    dist_segment_copy = dist_segment.copy()
    dist = abs(dist_segment_copy)
    
    dist_from_start_1 = np.sqrt(((Q -p1[0])*(Q -p1[0])).sum(axis =1))
    ds_1 = ((Q -p1[0])*w[0]).sum(axis =1)
    dist_from_start = ds_1 * np.maximum(abs(ds_1), np.sqrt(dist_from_start_1**2 - ds_1**2 + 10e-6))/ (dist_from_start_1 + 10e-6)


    dist_from_end_1 = np.sqrt(((Q -p2[-1])*(Q -p2[-1])).sum(axis =1))
    de_1 = ((Q -p2[-1])* w[-1]).sum(axis =1)
    dist_from_end = de_1 * np.maximum(abs(de_1), np.sqrt(dist_from_end_1**2 - de_1**2 + 10e-6))/ (dist_from_end_1+ 10e-6)

    dist_segment[:,0] = np.where(abs(dist[:,0]- dist_from_start_1)< 10e-8, dist_from_start, dist_segment[:,0]) 
    dist_segment[:,-1] = np.where(abs(dist[:,-1]- dist_from_end_1)< 10e-8, dist_from_end, dist_segment[:,-1]) 


    j = np.argmin(dist, axis =1)

    sign = np.ones(len(Q))
    for k in range(len(Q)): 
        if j[k] in J2[k]: 
            if j[k] == 0 and LA.norm(Q[k] - gamma[0]) < LA.norm(Q[k] - gamma[1]):
                sign[k] = 1
                
            elif j[k] == len(gamma)-2 and LA.norm(Q[k] - gamma[j[k]+1]) < LA.norm(Q[k] - gamma[j[k]]):
                sign[k] = 1
            
            elif LA.norm(Q[k] - gamma[j[k]]) < LA.norm(Q[k] - gamma[j[k]+1]):  
                q = 2 * gamma[j[k]] - (gamma[j[k]-1] + gamma[j[k]+1])/2
                sign[k] = np.sign((q-gamma[j[k]]).dot(w[j[k]-1] + w[j[k]]))
                    
            elif LA.norm(Q[k] - gamma[j[k]+1]) <= LA.norm(Q[k] - gamma[j[k]]) and j[k]+2 <=len(gamma)-1:
                q = 2 * gamma[j[k]+1] - (gamma[j[k]] + gamma[j[k]+2])/2
                sign[k] = np.sign((q-gamma[j[k]+1]).dot(w[j[k]] + w[j[k]+1]))

    E = dist_segment[np.arange(len(dist_segment)),j] 
    F = dist[np.arange(len(dist)),j] 
    dist_weighted = sign * (1/sigma) * (E.reshape(-1,1) * np.exp(-(F/sigma)**2).reshape(-1,1)).reshape(1,-1)

    return dist_weighted.reshape(len(Q))


# In[ ]:


def dist_signed_point(Q, gamma, sigma):
    if LA.norm(gamma[0]-gamma[-1]) > 10e-6:
        A = dist_signed_point_unclosed(Q, gamma, sigma)
    else: 
        A = dist_signed_point_closed(Q, gamma, sigma)
        
    return A


# # Choosing $Q$ and $\sigma$

# In[ ]:


m = 20
Q = np.ones((m,2))

for i in range(m):
    Q[i] = (random.uniform(-1, 51), random.uniform(-8, 8)) 
np.savetxt('Q.csv', Q, delimiter=',')


# In[ ]:


from google.colab import files
files.download('Q.csv')


# In[ ]:


sigma = 5


# #Mapping to $\mathbb{R}^m$ under $v_Q^{\sigma}$

# In[ ]:


projected_go = [0] * n
projected_back = [0] * n

for i in range(n):
    projected_go[i] = np.concatenate((dist_signed_point(Q,data[i],sigma),[1]), axis = 0)

for i in range(n):
    projected_back[i] = np.concatenate((dist_signed_point(Q,data[n+i],sigma),[-1]), axis = 0)
    
projected_go = np.array(projected_go)
projected_back = np.array(projected_back)


# # Classifiers

# In[ ]:


clf0 = KNeighborsClassifier(n_neighbors=5) 
clf1 = svm.SVC(kernel='linear') 
clf2 = make_pipeline(StandardScaler(), svm.SVC(C= 20000, kernel = 'rbf', gamma= 'auto', max_iter = 200000))
clf3 = make_pipeline(StandardScaler(), svm.SVC(C= 10000, kernel = 'rbf', gamma= 'auto', max_iter = 200000))
clf4 = make_pipeline(StandardScaler(), svm.SVC(C= 15000, kernel = 'rbf', gamma= 'auto', max_iter = 200000))
clf5 = make_pipeline(StandardScaler(), svm.SVC(C=100, kernel = 'poly', degree =3, max_iter = 400000))
clf6 = make_pipeline(StandardScaler(), svm.SVC(C=10000, kernel = 'poly', degree =2, max_iter = 400000))
clf7 = DecisionTreeClassifier()
clf8 = DecisionTreeClassifier(max_depth= 4)
clf9 = DecisionTreeClassifier(max_depth= 5)
clf10 = RandomForestClassifier(n_estimators=100, max_depth=5)
clf11 = RandomForestClassifier(n_estimators=100, max_depth=6)
clf12 = RandomForestClassifier(n_estimators=100, max_depth=7)
clf13 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5)
clf14 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=6)
clf15 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=7)
clf16 = AdaBoostClassifier(n_estimators= 100,learning_rate=0.95)
clf17 = AdaBoostClassifier(n_estimators= 100,learning_rate=1.05)

clf = [clf0,clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11,clf12,clf13,clf14,clf15,clf16,clf17]


# # Classification after using our feature map

# In[ ]:


X_1 = projected_go
X_2 = projected_back


# In[ ]:


Start_time = time.time()

t = 1000

error_train = np.zeros((len(clf), t))
error_test = np.zeros((len(clf), t))

for i in range(t): 

    R1 = random.sample(range(n), 30)
    R = np.sort(R1)
    R_c = np.sort(list(set(range(n)) - set(R)))
    
    S1 = random.sample(range(n), 30)
    S = np.sort(S1)
    S_c = np.sort(list(set(range(n)) - set(S)))
    
    data_train = np.insert(X_1[R_c], len(X_1[R_c]), X_2[S_c], axis = 0)
    data_test = np.insert(X_1[R], len(X_1[R]), X_2[S], axis = 0)
    
    data_train = list(data_train)
    data_test = list(data_test)
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    for k in range(len(clf)):
        
        model = clf[k]
        #Train the model using the training sets
        model.fit(data_train[:,:-1], data_train[:,-1])

        #Predict the response for test dataset
        y_pred = model.predict(data_test[:,:-1])
        error_test[k][i] = 1 - metrics.accuracy_score(data_test[:,-1], y_pred)
        
        x_pred = model.predict(data_train[:,:-1])
        error_train[k][i] = 1 - metrics.accuracy_score(data_train[:,-1], x_pred)
        
print('total time =', time.time() - Start_time)


# In[ ]:


print('|Q|=20', 'sigma=', sigma, ', t=', t, ',data = B')

Dic1 = {}

models = ["KNN", "Linear kernel SVM", "Gaussian SVM, C=1, gamma= auto", 
          "Gaussian SVM, C = 10, gamma = auto", "Gaussian SVM, C = 100, gamma = auto",
          "Poly kernel SVM, deg=3", "Poly kernel SVM, deg=2, C=100", 
          "Decision Tree", "Decision Tree, depth=3", "Decision Tree, depth=4",
          'RF, gini, max_depth=5, 100 estimators',
          'RF, gini, max_depth=6, 100 estimators',
          'RF, gini, max_depth=7, 100 estimators', 
          'RF, entropy, max_depth=5, 100 estimators',
          'RF, entropy, max_depth=6, 100 estimators', 
          'RF, entropy, max_depth=7, 100 estimators', 
          'AdaBoost, learning rate=0.95, 100 estimators',
          'AdaBoost, learning rate=1.05, 100 estimators']

for k in range(len(models)): 
    Dic1[k+1] = [models[k], np.round(np.mean(error_train[k]), decimals = 4), 
                np.round(np.mean(error_test[k]), decimals = 4),
               np.round(np.std(error_test[k]), decimals = 4)]
    
df2 = pd.DataFrame.from_dict(Dic1, orient='index', columns=['Classifier','Train Error', 
                                        'Test Error', 'Standard Deviation'])
df2


# # Classification with old distance

# In[2]:


def old_dist(Q, gamma):
    
    p2 = gamma[1:]
    p1 = gamma[:-1]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    II = np.where(L>10e-8)[0]
    L = L[II]
    p1 = p1[II]
    p2 = p2[II]
    w = (p1-p2)*(-1,1)/(L*np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
    dist_dot = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    
    x = abs(dist_dot.copy())
    R = (L**2).reshape(-1,1)
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)
    
    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))

    dist = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-8), x, np.minimum(d1, d2))

    j = np.argmin(dist, axis =1)
    dists = dist[np.arange(len(dist)),j]
    
    return dists.reshape(len(Q)) 


# In[ ]:


proj_go = [0] * n
proj_back = [0] * n

for i in range(n):
    proj_go[i] = np.concatenate((old_dist(Q,data[i]),[1]), axis = 0)

for i in range(n):
    proj_back[i] = np.concatenate((old_dist(Q,data[n+i]),[-1]), axis = 0)
    
proj_go = np.array(proj_go)
proj_back = np.array(proj_back)


# #Classification

# In[ ]:


X_1 = proj_go
X_2 = proj_back


# In[ ]:


Start_time = time.time()

t = 1000

error_train_list = np.zeros((len(clf), t))
error_test_list = np.zeros((len(clf), t))

for i in range(t): 

    R1 = random.sample(range(n), 30)
    R = np.sort(R1)
    R_c = np.sort(list(set(range(n)) - set(R)))
    
    S1 = random.sample(range(n), 30)
    S = np.sort(S1)
    S_c = np.sort(list(set(range(n)) - set(S)))
    
    data_train = np.insert(X_1[R_c], len(X_1[R_c]), X_2[S_c], axis = 0)
    data_test = np.insert(X_1[R], len(X_1[R]), X_2[S], axis = 0)
    
    data_train = list(data_train)
    data_test = list(data_test)
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    for k in range(len(clf)):
        
        model = clf[k]
        #Train the model using the training sets
        model.fit(data_train[:,:-1], data_train[:,-1])

        #Predict the response for test dataset
        y_pred = model.predict(data_test[:,:-1])
        error_test_list[k][i] = 1 - metrics.accuracy_score(data_test[:,-1], y_pred)
        
        x_pred = model.predict(data_train[:,:-1])
        error_train_list[k][i] = 1 - metrics.accuracy_score(data_train[:,-1], x_pred)
        
print('total time =', time.time() - Start_time)


# In[ ]:


print('|Q|=20,', ' t=', t, ',data = B,', 'Old Distance')

Dic2 = {}

models = ["KNN", "Linear kernel SVM", "Gaussian SVM, C=1, gamma= auto", 
          "Gaussian SVM, C = 10, gamma = auto", "Gaussian SVM, C = 100, gamma = auto",
          "Poly kernel SVM, deg=3", "Poly kernel SVM, deg=2, C=100", 
          "Decision Tree", "Decision Tree, depth=3", "Decision Tree, depth=4",
          'RF, gini, max_depth=5, 100 estimators',
          'RF, gini, max_depth=6, 100 estimators',
          'RF, gini, max_depth=7, 100 estimators', 
          'RF, entropy, max_depth=5, 100 estimators',
          'RF, entropy, max_depth=6, 100 estimators', 
          'RF, entropy, max_depth=7, 100 estimators', 
          'AdaBoost, learning rate=0.95, 100 estimators',
          'AdaBoost, learning rate=1.05, 100 estimators']

for k in range(len(models)): 
    Dic2[k+1] = [models[k], np.round(np.mean(error_train_list[k]), decimals = 4), 
                np.round(np.mean(error_test_list[k]), decimals = 4),
               np.round(np.std(error_test_list[k]), decimals = 4)]
    
df2 = pd.DataFrame.from_dict(Dic2, orient='index', columns=['Classifier','Train Error', 
                                        'Test Error', 'Standard Deviation'])
df2


# In[ ]:


E3 = df3.to_latex(index=False)
np.savetxt('Old_feature_map_classification_results.tex', [E3], fmt='%s')
files.download('Old_feature_map_classification_results.tex')


# # KNN with $d_Q^{\sigma}$, $d_Q$, dtw, Frechet

# ## KNN with $d_Q^{\sigma}$

# In[ ]:


Start_time = time.time()

t = 100

error_train_d_Q_sigma = np.zeros(t)
error_test_d_Q_sigma = np.zeros(t)

for i in range(t): 

    R1 = random.sample(range(n), 30)
    R = np.sort(R1)
    R_c = np.sort(list(set(range(n)) - set(R)))
    
    S1 = random.sample(range(n), 30)
    S = np.sort(S1)
    S_c = np.sort(list(set(range(n)) - set(S)))
    
    data_train = np.insert(projected_go[R_c], len(projected_go[R_c]), projected_back[S_c], axis = 0)
    data_test = np.insert(projected_go[R], len(projected_go[R]), projected_back[S], axis = 0)
    
    data_train = list(data_train)
    data_test = list(data_test)
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
        #Train the model using the training sets
    model.fit(data_train[:,:-1], data_train[:,-1])

        #Predict the response for test dataset
    y_pred = model.predict(data_test[:,:-1])
    error_test_d_Q_sigma[i] = 1 - metrics.accuracy_score(data_test[:,-1], y_pred)
        
    x_pred = model.predict(data_train[:,:-1])
    error_train_d_Q_sigma[i] = 1 - metrics.accuracy_score(data_train[:,-1], x_pred)
        
print('total time =', time.time() - Start_time)


# In[ ]:


print(np.mean(error_train_d_Q_sigma), np.median(error_train_d_Q_sigma), np.std(error_train_d_Q_sigma))
print(np.mean(error_test_d_Q_sigma),np.median(error_test_d_Q_sigma), np.std(error_test_d_Q_sigma))


# # KNN with $d_Q$

# In[ ]:


Start_time = time.time()

t = 10

error_train_d_Q = np.zeros(t)
error_test_d_Q = np.zeros(t)

for i in range(t): 

    R1 = random.sample(range(n), 30)
    R = np.sort(R1)
    R_c = np.sort(list(set(range(n)) - set(R)))
    
    S1 = random.sample(range(n), 30)
    S = np.sort(S1)
    S_c = np.sort(list(set(range(n)) - set(S)))
    
    data_train = np.insert(proj_go[R_c], len(proj_go[R_c]), proj_back[S_c], axis = 0)
    data_test = np.insert(proj_go[R], len(proj_go[R]), proj_back[S], axis = 0)
    
    data_train = list(data_train)
    data_test = list(data_test)
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
        #Train the model using the training sets
    model.fit(data_train[:,:-1], data_train[:,-1])

        #Predict the response for test dataset
    y_pred = model.predict(data_test[:,:-1])
    error_test_d_Q[i] = 1 - metrics.accuracy_score(data_test[:,-1], y_pred)
        
    x_pred = model.predict(data_train[:,:-1])
    error_train_d_Q[i] = 1 - metrics.accuracy_score(data_train[:,-1], x_pred)
        
print('total time =', time.time() - Start_time)


# In[ ]:


print(np.mean(error_train_d_Q), np.median(error_train_d_Q), np.std(error_train_d_Q))
print(np.mean(error_test_d_Q),np.median(error_test_d_Q), np.std(error_test_d_Q))


# # KNN with dtw

# In[ ]:


def func1(a,b):
    c = np.zeros(len(b))
    for i in range(len(b)):
        c[i] = tslearn.metrics.dtw(a,b[i])
    return c


# In[ ]:


A = data[:100]
B = data[100:]
E = [0] * len(A)
F = [0] * len(B)

for i in range(len(A)):
    E[i] = np.concatenate((A[i],[[1,1]]), axis = 0)

for i in range(len(B)):
    F[i] = np.concatenate((B[i],[[-1,-1]]), axis = 0)
    
E = np.array(E)
F = np.array(F)


# In[ ]:


Start_time = time.time()

t = 10

error_train_dtw = np.zeros(t)
error_test_dtw = np.zeros(t)

for i in range(t): 

    R1 = random.sample(range(n), 30)
    R = np.sort(R1)
    R_c = np.sort(list(set(range(n)) - set(R)))
    
    S1 = random.sample(range(n), 30)
    S = np.sort(S1)
    S_c = np.sort(list(set(range(n)) - set(S)))
        
    data_train = np.insert(E[R_c], len(E[R_c]), F[S_c], axis = 0)
    data_test = np.insert(E[R], len(E[R]), F[S], axis = 0)
    data_train = list(data_train)
    data_test = list(data_test)
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    D_train = np.zeros((len(data_train),len(data_train)))

    for k in range(len(data_train)):
        D_train[k] = func1(data_train[k,:-1], data_train[:,:-1])
    
    D_test = np.zeros((len(data_test),len(data_train))) 
    
    for j in range(len(data_test)):
        D_test[j] = func1(data_test[j,:-1], data_train[:,:-1])
    
    model = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    #Train the model using the training sets

    model.fit(D_train, data_train[:,-1][:,0])
    
    #Predict the response for test dataset
    y_pred = model.predict(D_test)
    error_test_dtw[i] = 1 - metrics.accuracy_score(data_test[:,-1][:,0], y_pred)
    #error_test_dtw[i] = np.sum(abs(y_pred - data_test[:,-1][:,0]))/len(data_test)
        
    x_pred = model.predict(D_train)
    error_train_dtw[i] = 1 - metrics.accuracy_score(data_train[:,-1][:,0], x_pred)

print('total time =', time.time() - Start_time)


# In[ ]:


print(np.mean(error_train_dtw), np.median(error_train_dtw), np.std(error_train_dtw))
print(np.mean(error_test_dtw),np.median(error_test_dtw), np.std(error_test_dtw))


# # KNN with Frechet

# In[ ]:


# run time is very high: I ran for 20 and 6 instead of 100 and 30
Start_time = time.time()

t = 1

error_train_fr = np.zeros(t)
error_test_fr = np.zeros(t)

for i in range(t): 

    R1 = random.sample(range(20), 6)
    R = np.sort(R1)
    R_c = np.sort(list(set(range(20)) - set(R)))
    
    S1 = random.sample(range(20), 6)
    S = np.sort(S1)
    S_c = np.sort(list(set(range(20)) - set(S)))
    
    data_train = np.insert(E[R_c], len(E[R_c]), F[S_c], axis = 0)
    data_test = np.insert(E[R], len(E[R]), F[S], axis = 0)
    data_train = list(data_train)
    data_test = list(data_test)
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    D_train = np.zeros((len(data_train),len(data_train)))

    for k in range(len(data_train)-1):
        for s in range(k+1, len(data_train)):
            D_train[k][s] = similaritymeasures.frechet_dist(data_train[k,:-1], data_train[s,:-1])
            D_train[s][k] = D_train[k][s]
    
    D_test = np.zeros((len(data_test),len(data_train))) 
    
    for j in range(len(data_test)):
        for u in range(len(data_train)):
            D_test[j][u] = similaritymeasures.frechet_dist(data_test[j,:-1], data_train[u,:-1])
    
    model = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    #Train the model using the training sets

    model.fit(D_train, data_train[:,-1][:,0])
    
    #Predict the response for test dataset
    y_pred = model.predict(D_test)
    error_test_fr[i] = 1 - metrics.accuracy_score(data_test[:,-1][:,0], y_pred)
        
    x_pred = model.predict(D_train)
    error_train_fr[i] = 1 - metrics.accuracy_score(data_train[:,-1][:,0], x_pred)
        
print('total time =', time.time() - Start_time)


# In[ ]:


print(np.mean(error_train_fr), np.median(error_train_fr), np.std(error_train_fr))
print(np.mean(error_test_fr), np.median(error_test_fr), np.std(error_test_fr))


# # Presenting results in a dataframe and Latex output

# In[ ]:


print('|Q|=20,', 't=10,', '|data|= 100 + 100= 200,', 'data = B')

Dic3 = {}

Dic3[1] = ["$d_Q^{\sigma}$ distance", np.round(np.mean(error_train_d_Q_sigma), decimals = 4), 
                np.round(np.mean(error_test_d_Q_sigma), decimals = 4),
                np.round(np.median(error_test_d_Q_sigma), decimals = 4),
               np.round(np.std(error_test_d_Q_sigma), decimals = 4)]

Dic3[2] = ["DTW distance", np.round(np.mean(error_train_dtw), decimals = 4), 
                np.round(np.mean(error_test_dtw), decimals = 4),
                np.round(np.median(error_test_dtw), decimals = 4),
               np.round(np.std(error_test_dtw), decimals = 4)]

Dic3[3] = ["Frechet distance", np.round(np.mean(error_train_fr), decimals = 4), 
                np.round(np.mean(error_test_fr), decimals = 4),
                np.round(np.median(error_test_fr), decimals = 4),
               np.round(np.std(error_test_fr), decimals = 4)]


Dic3[4] = ["$d_Q$ distance", np.round(np.mean(error_train_d_Q), decimals = 4), 
                np.round(np.mean(error_test_d_Q), decimals = 4),
                np.round(np.median(error_test_d_Q), decimals = 4),
               np.round(np.std(error_test_d_Q), decimals = 4)]

df4 = pd.DataFrame.from_dict(Dic3, orient='index', columns=['Distance','Train Error', 
                                        'Test Error', 'Median Error', 'Standar Deviation'])
df4


# In[ ]:


E4 = df4.to_latex(index=False)
np.savetxt('table-KNN.tex', [E4], fmt='%s')
files.download('table-KNN.tex')

