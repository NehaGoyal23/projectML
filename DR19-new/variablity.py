
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Components = 30
dff = pd.read_csv('Gear18SNE30.csv')
dff = dff.dropna(axis = 1, how ='all') 
#removing columns havin more than 90% zero
dff=dff.drop(columns=dff.columns[((dff==0).mean()>0.90)],axis=1)
X = dff.iloc[:, :-1].values
y = dff.iloc[:, -1].values
from scipy.stats import variation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_t = X_train.transpose()
COV = np.matmul(X_t,X_train)
from scipy import linalg as LA
e_vals, e_vecs = LA.eig(COV)
i=1
for item in e_vals:
    Variability=item/sum(e_vals)
    
    print(" " + str(i) + " - " + str(Variability)[1:8])
    i = i + 1