# The code to save the data from matlab files and compile 
# 
# %%
import pandas as pd
import numpy as np

from pathlib import Path
import glob

from loadmat import loadmat

from scipy import stats

from outliers import smirnov_grubbs as grubbs

import seaborn as sns

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler

# %%
p = Path('.')
data_filepath = list(p.glob('./MATLAB/Test/*.mat'))
fname=[]

# for index, csv in enumerate(data_filepath):
#     fname.append(data_filepath[index].name)
#     # data_dict = { i : loadmat(i) for i in fname }


# Variables that we will use constantly
dispang = range(-90,95,5)
colorPattern = ['r','g','b']

data_dict = { data_filepath[index].name : loadmat(data_filepath[index]) 
    for index, csv in enumerate(data_filepath) }

testl = []

testl = np.array(data_dict['S1_45.mat']['Sx']['angle10']['S0'])

(m,n,l) = testl.shape
# test_r = (testl[(int(m/2))-10:(int(m/2))+10,(int(n/2))-10:(int(n/2))+10,0]).flatten()
# %% get dataset of less values

test_r = (testl[(int(m/2))-10:(int(m/2))+10,(int(n/2))-5:(int(n/2))+5,0])
outliersout = (test_r - test_r.mean(axis=0)) / test_r.std(axis=0)
x_normed = (test_r[:,0] - test_r[:,0].min(0)) / test_r[:,0].ptp(0)
out_normed = (outliersout - outliersout.min(0)) / outliersout.ptp(0)

# transpose = test_r.transpose()
# x = QuantileTransformer(output_distribution='uniform').fit_transform(test_r)
# test_r0 = test_r[:,0].reshape(-1, 1)

# y = QuantileTransformer(output_distribution='normal').fit_transform(test_r0)

# %% box plot


# data_dict = { data_filepath[index].name : loadmat(data_filepath[index]) for index, csv in enumerate(data_filepath) }
sns.boxplot(data=outliersout).set_title('outliers')
#%%
sns.boxplot(data=out_normed).set_title('out normalized')
#%%
sns.boxplot(data=test_r).set_title('original')
#%%
sns.boxplot(data=x_normed).set_title('original normilized')
# sns.boxplot(data=x_normed)

# %%
remOut = grubbs.test(test_r[:,2], alpha=0.05)
x = []
for column in test_r.T:
    print(column)
    x.append(grubbs.test(column, alpha=0.5))
y = np.array(x)
#
# yout = (y - y.mean(axis=0)) / y.std(axis=0)    
# y_normed = (y - y.min(0)) / y.ptp(0)


#%%
sns.boxplot(data=remOut).set_title('remove outliers')

#%%
sns.boxplot(data=y).set_title('quartile outliers')


#%%
