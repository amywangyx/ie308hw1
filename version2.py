# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 01:58:53 2019

@author: Amy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:09:28 2019

@author: Amy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt # plotting

fields = ['NPI','NPPES_PROVIDER_ZIP','NPPES_PROVIDER_STATE','HCPCS_CODE',
 'LINE_SRVC_CNT',
 'BENE_UNIQUE_CNT',
 'AVERAGE_MEDICARE_ALLOWED_AMT','AVERAGE_MEDICARE_PAYMENT_AMT', 
 'AVERAGE_SUBMITTED_CHRG_AMT']


num_lines = sum(1 for line in open('C:\\Users\\Amy\\Downloads\\308dataset\\Medicare_Provider_Util_Payment_PUF_CY2016.txt'))
print (num_lines)

skiplines = np.random.choice(np.arange(1, 9714898), size=9714898-1-1000000, replace=False)
skiplines=np.sort(skiplines)
df = pd.read_csv('C:\\Users\\Amy\\Downloads\\308dataset\\Medicare_Provider_Util_Payment_PUF_CY2016.txt',sep='\t',low_memory=False,usecols=fields,skiprows=skiplines)

missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']


df['cover_ratio']=df['AVERAGE_MEDICARE_ALLOWED_AMT']/df['AVERAGE_SUBMITTED_CHRG_AMT']
df=df.dropna()

df.head(10)

print("Shape of loaded data: ", df.shape)
corrdf=df.corr()


#Check normality, then log, then check normality again
plt.hist(df['AVERAGE_SUBMITTED_CHRG_AMT'],range=[0,5000])
plt.show()

plt.hist(df['cover_ratio'])
plt.show()

plt.scatter(df['AVERAGE_SUBMITTED_CHRG_AMT'],df['cover_ratio'],s=0.5)
plt.show()
#,range=[-3000,50000]

#remove outlier for total_submitted_charge_amt
print(df['AVERAGE_SUBMITTED_CHRG_AMT'].min())
print(df['AVERAGE_SUBMITTED_CHRG_AMT'].max())
df_no_outlier = df[np.abs(df['AVERAGE_SUBMITTED_CHRG_AMT']) - (df['AVERAGE_SUBMITTED_CHRG_AMT'].mean()) <= (3*df['AVERAGE_SUBMITTED_CHRG_AMT'].std())]
df_no_outlier = df_no_outlier.reset_index(drop=True)
print(df_no_outlier['AVERAGE_SUBMITTED_CHRG_AMT'].min())
print(df_no_outlier['AVERAGE_SUBMITTED_CHRG_AMT'].max())


df_no_outlier['log_ave_submit_charge'] = np.log2(df_no_outlier['AVERAGE_SUBMITTED_CHRG_AMT'])
plt.hist(df_no_outlier['log_ave_submit_charge'])
plt.show()

log_ave_submit_charge_ave = np.mean(df_no_outlier['log_ave_submit_charge'])
log_ave_submit_charge_std = np.std(df_no_outlier['log_ave_submit_charge'])

df_no_outlier['log_ave_submit_charge_standardized'] = (df_no_outlier['log_ave_submit_charge'] - log_ave_submit_charge_ave) / log_ave_submit_charge_std


plt.scatter(df_no_outlier['log_ave_submit_charge_standardized'],df_no_outlier['cover_ratio'])
plt.show()

dftrain=df_no_outlier[['log_ave_submit_charge_standardized','cover_ratio']]
cluster_quality = pd.DataFrame(columns = ['inertia', 'number of clusters'])
for i in range(2,10):
    kmeans = KMeans(n_clusters = i)
    kmeans = kmeans.fit(dftrain)
    cluster_quality.loc[i-2] = [kmeans.inertia_,i]
cluster_quality.plot(x='number of clusters',y='inertia',title='Scree Plot')


kmeans = KMeans(n_clusters = 5)
kmeans = kmeans.fit(dftrain)
labels = kmeans.predict(dftrain)
centroids = kmeans.cluster_centers_
lab=kmeans.labels_

#silhouette score
df_with_labels = pd.DataFrame(dftrain).copy()
df_with_labels['labels'] = labels
sampled_data = df_with_labels.sample(n=10000)
silhouette_avg = silhouette_score(sampled_data.drop('labels',axis=1),sampled_data['labels'])
print('The average silhouette score is ' + str(silhouette_avg))

colors=['yellow','red','green','blue','purple']
plt.scatter(df_with_labels['log_ave_submit_charge_standardized'],df_with_labels['cover_ratio'],c=lab,cmap=matplotlib.colors.ListedColormap(colors),s=0.5)
plt.show()

# load with map, or include more feature 
result = pd.concat([df_no_outlier,df_with_labels],axis=1,join='inner')
result['final_zip']=result['NPPES_PROVIDER_ZIP'].str[:5]
resultcopy=result.copy()
del resultcopy['NPPES_PROVIDER_ZIP']
del resultcopy['LINE_SRVC_CNT']
del resultcopy['BENE_UNIQUE_CNT']
del resultcopy['AVERAGE_MEDICARE_ALLOWED_AMT']
del resultcopy['AVERAGE_MEDICARE_PAYMENT_AMT']

plt.scatter(resultcopy['AVERAGE_SUBMITTED_CHRG_AMT'],resultcopy.iloc[:,4],c=lab,cmap='viridis',s=0.5)
plt.show()
