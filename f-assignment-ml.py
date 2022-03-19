#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>
# 

# In this notebook we try to practice all the classification algorithms that we have learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[2]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[4]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[5]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[6]:


df.shape


# In[7]:


df.dtypes


# ### Convert to date time object
# 

# In[8]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[9]:


df.dtypes


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[14]:


df['loan_status'].value_counts()


# In[11]:


df["Gender"].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[12]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[15]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[16]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[17]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[18]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[22]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[23]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[24]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[25]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[26]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[31]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[32]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[34]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# In[38]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.2 , random_state=4)


# In[39]:


print("Shape of xtrain:",xtrain.shape,"; and shape of xtest:",xtest.shape)
print("Shape of ytrain:",ytrain.shape,"; and shape of ytest:",ytest.shape)


# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[64]:


k = 20
mean_acc = np.zeros((k-1))
std_acc = np.zeros((k-1))

for n in range(1,k):
    KNN = KNeighborsClassifier(n_neighbors = n).fit(xtrain,ytrain)
    KNNy=KNN.predict(xtest)
    mean_acc[n-1] = metrics.accuracy_score(ytest, KNNy)

mean_acc


# In[101]:


print("In a range of k:[1;20] the max KNN accuracy is",mean_acc.max(),"with a k =",mean_acc.argmax()+1)


# In[124]:


KNN7 = KNeighborsClassifier(n_neighbors = 7).fit(xtrain,ytrain)
KNN7y = KNN7.predict(xtest)
KNN7y


# In[125]:


print("The accuracy score from ytrain and KNN predict is",
      metrics.accuracy_score(ytrain,KNN7.predict(xtrain)),
      "; and accuracy score from ytest and KNN predict",
      metrics.accuracy_score(ytest,KNN7y))


# # Decision Tree
# 

# In[66]:


from sklearn.tree import DecisionTreeClassifier


# In[84]:


TREE = DecisionTreeClassifier(criterion="entropy",max_depth=4)
TREE.fit(xtrain,ytrain)
TREEy = TREE.predict(xtest)
TREEy


# In[86]:


print("The accuracy score from ytrain and tree predict is",
      metrics.accuracy_score(ytrain,TREE.predict(xtrain)),
      "; and accuracy score from ytest and tree predict",
      metrics.accuracy_score(ytest,TREEy))


# # Support Vector Machine
# 

# In[88]:


from sklearn import svm


# In[99]:


krnl="rbf"
SUP = svm.SVC(kernel=krnl)
SUP.fit(xtrain,ytrain)
SUPy = SUP.predict(xtest)
SUPy


# In[100]:


print("The accuracy score from ytrain and SVM predict is",
      metrics.accuracy_score(ytrain,SUP.predict(xtrain)),
      "; and accuracy score from ytest and SVM predict",
      metrics.accuracy_score(ytest,SUPy))


# # Logistic Regression
# 

# In[102]:


from sklearn.linear_model import LogisticRegression


# In[107]:


LRE = LogisticRegression(C=0.01,solver="liblinear")
LRE.fit(xtrain,ytrain)
LREy = LRE.predict(xtest)
LREy


# In[106]:


print("The accuracy score from ytrain and LRE predict is",
      metrics.accuracy_score(ytrain,LRE.predict(xtrain)),
      "; and accuracy score from ytest and LRE predict",
      metrics.accuracy_score(ytest,LREy))


# # Model Evaluation using Test set
# 

# In[108]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:
# 

# In[109]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation
# 

# In[110]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[120]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_Feature.head()


# In[122]:


x_test = test_Feature
x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
x_test[0:5]


# In[123]:


y_test = test_df['loan_status'].values
y_test[0:5]


# # Model evaluation: KNN

# In[140]:


MEKy = KNN7.predict(x_test)
MEKp = KNN7.predict_proba(x_test)


# In[133]:


jaccard_score(y_test, MEKy,pos_label="PAIDOFF")


# In[137]:


f1_score(y_test,MEKy,average="weighted")


# In[141]:


log_loss(y_test, MEKp)


# # Model evaluation: Decision Tree

# In[156]:


TREEy = TREE.predict(x_test)
TREEp = TREE.predict_proba(x_test)


# In[146]:


jaccard_score(y_test, TREEy,pos_label="PAIDOFF")


# In[147]:


f1_score(y_test,TREEy,average="weighted")


# In[157]:


log_loss(y_test,TREEp)


# # Model evaluation: SVM

# In[155]:


SUPy = SUP.predict(x_test)


# In[150]:


jaccard_score(y_test,SUPy,pos_label="PAIDOFF")


# In[151]:


f1_score(y_test,SUPy,average="weighted")


# # Model evaluation: Logistic Regression

# In[153]:


LREy = LRE.predict(x_test)
LREp = LRE.predict_proba(x_test)


# In[158]:


jaccard_score(y_test,LREy,pos_label="PAIDOFF")


# In[159]:


f1_score(y_test,LREy,average="weighted")


# In[160]:


log_loss(y_test,LREp)


# # Report
# 
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# | Algorithm          | Jaccard | F1-score | LogLoss |
# | ------------------ | ------- | -------- | ------- |
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |
# 

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                                                             |
# | ----------------- | ------- | ------------- | ------------------------------------------------------------------------------ |
# | 2020-10-27        | 2.1     | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |
# | 2020-08-27        | 2.0     | Malika Singla | Added lab to GitLab                                                            |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <p>
# 
