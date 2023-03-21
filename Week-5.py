#!/usr/bin/env python
# coding: utf-8

# ## Model building from scratch

# by R.Ramu 
# 
# ML Engineer

# ## Business Problem
# 
# The business problem for telecom churn is to predict which customers are likely to churn or switch to another service provider, so that proactive measures can be taken to retain those customers and reduce customer churn.

# ## Attributes description

# 
# 
# CustomerID: A unique ID that identifies each customer.
# 
# Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.
# 
# Gender: The customer’s gender: Male, Female
# 
# Senior Citizen: Indicates if the customer is 65 or older: Yes, No
# 
# Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
# 
# Country: The country of the customer’s primary residence.
# 
# State: The state of the customer’s primary residence.
# 
# City: The city of the customer’s primary residence.
# 
# Lat Long: The combined latitude and longitude of the customer’s primary residence.
# 
# Latitude: The latitude of the customer’s primary residence.
# 
# Longitude: The longitude of the customer’s primary residence.
# 
# Zip Code: The zip code of the customer’s primary residence.
# 
# Tenure in Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
# 
# Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
# 
# Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
# 
# Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
# 
# Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
# 
# Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
# 
# Device Protection Plan: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
# 
# Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
# 
# Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
# 
# Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
# 
# Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
# 
# Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
# 
# Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
# 
# Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
# 
# Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.    
# 
# Customer Status: Indicates the status of the customer at the end of the quarter: Churned, Stayed, or Joined
# 
# Churn Label: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.
# 
# Churn Value: 1 = the customer left the company this quarter. 0 = the customer remained with the company. Directly related to Churn Label.
# 
# Churn Score: A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
# 
# CLTV: Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.
# 
# Churn Reason: A customer’s specific reason for leaving the company. Directly related to Churn Category.

# # Importing the required libraries

# In[1]:


import pandas as pd
import numpy as np


# # Reading the datasets

# In[2]:


churn = pd.read_csv("churn_tel.csv")
tel = pd.read_excel("tel_churn.xlsx")


# In[3]:


churn.head()


# In[4]:


tel.head()


# In[5]:


churn.rename(columns={'customerID':'customerid'}, inplace=True)   # Renaming the customerid


# In[6]:


tel.columns = tel.columns.str.lower()   # converting the uppercase to lowercase


# # Merging the datasets

# In[7]:


tel_churn = pd.merge(churn,tel,on =['customerid'])


# In[8]:


tel_churn.head()


# # Basic preprosessing

# ## Droping the feastures

# In[12]:


tel_churn.drop(['churn label','Churn'],inplace = True ,axis = 1)


# In[13]:


tel_churn.drop('churn reason',inplace=True,axis = 1)


# In[14]:


tel_churn.drop(['zip code','city','state','country','count'],inplace = True,axis = 1)


# In[15]:


tel_churn.drop('customerid',inplace=True,axis=1)


# In[16]:


tel_churn.drop(['gender_y', 'senior citizen', 'partner', 'dependents',
       'tenure months', 'phone service', 'multiple lines', 'internet service',
       'online security', 'online backup', 'device protection', 'tech support',
       'streaming tv', 'streaming movies', 'contract', 'paperless billing',
       'payment method', 'monthly charges', 'total charges'],inplace = True,axis =1)


# # Data Type conversion

# In[17]:


for i in ['gender_x','Partner','Dependents','PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','PaperlessBilling','PaymentMethod',
          'SeniorCitizen','churn value','Contract','TechSupport','StreamingTV', 'StreamingMovies']:
      tel_churn[i]=tel_churn[i].astype('category')


# In[18]:


tel_churn.dtypes


# ### I have found that some of the datapoints in the 'TotalCharges' are having  ' ' these commas , so replacing them with None 

# In[19]:


count=0
count=tel_churn['TotalCharges']==' '


# In[20]:


l=0
for i in count:
    if i==True:
        l=l+1
l        


# In[21]:


tel_churn['TotalCharges'].replace('',None,inplace=True)


# In[22]:


tel_churn['TotalCharges'].replace(' ',None,inplace=True)


# In[23]:


tel_churn['TotalCharges'].isnull().sum()


# ## Filling the None values using median

# In[24]:


tel_churn['TotalCharges'] = tel_churn['TotalCharges'].fillna(tel_churn['TotalCharges'].median())


# In[25]:


tel_churn.dtypes


# In[26]:


tel_churn.describe()


# ## Checking the null values

# In[27]:


tel_churn.isnull().sum()


# #### Unique value counts

# In[28]:


tel_churn.nunique()


# In[29]:


tel_churn['TotalCharges'].unique()


# In[30]:


tel_churn['TotalCharges']=tel_churn['TotalCharges'].astype('float64')


# ### Basic Visualizations and Analysing the data

# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[32]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(tel_churn['tenure'], ax=ax[0])
sns.boxplot(tel_churn['tenure'], ax=ax[1])
ax[1].set_title("Box plot of 'Tenure' Feature")
ax[0].set_title("Distibution plot of 'Tenure' Feature")
fig.show()


# In[33]:


ctab = pd.crosstab(index=tel_churn['InternetService'], columns=tel_churn['churn value'])
ctab.plot.bar(figsize=(10,8), xlabel='InternetService')


# In[34]:


ctab = pd.crosstab(index=tel_churn['TechSupport'], columns=tel_churn['churn value'])
ctab.plot.bar(figsize=(10,8), xlabel='TechSupport')


# In[35]:


fig, ax = plt.subplots(figsize=(8, 8))
tel_churn.groupby('OnlineBackup').size().plot(kind='pie', autopct='%.2f')
plt.axis('equal')
plt.show()


# ### From the pie chart we can analyse that most of the customers are having the onlinebackup

# In[40]:


data = tel_churn['cltv']
fig = plt.figure(figsize =(10, 7))
plt.boxplot(data)
plt.show()


# In[42]:


data = tel_churn['tenure']
fig = plt.figure(figsize =(8, 5))
plt.boxplot(data)
plt.show()


# ### From the above box we can observe that there are no outliers

# In[46]:


data = tel_churn.groupby(['gender_x', 'Partner']).size().unstack()
fig, ax = plt.subplots()
data.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('gender_x')
ax.set_ylabel('Count')
ax.set_title('Telecom Churn Stacked Bar Chart')
ax.legend(title='Partner')


# In[36]:


tel_churn.drop('lat long',axis=1,inplace=True)


# In[37]:


tel_churn.dtypes


# ## Checking the correlation 

# In[38]:


corr = tel_churn.corr(method='spearman')
corr.style.background_gradient(cmap='coolwarm')


# In[47]:


X = tel_churn.drop('churn value',axis = 1)
y = tel_churn['churn value']


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, stratify=y, random_state=321) 


# ## Performing one hot encoding

# In[49]:


from sklearn.preprocessing import OneHotEncoder


# In[50]:


enc=OneHotEncoder(drop='first',handle_unknown='ignore')


# In[51]:


cat_cols=X_train.select_dtypes(['category','object']).columns
X_train_encoding=pd.DataFrame(enc.fit_transform(X_train[cat_cols]).toarray(),columns=enc.get_feature_names_out())
X_val_encoding=pd.DataFrame(enc.transform(X_val[cat_cols]).toarray(),columns=enc.get_feature_names_out())


# In[52]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
num_attr = X_train.select_dtypes(include=['int64','float64']).columns
num_attr
scaler = StandardScaler()
scaler.fit(X_train[num_attr])
X_train_std = pd.DataFrame(scaler.transform(X_train[num_attr]),columns=X_train[num_attr].columns)
X_val_std = pd.DataFrame(scaler.transform(X_val[num_attr]),columns=X_train[num_attr].columns)


# In[53]:


X_train = pd.concat([X_train_std, X_train_encoding],axis=1)
X_val = pd.concat([X_val_std, X_val_encoding], axis=1)


# In[54]:


X_train.shape


# ## Building the model from scratch

# In[55]:


import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)
        X=np.array(X)
        y=np.array(y)
        

    def _grow_tree(self, X, y, depth=0):
        X=np.array(X)
        y=np.array(y)
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        X=np.array(X)
        y=np.array(y)
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        y=np.array(y)
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

#     def _entropy(self, y):
#         hist = np.bincount(y)
#         ps = hist / len(y)
#         return -np.sum([p * np.log(p) for p in ps if p>0])
    def _entropy(self, y):
        ''' function to compute entropy '''
        y=np.array(y)
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy


    def _most_common_label(self, y):
        #y=pd.DataFrame(y)
        #value = y.value_counts().index[0]
        y=np.array(y)
        if len(y)!=0:
            vals, counts = np.unique(y, return_counts=True)
            value = np.argwhere(counts == np.max(counts))[0][0]
            return value
        else:
            return None

    def predict(self, X):
        X=np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        


# In[56]:


clf = DecisionTree(max_depth=5,min_samples_split=2)
clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)


# In[57]:


train_pred= clf.predict(X_train)
test_pred = clf.predict(X_val)


# In[58]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


# In[59]:


def evaluate_model(act, pred):
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))    


# In[61]:


print("--Train--")
evaluate_model(y_train, train_pred)
print("--Test--")
evaluate_model(y_val, test_pred)


# ## Hyperparametre tuning

# In[62]:


clf = DecisionTree(max_depth=8,min_samples_split=3)
clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)


# In[63]:


train_pred= clf.predict(X_train)
test_pred = clf.predict(X_val)


# In[64]:


print("--Train--")
evaluate_model(y_train, train_pred)
print("--Test--")
evaluate_model(y_val, test_pred)


# In[68]:


clf = DecisionTree(max_depth=15,min_samples_split=5)
clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)


# In[69]:


train_pred= clf.predict(X_train)
test_pred = clf.predict(X_val)


# In[70]:


print("--Train--")
evaluate_model(y_train, train_pred)
print("--Test--")
evaluate_model(y_val, test_pred)


# In[71]:


clf = DecisionTree(max_depth=3,min_samples_split=2)
clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)


# In[72]:


train_pred= clf.predict(X_train)
test_pred = clf.predict(X_val)


# In[73]:


print("--Train--")
evaluate_model(y_train, train_pred)
print("--Test--")
evaluate_model(y_val, test_pred)


# ### After hyperparametre tuning the best model that formed is the model having the max_depth = 5, min_samples_split = 2 

# In[ ]:




