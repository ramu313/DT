#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


dt = pd.read_csv("Customer_data.csv")


# In[3]:


dt.head()


# In[4]:


dt.nunique()


# In[5]:


dt.drop(["Unnamed: 0","id"],inplace = True, axis = 1)


# In[6]:


dt.shape


# In[7]:


dt.dtypes


# In[8]:


dt.isnull().sum()


# In[9]:


median_val = dt['Arrival Delay in Minutes'].median()
# Filling missing values with median as distribution of arrival delay was heavily skewed
dt['Arrival Delay in Minutes'] = dt['Arrival Delay in Minutes'].fillna(median_val)


# In[10]:


dt.describe()


# ##Average departure delay: 14 Minutes
# ##Average arrival delay: 15 Minutes
# ##Inflight wifi service has the lowest rating out of 5 (2.72)

# In[11]:


cat_cols = ['Inflight wifi service', 'Departure/Arrival time convenient','Ease of Online booking', 'Gate location', 
                 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 
                 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']


# In[12]:


dt[cat_cols] = dt[cat_cols].astype(object)


# In[13]:


dt.groupby('Customer Type').size().plot(kind='pie', autopct='%.f')
plt.axis('equal')
plt.show()


# ### Most of the customers are loyal customers (82%) 

# In[14]:


dt['Checkin service'].value_counts().plot(kind='bar')


# ### The passangers who are satisfied with the checkin service gave 4 star rating  

# In[15]:


sns.boxplot(x='satisfaction',y = 'Inflight entertainment',data=dt)


# #### The more satisfied the person is with online boarding then there are greater chances that the person will be satisfied. Same is the case for all the other parameters

# In[16]:


sns.boxplot(x='Inflight wifi service',y = 'Online boarding',data=dt)


# ### People who gets better service of inflight wifi likely to apply for online boarding and gives better rating 

# In[17]:


num_bins = 20
range_min = 0
range_max = 8000

# Create the bins
bins = np.linspace(range_min, range_max, num_bins + 1)

# Count the number of data points in each bin
hist, _ = np.histogram(dt['Flight Distance'], bins=bins)

# Plot the histogram
fig, ax = plt.subplots()
ax.bar(bins[:-1], hist, width=bins[1]-bins[0], align='edge')

# Add labels and title
ax.set_xlabel('Flight Distance (miles)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Flight Distances Traveled')

# Display the plot
plt.show()


# ### Most of the flights are between 0 to 1000 miles 

# In[18]:


sns.histplot(x='Flight Distance',hue="satisfaction",data=dt,element="poly")


# ### as the distance increases, the satisfaction increases!!! 

# In[19]:


dt["Gender"] = dt["Gender"].map({"Male":1,"Female":0})
dt["Customer Type"] = dt["Customer Type"].map({"Loyal Customer":1,"disloyal Customer":0})
dt["Type of Travel"] = dt["Type of Travel"].map({"Personal Travel":1,"Business travel":0})
dt["Class"] = dt["Class"].map({"Eco Plus":1,"Eco":0,"Business":2})
dt["satisfaction"] = dt["satisfaction"].map({"satisfied":1,"neutral or dissatisfied":0,})


# In[20]:


dt.head()


# In[21]:


dt.corr()


# In[22]:


corr = dt.corr()
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True)


# ### there is a high correlation between Arrival delay in minutes and departure delay in minutes
# so we can drop any one of them

# In[23]:


dt.drop(['Departure Delay in Minutes'], axis = 1,inplace=True)


# In[24]:


dt.dtypes


# In[25]:


X=dt.drop("satisfaction" , axis=1)
y=dt['satisfaction']


# In[26]:



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, stratify=y, random_state=321)


# In[27]:


features = list(X.columns)
features


# In[28]:


class DecisionTreeCART:
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)
    
    def build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return self.get_leaf
        
        feature_idxs = np.arange(n_features)
        if self.max_features and self.max_features <= n_features:
            feature_idxs = np.random.choice(feature_idxs, size=self.max_features, replace=False)
        
        best_feature, best_threshold = self.get_best_split(X, y, feature_idxs)
        if best_feature is None or best_threshold is None:
            return self.get_leaf
        
        left_idxs, right_idxs = self.split(X[:, best_feature], best_threshold)
        left_tree = self.build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right_tree = self.build_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
        }
    
    def get_leaf(self, y):
        class_counts = np.bincount
        return np.argmax(class_counts)
    
    def get_best_split(self, X, y, feature_idxs):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs, right_idxs = self.split(X[:, feature], threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                gain = self.get_gini_index(y, y[left_idxs], y[right_idxs])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def split(self, X_feature, threshold):
        left_idxs = np.where(X_feature <= threshold)[0]
        right_idxs = np.where(X_feature > threshold)[0]
        return left_idxs, right_idxs
    
    def get_gini_index(self, y, y_left, y_right):
        p = len(y_left) / len
        q = len(y_right) / len
        gini_parent = self.get_gini_impurity
        gini_left = self.get_gini_impurity(y_left)
        gini_right = self.get_gini_impurity(y_right)
        gini_index = gini_parent - (p * gini_left + q * gini_right)
        return gini_index
    
    def get_gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len
        gini_impurity = 1 - np.sum(probabilities**2)
        return gini_impurity
    
        def predict(self, X):
            if self.tree is None:
                raise Exception('Tree has not been trained yet.')
        
        predictions = np.empty(len(X), dtype=np.int)
        for i, x in enumerate(X):
            node = self.tree
            while isinstance(node, dict):
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions[i] = node
        
        return predictions


# In[29]:


hp = {'max_depth': 8,'min_samples_split': 6,'min_samples_leaf':8,'max_features':21}


# In[30]:


root = DecisionTreeCART(**hp)


# In[31]:


root.fit(X_train,y_train)


# In[ ]:


root.grow_tree()


# In[ ]:


root.print_tree()


# In[ ]:


train_pred=root.predict(X_train)
test_pred=root.predict(X_val)


# In[ ]:




