
# coding: utf-8

# In[7]:

from numpy import genfromtxt, zeros
# read the first 4 columns
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)

print (data.shape)

print (target.shape)

print (set (target))


# In[8]:

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3


# In[73]:

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(leaf_size=40,n_neighbors=10)


# In[74]:

classifier.fit(data,t) # training on the iris dataset


# In[75]:

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t, test_size=0.4, random_state=0)


# In[76]:

classifier.fit(train,t_train) # train
print (classifier.score(test,t_test)) # test


# In[77]:

from sklearn.metrics import confusion_matrix
print (confusion_matrix(classifier.predict(test),t_test))


# In[78]:

from sklearn.metrics import classification_report
print (classification_report(classifier.predict(test), t_test,
target_names=['setosa', 'versicolor', 'virginica']))


# In[ ]:




# In[ ]:




# In[ ]:



