import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1,-1],[-3,-4],[5,7],[-5,-7],[2,3]])
Y = np.array([0,0,1,0,1])

clf = GaussianNB()
clf.fit(X,Y)

print(clf.predict([[3,3]]))
print(clf.predict([[-3,-3]]))
