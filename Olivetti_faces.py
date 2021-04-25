import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_olivetti_faces
dataset = fetch_olivetti_faces()

images = dataset.images

for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], "gray")
plt.show()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

##################### Ensemble Techniques ####################

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

###################### Stacking/Voting ######################

from sklearn.ensemble import VotingClassifier
vote = VotingClassifier([('LR', log_reg),
                         ('NB', nb),
                         ('DT', dtf),
                         ('KNN', knn)])

vote.fit(X_train, y_train)

print(vote.score(X_train, y_train))
print(vote.score(X_test, y_test))

##### Compute all other performance metrics from confusion
##### matrix to precision to everything to tuning.

##################### Bagging ################################

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(dtf, n_estimators = 7)

bag.fit(X_train, y_train)

print(bag.score(X_train, y_train))
print(bag.score(X_test, y_test))

##################### Random Forest ###########################

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 70)

rf.fit(X_train, y_train)

print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

################## Extra Trees ###############################

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators = 45)

et.fit(X_train, y_train)

print(et.score(X_train, y_train))
print(et.score(X_test, y_test))

#################### Adaptive Boosting #######################

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators = 200)

ada.fit(X_train, y_train)

print(ada.score(X_train, y_train))
print(ada.score(X_test, y_test))

################### Gradient Boosting ########################

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

print(gb.score(X_train, y_train))
print(gb.score(X_test, y_test))


























































