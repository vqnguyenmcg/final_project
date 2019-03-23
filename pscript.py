#/usr/bin/env python

from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()
data, target, target_names, feature_names = df.data, df.target, df.target_names, df.feature_names

# plot heatmap
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.DataFrame(data, columns=feature_names)
sns.heatmap(df.corr())
plt.title('Correlation matrix')
plt.savefig('Correlation_matrix.pdf', cmap="YlGnBu")

# sub_features = feature_names[np.random.randint(0,31, 3)]
# pp= sns.pairplot(df[sub_features], kind='reg')
# pp.fig.suptitle('Pairplot')
# plt.savefig('Pairplot.pdf')
# plt.close(fig=None)
# training and testing splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# # NearestNeighbors
# print("K-Nearest Neighbors classification")
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# classifier = KNeighborsClassifier(n_neighbors=2)
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
# print("Mean accuracy {:0.2f} ".format(classifier.score(X_test,y_test)))
# print("Classification Report")
# print(classification_report(y_test,predictions))

# Logistic regression
print("Logistic Regression")
from sklearn.linear_model import LogisticRegression as LR
classifier = LR(solver='newton-cg')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print("Mean accuracy {:0.2f} ".format(classifier.score(X_test,y_test)))
print("Classification Report")
print(classification_report(y_test,predictions))

# # Bayes Inference
# print("Bayes Inference")
# from sklearn.naive_bayes import GaussianNB as GNB
# classifier = GNB()
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
# print("Mean accuracy {:0.2f} ".format(classifier.score(X_test,y_test)))
# print("Classification Report")
# print(classification_report(y_test,predictions))