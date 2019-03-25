#/usr/bin/env python

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
# import argparse, os
#
# parser = argparse.ArgumentParser()
# parser.add_argument("path_to_data")
# args = parser.parse_args()
# testfile = args.path_to_data

df = load_breast_cancer()
data, target, target_names, feature_names = df.data, df.target, df.target_names, df.feature_names

# plot heatmap
df = pd.DataFrame(data, columns=feature_names)
df['target'] = target
sns.countplot(x='target',data=df,palette='hls')
plt.title('Class counts')
plt.savefig('./figures/count_plot.png')
plt.close(fig=None)
plt.figure(figsize=(18,18))
sns.heatmap(df.corr(), cmap="YlGnBu")
plt.title('Correlation matrix (Breast Cancer Data Set')
plt.savefig('./figures/Correlation_matrix.png')
plt.close(fig=None)
sns.pairplot(df, vars=feature_names[0:3], hue='target', kind='reg')
plt.savefig('./figures/pairplot.png')
plt.close(fig=None)

# training and testing splitting
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=42)

# Logistic regression
print("Running Logistic Regression")
from sklearn.linear_model import LogisticRegression as LR
classifier = LR(solver='newton-cg')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Scores:')
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()

ax1 = plt.subplot(1, 2, 1)
ax1.plot(fpr, tpr, label='Logistic Regression')
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
plt.grid(True)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(fpr, tpr, label='Logistic Regression')
ax2.plot([0, 1], [0, 1],'r--')
ax2.set_xlim(0, 0.2)
ax2.set_ylim(0.8, 1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Zoomed ROC curve')
plt.grid(True)

plt.legend(loc="best")
plt.tight_layout()
plt.savefig('./figures/ROC.png')
plt.close(fig=None)

testdata = pd.read_csv('./data/test.csv')
testid, testfeatures = testdata.iloc[:, 0].tolist(), testdata.iloc[:, 1:]

predictions = classifier.predict(testfeatures)
print('Patient ID  Prediction')
for id in testid:
    print('{0:10d}   {1:2d}'.format(id, predictions[testid.index(id)]))


#
# # Predict on test data
# if os.path.isfile(testfile):
#     print("Test file exists")
#     testdata = pd.read_csv(testfile)
#     testid, testfeatures = testdata.iloc[:, 0].tolist(), testdata.iloc[:, 1:]
#
#     predictions = classifier.predict(testfeatures)
#     print('Patient ID  Prediction')
#     for id in testid:
#         print('{0:10d}   {1:2d}'.format(id, predictions[testid.index(id)]))
# else:
#     print('Test file does not exist')

