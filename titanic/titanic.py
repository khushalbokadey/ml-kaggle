#!/usr/bin/python

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import csv

with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    trainingData = list(reader)
    trainingData.pop(0)


with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    testData = list(reader)
    testData.pop(0)


train_X = []
train_Y = []

v = DictVectorizer(sparse=False)

for row in trainingData:
	featureData = {}
	train_Y.append(row[1])

	featureData['pclass'] = row[2]
	featureData['sex'] = row[4]
	featureData['age'] = row[5]
	# featureData['sibsp'] = row[6]
	# featureData['parch'] = row[7]
	featureData['fare'] = row[9]
	featureData['cabin'] = row[10]
	featureData['embarked'] = row[11]

	train_X.append(featureData)

train_X_vect = v.fit_transform(train_X)

# cv_X_vect = train_X_vect[801:]
# train_X_vect = train_X_vect[:800]

# cv_Y = train_Y[801:]
# train_Y = train_Y[:800]


clf = MultinomialNB()
# clf = clf.fit(train_X_vect, train_Y)

# pred = clf.predict(cv_X_vect)
# print accuracy_score(cv_Y, pred)

test_X = []

outputList = []

for row in testData:
	featureData = {}

	featureData['pclass'] = row[1]
	featureData['sex'] = row[3]
	featureData['age'] = row[4]
	# featureData['sibsp'] = row[5]
	# featureData['parch'] = row[6]
	featureData['fare'] = row[8]
	featureData['cabin'] = row[9]
	# featureData['embarked'] = row[10]

	test_X.append(featureData)
	outputList.append([row[0]])

test_X_vect = v.transform(test_X)

clf = clf.fit(train_X_vect, train_Y)
pred = clf.predict(test_X_vect)

for i in range(0, len(outputList)):
	outputList[i].append(pred[i])

print outputList

with open('output.csv', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(outputList)