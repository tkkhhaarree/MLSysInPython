import numpy as np
import scipy as sp
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# code to fetch data from tsv file
data = []
labels = []
with open('seeds.tsv') as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
features = np.array(data)
labels = np.array(labels)

#
classifier = KNeighborsClassifier(n_neighbors=1)
means = []
kf = KFold(n_splits=5, shuffle=True)
kf = kf.split(features)
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)
print("Mean accuracy: %f" % np.mean(means))  # will come different each time due to shuffling.
means = []
classifier = Pipeline([("norm", StandardScaler()), ("knn", classifier)])  # standard scaler makes features have similar dimensions.
crossed = cross_val_score(classifier, features, labels)  # score of individual folds after applying cross validation, default no. of folds= 3.
print("accuracy for each fold when prescaled data was used: {}".format(crossed))
