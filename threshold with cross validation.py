import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from threshold import fit_model, accuracy, predict
from sklearn.datasets import load_iris

data = load_iris()
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names
# draw graph
for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    if t == 1:
        c = 'g'
        marker = 'o'
    if t == 2:
        c = 'b'
        marker = 'x'
    plt.scatter(features[target == t, 0],
                # take x coordinate of scatter pts as sepal length of only those datas, whose target is t.
                features[target == t, 1],
                # take y coordinate of scatter pts as sepal width of only those datas, whose target is t.
                marker=marker,
                c=c)

# plt.show()
labels = target_names[target]
plength = features[:, 2]
is_setosa = (labels == 'setosa')
max_setosa = plength[is_setosa].max()
min_others = plength[~is_setosa].min()
print("max of setosa: %f" % max_setosa)  # 1.90
print("min of others: %f" % min_others)  # 3.00
# thus we are able to classify whether plant is setosa or not by checking if petal length is less than 2 (as 2 is near 1.9) or not.

# threshold method for classifying in a plant is virginica or not.
features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')
best_acc = -1
for fi in range(features.shape[1]):
    thresh = features[:, fi]
    for t in thresh:
        feature_i = features[:, fi]
        pred = (feature_i > t)
        acc = (pred == is_virginica).mean()
        rev_acc = (pred == ~is_virginica).mean()
        if rev_acc > acc:
            acc = rev_acc
            reverse = True
        else:
            reverse = False
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse

print("if feature name: ", feature_names[best_fi])
print("is greater than: ", best_t)
print("then, the plant being virginica is: ", bool(~best_reverse))
print("with accuracy percentage: ", best_acc * 100)

# cross validation: using one data as testing data while rest as training data, repeat this with each data and check accuracy.
correct = 0.0
for ei in range(len(features)):
    training = np.ones(len(features), bool)
    training[ei] = False
    testing = ~training
    model = fit_model(features[training], is_virginica[training])
    predictions = predict(model, features[testing])
    correct += np.sum(predictions == is_virginica[testing])
acc = correct / float(len(features))
print('\nAccuracy of cross validation using threshold model: {0:.1%}'.format(acc))
