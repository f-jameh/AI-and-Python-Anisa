# In the name of God

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, zero_one_loss
from joblib import dump

# read dataset
file_path = '/home/farhad/Desktop/priv/codes/ml/knn-iris-classifier/iris.csv'
dataset = pd.read_csv(file_path, header=None)

# print loaded dataset information
print(f' the dataset shape is {dataset.shape}')
print('some head rows od dataset:')
print(dataset.head())

# Split features and labels

features = dataset.iloc[:, :4]
print('features are:')
print(features)
print(f' type of featues is: {type(features)}')
print(f' lenght of features are : {len(features)}')

labels = dataset.iloc[:, 4]
print('labels are:')
print(labels)
print(f' type of labels is: {type(labels)}')
print(f' lenght of labels are : {len(labels)}')

# split trains from tests data
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=44)

# optional: print splited data and labels
print('x train is:')
print(xtrain)

print('x test is:')
print(xtest)

print('y train is:')
print(ytrain)

print('y test is:')
print(ytest)

# train model
clf = KNeighborsClassifier(5)
clf.fit(xtrain, ytrain)

ypred = clf.predict(xtest)
print(ypred)
print(f'number of predicted lables are: {len(ypred)}')

# calculte accuracy metrics
acc = accuracy_score(ytest, ypred)
loss = zero_one_loss(ytest, ypred)
print(f'accuracy of this model is: {acc * 100:.2f}%')
print(f'accuracy of this model is: {loss * 100:.2f}%')

# save the model output
dump(clf, '/home/farhad/knn_model.pkl')
print('the model has saved')

















