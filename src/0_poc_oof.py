from numpy import hstack
from numpy import array
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# create a meta dataset
def create_meta_dataset(data_x, yhat1, yhat2):
	# convert to columns
	yhat1 = array(yhat1).reshape((len(yhat1), 1))
	yhat2 = array(yhat2).reshape((len(yhat2), 1))

	# stack as separate columns
	meta_X = hstack((data_x, yhat1, yhat2))
	
	return meta_X
 
# make predictions with stacked model
def stack_prediction(model1, model2, meta_model, X):
	# make predictions
	yhat1 = model1.predict_proba(X)[:, 0]
	yhat2 = model2.predict_proba(X)[:, 0]

	# create input dataset
	meta_X = create_meta_dataset(X, yhat1, yhat2)

	# predict
	return meta_model.predict(meta_X)
 
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# collect out of sample predictions
data_x, data_y, knn_yhat, cart_yhat = list(), list(), list(), list()
kfold = KFold(n_splits=10, shuffle=True)

for train_ix, test_ix in kfold.split(X_train):
	# get data
	train_X, test_X = X_train[train_ix], X_train[test_ix]
	train_y, test_y = y_train[train_ix], y_train[test_ix]
	data_x.extend(test_X)
	data_y.extend(test_y)

	# fit and make predictions with cart
	model1 = DecisionTreeClassifier()
	model1.fit(train_X, train_y)
	yhat1 = model1.predict_proba(test_X)[:, 0]
	cart_yhat.extend(yhat1)

	# fit and make predictions with cart
	model2 = KNeighborsClassifier()
	model2.fit(train_X, train_y)
	yhat2 = model2.predict_proba(test_X)[:, 0]
	knn_yhat.extend(yhat2)

# construct meta dataset
meta_X = create_meta_dataset(data_x, knn_yhat, cart_yhat)

# # fit final submodels
model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)

# construct meta classifier
meta_model = LogisticRegression(solver='liblinear')
meta_model.fit(meta_X, data_y)

# evaluate sub models on hold out dataset
acc1 = accuracy_score(y_test, model1.predict(X_test))
acc2 = accuracy_score(y_test, model2.predict(X_test))
print('Model1 Accuracy: %.3f, Model2 Accuracy: %.3f' % (acc1, acc2))

# evaluate meta model on hold out dataset
yhat = stack_prediction(model1, model2, meta_model, X_test)
acc = accuracy_score(y_test, yhat)
print('Meta Model Accuracy: %.3f' % (acc))