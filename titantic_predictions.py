# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Set training / test data location
training_file_location = "../data/titanic_data/train.csv"
test_file_location = "../data/titanic_data/test.csv"

# Import training / test dataset
training_dataset = pandas.read_csv(training_file_location)
test_dataset = pandas.read_csv(test_file_location)

# Drop the columns that do not contribute to the model
training_dataset = training_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# Remove the rows with nan's
training_dataset = training_dataset.dropna()
# Change the Sex column to values
gender = {'male': 1,'female': 0}
# Change the Emarkement column to values
embarked = {'C':0, 'Q':1, 'S':2}
training_dataset.Sex = [gender[item] for item in training_dataset.Sex]
training_dataset.Embarked = [embarked[item] for item in training_dataset.Embarked]

print(training_dataset.head(20))

# Split-out the validation dataset
array = training_dataset.values
X = array[:,1:8]
Y = array[:,0]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Predict the training based off the train and test data
lda = LinearDiscriminantAnalysis()
lda.fit(X, Y)
# Get the test data in the correct format
















