# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 
data = pd.read_csv(path)

# Overview of the data
data.describe()

#Dividing the dataset set in train and test set and apply base logistic model
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns="1"), data["1"], test_size=0.3, random_state=6)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Calculate accuracy , print out the Classification report and Confusion Matrix.
accuracy_score = lr_model.score(X_test, y_test)
accuracy_score
y_pred = lr_model.predict(X_test)
cr = classification_report(y_test, y_pred)
cr
cm = confusion_matrix(y_test, y_pred)
cm

# Copy df in new variable df1
df1 = data
df1 = df1.corr().unstack()
df1 = df1[(abs(df1)>0.75) & (abs(df1)!=1)]

# Remove Correlated features above 0.75 and then apply logistic model
df1 = data.drop(columns=["0.25", "0.31"])

# Split the new subset of data and fit the logistic model on training data
X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns="1"), df1["1"], test_size=0.3, random_state=6)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
accuracy_score = lr_model.score(X_test, y_test)
accuracy_score

# Calculate accuracy , print out the Classification report and Confusion Matrix for new data
y_pred = lr_model.predict(X_test)
cr = classification_report(y_test, y_pred)
cr

# Apply Chi Square and fit the logistic model on train data use df dataset
test = SelectKBest(score_func=chi2, k='all')
X_train = test.fit_transform(X_train, y_train)
X_test = test.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
score
cm = confusion_matrix(y_test, lr.predict(X_test))
cm

# Calculate accuracy , print out the Confusion Matrix 


# Apply Anova and fit the logistic model on train data use df dataset



# Calculate accuracy , print out the Confusion Matrix 


# Apply PCA and fit the logistic model on train data use df dataset
X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns="1"), df1["1"], test_size=0.3, random_state=6)
pca = PCA(n_components=35, random_state=0)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_perd = lr.predict(X_test)
score = lr.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)
score, cm
   

# Calculate accuracy , print out the Confusion Matrix 


# Compare observed value and Predicted value




