import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


#-----------
# LOADING THE DATA
#------------

filename = 'Credit Card Defaulter Prediction.csv'
if not os.path.exists(filename):
  print("File not found. Please refer to the specifications and download the correct file.")
  exit()

#-----------
# CLEANING AND PREPARING DATA
#------------

df = pd.read_csv('Credit Card Defaulter Prediction.csv')
print(df.head())

print(df.describe())

print(df.info())

#Drop ID so that the model does not memorize IDs 
df.drop(['ID'],axis=1,inplace=True)

print(df['MARRIAGE'].value_counts())
print(df['EDUCATION'].value_counts())

print(df.columns)

#Change Y and N to 1 and 0 
target = 'default '
df[target] = df[target].map({'N':0,'Y':1})

#remove default column to split data 
X = df.drop(target,axis=1)
y = df[target]
print(y.head())


#Ensure all X variables are numbers 
X = pd.get_dummies(X, drop_first=True)
print(X.dtypes)


#-----------
# MODEL 
#------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# instance of RandomForestClassifier model 
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

#Fit the model for training
model.fit(X_train, y_train)


#Predict on test set
y_pred = model.predict(X_test)


# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") #Accuracy rate: 0.8143333333333334

# save the model to disk
joblib.dump(model, "rf_model.sav")

print("\nDetailed Report:\n", classification_report(y_test, y_pred))


# Splits the data 5 different ways and averages the result to confirm accuracy
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Scores for each run:", scores)
print("True Average Accuracy:", scores.mean())  
#Scores for each run: [0.80466667 0.80866667 0.81666667 0.82783333 0.8195]
#True Average Accuracy: 0.8154666666666666
