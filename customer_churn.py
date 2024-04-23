import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

churn_df = pd.read_csv("E:\data\datasets\customer_churn.csv.csv")

# Remove nulls and duplicates
churn_df.drop_duplicates()
churn_df.dropna()

# Check for unique values in each column
churn_df.Partner.unique()
churn_df.gender.unique()
churn_df.Dependents.unique()
churn_df.PhoneService.unique()
churn_df.MultipleLines.unique()
churn_df.InternetService.unique()
churn_df.OnlineSecurity.unique()
churn_df.OnlineBackup.unique()
churn_df.DeviceProtection.unique()
churn_df.TechSupport.unique()
churn_df.StreamingTV.unique()
churn_df.StreamingMovies.unique()
churn_df.Contract.unique()
churn_df.PaperlessBilling.unique()
churn_df.PaymentMethod.unique()
churn_df.Churn.unique()

# Convert string columns to numeric values
"""
This is the long and less preferred method to adjust data values, the preferred method would be
to convert the data using SQL or adjust the data values in the csv file directly, but this method
is done to show that it is possible using python. 

(only acceptable because there is a relatively small number of records in the dataset, less than 
10,000)
"""
churn_df['Partner'] = churn_df['Partner'].map({'Yes': 1, 'No': 0})
churn_df['gender'] = churn_df['gender'].map({'Female': 1, 'Male': 0})
churn_df['Dependents'] = churn_df['Dependents'].map({'Yes': 1, 'No': 0})
churn_df['PaperlessBilling'] = churn_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
churn_df['PhoneService'] = churn_df['PhoneService'].map({'Yes': 1, 'No': 0})
churn_df['MultipleLines'] = churn_df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 2})
churn_df['InternetService'] = churn_df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
churn_df['OnlineSecurity'] = churn_df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
churn_df['OnlineBackup'] = churn_df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
churn_df['DeviceProtection'] = churn_df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
churn_df['TechSupport'] = churn_df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
churn_df['StreamingTV'] = churn_df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
churn_df['StreamingMovies'] = churn_df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
churn_df['Contract'] = churn_df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
churn_df['PaymentMethod'] = churn_df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 
                                                           'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
churn_df['Churn'] = churn_df['Churn'].map({'Yes': 1, 'No': 0})

"""
Even though we have dropped the null records from the dataframe, 
some data is missing from the TotalCharges column because they are empty strings, 
so we convert those records to null values, then drop them
"""
churn_df['TotalCharges'].replace(' ', np.nan, inplace=True)
churn_df.dropna()

# Split data into input and output
X = churn_df.drop(columns=['customerID', 'Churn']) 
X = X.drop(columns=['PaymentMethod', 'DeviceProtection', 'PaperlessBilling']) # pre-prune the tree to increase accuracy
y = churn_df['Churn']

# Create the ML model
model = DTC()

# Split data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03)

# Fit the model to the training data
model.fit(X_train, y_train)

# prediction
prediction = model.predict(X_test)

# test accuracy
score = accuracy_score(y_test, prediction)
print(score)

joblib.dump(model, 'customer_churn.joblib')