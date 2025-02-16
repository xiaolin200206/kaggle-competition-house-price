#load libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#load data and look what data looks like
train = pd.read_csv(r"C:\xmapp\competition\house_train.csv")
test = pd.read_csv(r"C:\xmapp\competition\house_test.csv")
print(train.shape, test.shape)
train.head()

#use isnull to fill missing values
#sum(): true is treated as 1, false is treated as 0, summing along each column gives you the total count of missing values in each column
#ascending = False: this sort the columns by the number of missing values in descening order, so the most missing value at the top
#missing>0: keep the number of missing value is greater than 0, effectively removing columns with no missing values
missing = train.isnull().sum().sort_values(ascending=False)
missing = missing[missing>0]
print(missing)

#imputer missing value with median for numerical and most frequent for categorical
num_imputer = SimpleImputer(strategy='median') #will replace missing value in numerical columns with the median value of that column
cat_imputer = SimpleImputer(strategy='most_frequent') # will replace missing value in categorical columns with most frequently in column

for col in train.columns:  #loop through every column in train dataframe
    if train[col].dtype == 'object': #check the data type type of the column
        train[col] = cat_imputer.fit_transform(train[[col]]) #this replace the missing value in categorical columns with the most frequent value
    else:
        train[col] = num_imputer.fit_transform(train[[col]]) #this replace the misssing value with the median value

#convert categorical variable into numeric format 
encoder = LabelEncoder()
for col in train.select_dtypes(include='object').columns: # select all columns in the train dataframe with object data type, which are typically categorical columns
    train[col] = encoder.fit_transform(train[col]) #fit: assigns each a unique integet, transform: it replace each value in train[col] with its corresponding integer label

#TotalSF mean Total Square footage
#TotalBsmtSF mean Total square footage of the basement
#1stF1rSF mean Total square of the first floor
#2ndF1rSF mean Total square of the seconde floor 
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'] #split the dataset  

#this step is for split data
X = train.drop(['SalePrice'], axis = 1) #drop the target column saleprice from the training dataset
Y = train['SalePrice']  #Y is training targe -- saleprice is our  try to predict
X_train,X_val,Y_train,Y_val = train_test_split(X,Y, test_size = 0.2, random_state = 42) # split dataset, test_size = 0.2 mean 20% of the data used for validation the model's performance,
#random_state=42 means ensure that every time run this code

#when use training set vs validation set
#training set  used during model training. the model learns the patterns, relationships and structure from this dataset
#when to use it??? fitting the model and adjust the model's internal parameters based on the data
#validation set used to evaluation the trained model. and check how well the model performs on data it hasn't seen before
#it helps to tune hyperparameters and detect overfitting
#when to use it???? after training is done and test the model's performance

# standardscaler ensure all feature have a mean of 0 and standard deviation of 1, why this tool is important??
#because many models perform better when the data is on a similar scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # fit calculates the mean and standard deviation from the training data
#transform apply that scaling to trnasform the training data
#why fit in training data??? because to avoid data leakage
X_val = scaler.transform(X_val) # use the same mean and standard deviation from the training set 
#why not this set fit ??? because validation set should only be transformed using the training set's scaling
#fitting again would mean calculate different mean and std

#train the model
#RandomForestRegressor is one model from sklearn.ensemble
#it use many decision trees to make prediction and combine predictions from all trees to give a final result
#n_estimators=100 means this model will build 100 decision trees which mean better performance
#ramdom_state=42 means results are consistent every time you run the code
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,Y_train)

#rmse mean root mean squared error
#why use mse? ?? beacuse mse can measure how close your prediction are to the actual value
# smaller mse and bigger sme: small mse is better model performance
#what if  bigger mse happend?? it means you model maybe too complex or too simple, and maybe missing some important feature which can affect house price
y_pred = model.predict(X_val) #after training model,use predict value for validation sets
rmse = np.sqrt(mean_squared_error(Y_val,y_pred)) # Y_val means actual house price from the validation set
               # Y_pred means predict house price from model 
#rmse equal square root of mse
print(f"Validation RMSE: {rmse}")

test = pd.read_csv(r"C:\xmapp\competition\house_test.csv")

print(test.columns)

#repeat preprocessing steps for test data 
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = cat_imputer.transform(test[[col]].values)
    else:
        test[col] = num_imputer.transform(test[[col]].values)


test['TotalSF'] = test['TotRmsAbvGrd'] + test['1stFlrSF'] + test['2ndFlrSF']
test = scaler.transform(test)

#predict on test data
predictions = model.predict(test)

