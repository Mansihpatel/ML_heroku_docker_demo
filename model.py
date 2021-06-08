import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("D:/ML_heroku_project/hiring.csv")
print(dataset)
print(dataset.columns)
dataset['experience'].fillna(0,inplace=True)
dataset['test_score(out of 10)'].fillna(0,inplace = True)
print(dataset)

X = dataset.iloc[:,:3]
X

# converting words to integer values

def convert_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'zero':0,0:0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_int(x))
y = dataset.iloc[:,-1]

# model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

# saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))