# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:17:50 2019

@author: Shriyash Shende
"""

import numpy as np
import pandas as pd
import seaborn as sns
import json

m = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\train.csv')
m1 = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\test.csv')
r = m['revenue']
m.drop(['revenue'], axis =1, inplace = True)
all_data = pd.concat([m,m1],axis = 0)
m.columns
m.info()
#Revenue
sns.distplot(r)
sns.distplot(np.log(r))
#Budget
sns.distplot(all_data['budget'])
sns.distplot(np.log(all_data['budget'] + 1))
sns.distplot(m.budget[all_data.budget != 0] + 1)
sns.distplot(np.log(all_data.budget[all_data.budget != 0] + 1))
#popularity 
sns.distplot(all_data['popularity'])
sns.distplot(np.log(all_data['popularity']))

all_data.isnull().sum()
#Droping Variables
all_data.columns
all_data['status'].unique()
all_data.drop(['original_title','Keywords','id', 'belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'overview', 'tagline','title'], axis =1 , inplace = True)


#Transforming variables 
def parse_json(x):
    try: return json.loads(x.replace("'", '"'))[0]['name']
    except: return ''

all_data['production_companies'] = all_data['production_companies'].apply(parse_json)


all_data['production_countries'] = all_data['production_countries'].apply(parse_json)
all_data['spoken_languages'] = all_data['spoken_languages'].apply(parse_json)


all_data['genres'] = all_data['genres'].apply(lambda x: str(x))
all_data['num_of_genres'] = all_data['genres'].apply(lambda x: x.count("name"))
all_data.drop(['genres'], axis = 1, inplace = True)
all_data['cast'] =all_data['cast'].apply(lambda x: x.count('{') if type(x)==str else 0)
all_data['crew'] =all_data['crew'].apply(lambda x: x.count('{') if type(x)==str else 0)

all_data['release_date'] = pd.to_datetime(all_data['release_date'], format='%m/%d/%y')
all_data['day'] = all_data['release_date'].dt.weekday
all_data['day'].isnull().sum()
all_data['day'].fillna(4, inplace = True)
all_data['day'].replace(to_replace =[0],  value ='Mon', inplace = True)
all_data['day'].replace(to_replace =[1],  value ='Tue', inplace = True)
all_data['day'].replace(to_replace =[2],  value ='Wed', inplace = True)
all_data['day'].replace(to_replace =[3],  value ='Thu', inplace = True)
all_data['day'].replace(to_replace =[4],  value ='Fri', inplace = True)
all_data['day'].replace(to_replace =[5],  value ='Sat', inplace = True)
all_data['day'].replace(to_replace =[6],  value ='Sun', inplace = True)

all_data['month'] = all_data['release_date'].dt.month
all_data['month'].fillna(9, inplace=True)
all_data['month'].replace(to_replace =[1],  value ='Jan', inplace = True)
all_data['month'].replace(to_replace =[2],  value ='Feb', inplace = True)
all_data['month'].replace(to_replace =[3],  value ='Mar', inplace = True)
all_data['month'].replace(to_replace =[4],  value ='Apr', inplace = True)
all_data['month'].replace(to_replace =[5],  value ='May', inplace = True)
all_data['month'].replace(to_replace =[6],  value ='Jun', inplace = True)
all_data['month'].replace(to_replace =[7],  value ='Jul', inplace = True)
all_data['month'].replace(to_replace =[8],  value ='Aug', inplace = True)
all_data['month'].replace(to_replace =[9],  value ='Sep', inplace = True)
all_data['month'].replace(to_replace =[10],  value ='Oct', inplace = True)
all_data['month'].replace(to_replace =[11],  value ='Nov', inplace = True)
all_data['month'].replace(to_replace =[12],  value ='Dec', inplace = True)

all_data['year'] = all_data['release_date'].dt.year
all_data['year'].fillna(all_data['year'].median(), inplace=True)
all_data.drop('release_date', axis = 1 , inplace =True)

all_data['year'].replace(to_replace =[2062],  value =1962, inplace = True)
all_data['year'].replace(to_replace =[2065],  value =1965, inplace = True)
all_data['year'].replace(to_replace =[2067],  value =1967, inplace = True)
all_data['year'].replace(to_replace =[2056],  value =1956, inplace = True)
all_data['year'].replace(to_replace =[2066],  value =1966, inplace = True)
all_data['year'].replace(to_replace =[2068],  value =1968, inplace = True)
all_data['year'].replace(to_replace =[2028],  value =1928, inplace = True)
all_data['year'].replace(to_replace =[2033],  value =1933, inplace = True)
all_data['year'].replace(to_replace =[2049],  value =1949, inplace = True)
all_data['year'].replace(to_replace =[2058],  value =1958, inplace = True)
all_data['year'].replace(to_replace =[2048],  value =1948, inplace = True)
all_data['year'].replace(to_replace =[2059],  value =1959, inplace = True)
all_data['year'].replace(to_replace =[2041],  value =1941, inplace = True)
all_data['year'].replace(to_replace =[2046],  value =1946, inplace = True)
all_data['year'].replace(to_replace =[2057],  value =1957, inplace = True)
all_data['year'].replace(to_replace =[2025],  value =1925, inplace = True)
all_data['year'].replace(to_replace =[2061],  value =1961, inplace = True)
all_data['year'].replace(to_replace =[2060],  value =1960, inplace = True)
all_data['year'].replace(to_replace =[2044],  value =1944, inplace = True)
all_data['year'].replace(to_replace =[2039],  value =1939, inplace = True)
all_data.year = all_data.year.astype(str)



all_data.isnull().sum()
all_data['runtime'].describe()
all_data['runtime'].fillna(104, inplace=True)

all_data['status'].describe()
all_data['status'].fillna('Released', inplace =True)
all_data.columns
all_data.drop(['spoken_languages', 'production_companies', 'production_countries'], axis =1 , inplace = True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(all_data['year'])
all_data['Year'] = le.transform(all_data['year'])
all_data.drop(['year'], axis = 1, inplace = True)

all_data_dummies = pd.get_dummies(all_data, drop_first = True)

all_data_dummies.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\ALL.csv')

final_train = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\train1.csv')
final_test = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\test1.csv')

final_train.drop(['Unnamed: 0'], axis = 1, inplace = True)
final_test.drop(['Unnamed: 0'], axis = 1, inplace = True)
r = pd.DataFrame(r)
final_train = final_train.join(r)
final_train.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\final_train.csv')
final_test.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\final_test.csv')
