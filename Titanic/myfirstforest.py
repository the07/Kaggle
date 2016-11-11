import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

#Load the train data into a dataframe
train_df = pd.read_csv('csv/train.csv', header = 0)

#We need to convert all strings to integer classifiers
#We need to fill in the missing values of the data and make it complete.

#female = 0, male = 1
train_df['Gender'] = train_df['Sex'].map( {'female':0, 'male': 1} ).astype(int)

#Embarked from 'C', 'Q', 'S'
#All missing Embarked -> just make them embark from the most common places

if len(train_df.Embarked[train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked']))) #determine all values of Embarked,
Ports_dict = { name: i for i, name in Ports }            #set up a dictionary in the form Ports: index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# All the ages with no data -> make the median of all ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age' ] = median_age

#Remove the Name column, Cabin, Ticket and Sex (since we copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket','Cabin','PassengerId'], axis = 1)

#Test data
test_df = pd.read_csv('csv/test.csv', header=0)

#we need to do the same to the test data
test_df['Gender'] = test_df['Sex'].map( {'female':0, 'male':1} ).astype(int)

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

#Convert all embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#All the ages with no data -> make the median of all ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print ('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print ('Predicting...')
output = forest.predict(test_data).astype(int)


predictions_file = open("Prediction/myfirstforest.csv", 'w')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print ('Done.')
