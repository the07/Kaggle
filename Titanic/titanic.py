#the first thing we are doing is importing the relevant packages that we need
#for our script. This includes Numpy and csv.

import csv as csv
import numpy as np

csv_file_object = csv.reader(open('csv/train.csv'))
header = csv_file_object.__next__() #the next command skips the first line which is
                                #a header.
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

women_onboard = data[women_only_stats, 1].astype(np.float)
men_onboard = data[men_only_stats, 1].astype(np.float)

propotion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
propotion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print ('Propotion of women who survived is %s' %propotion_women_survived)
print('Propotion of men who survived is %s' %propotion_men_survived)

test_file = open('csv/test.csv')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

prediction_file = open("genderbasedmodel","w")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], '1'])
    else:
        prediction_file_object.writerow([row[0], '0'])

test_file.close()
prediction_file.close()
