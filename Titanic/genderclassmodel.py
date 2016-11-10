import csv as csv
import numpy as np

csv_file_object = csv.reader(open('csv/train.csv'))
header = csv_file_object.__next__()
data = []

for row in csv_file_object:
    data.append(row)
data = np.array(data)

fare_ceiling = 40
data[data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

number_of_classes = 3

#but its better to calculate this from the data directly
number_of_classes = len(np.unique(data[0::,2]))

#initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in range(number_of_classes):
    for j in range(number_of_price_brackets):

        women_only_stats = data[(data[0::,4] == "female") & (data[0::,2].astype(np.float) == i +1) \
                                                          & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                                          & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) \
                                                          , 1]


        men_only_stats = data[(data[0::,4] != "female") & (data[0::,2].astype(np.float) == i +1) \
                                                          & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                                          & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) \
                                                          , 1]


        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

#since in python if it tries to find the mean of an array with nothing in it
#(such that denominator is zero), then it returns nan, we can convert these to 0
#by just saying where does the array not equal the array, and set those to 0
survival_table[survival_table != survival_table] = 0


#now we have the propotion of survivors, simply round them such that if < 0.5
# predict that they do not survive and if >= 0.5 they do.
survival_table[survival_table<0.5] = 0
survival_table[survival_table >= 0.5] = 1

test_file = open('csv/test.csv')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

prediction_file = open('Prediction/genderclassmodel.csv', 'w')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

#First thing to do is bin up the price falls
for row in test_file_object:
    for j in range(number_of_price_brackets):
        try:
            row[8] = float(row[8])

        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1)*fare_bracket_size:

            bin_fare = 1
            break

    if row[3] == 'female':
        prediction_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) -1, bin_fare])])
    else:
        prediction_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) -1, bin_fare])])

test_file.close()
prediction_file.close()
