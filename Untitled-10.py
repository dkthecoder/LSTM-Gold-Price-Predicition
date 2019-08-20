from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import sklearn
import tensorflow as tf
import pandas
import numpy
import keras


#Step 1a Load Data //CSV needs to be in columbs
csv_path = "/tf/POINT_CSV_NAME.csv"
timedate = pandas.read_csv(csv_path, usecols=[0], parse_dates=True, infer_datetime_format=True)
dataOPEN = pandas.read_csv(csv_path, usecols=[1])
dataHIGH = pandas.read_csv(csv_path, usecols=[2])
dataLOW = pandas.read_csv(csv_path, usecols=[3])
dataCLOSE = pandas.read_csv(csv_path, usecols=[4])


#Step 1b Scale data betwen 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))
openScaled = scaler.fit_transform(dataOPEN)
highScaled = scaler.fit_transform(dataHIGH) 
lowScaled = scaler.fit_transform(dataLOW) 
closeScaled = scaler.fit_transform(dataCLOSE) 


#SAVE!
numpy.savetxt('openScaled.csv', openScaled, delimiter=',')
numpy.savetxt('highScaled.csv', highScaled, delimiter=',')
numpy.savetxt('lowScaled.csv', lowScaled, delimiter=',')
numpy.savetxt('close_scaled.csv', closeScaled, delimiter=',')


#VARIABLES
#allValues for total set of values
#splitUnit for values per split
splitUnit = 26
allValues = 500


#Step 1c load data into appropriate values, convert to numpy and reshape
features_set_open = []  
labels_open = []  
for i in range(splitUnit, allValues):  
    features_set_open.append(openScaled[i-splitUnit:i, 0])
    labels_open.append(openScaled[i, 0])
features_set_open, labels_open = numpy.array(features_set_open), numpy.array(labels_open)
features_set_open = numpy.reshape(features_set_open, (features_set_open.shape[0], features_set_open.shape[1], 1))  

features_set_high = []  
labels_high = []  
for i in range(splitUnit, allValues):  
    features_set_high.append(highScaled[i-splitUnit:i, 0])
    labels_high.append(highScaled[i, 0])
features_set_high, labels_high = numpy.array(features_set_high), numpy.array(labels_high)
features_set_high = numpy.reshape(features_set_high, (features_set_high.shape[0], features_set_high.shape[1], 1))  

features_set_low = []  
labels_low = []  
for i in range(splitUnit, allValues):  
    features_set_low.append(lowScaled[i-splitUnit:i, 0])
    labels_low.append(lowScaled[i, 0])
features_set_low, labels_low = numpy.array(features_set_low), numpy.array(labels_low)
features_set_low = numpy.reshape(features_set_low, (features_set_low.shape[0], features_set_low.shape[1], 1))  

features_set_close = []  
labels_close = []  
for i in range(splitUnit, allValues):  
    features_set_close.append(closeScaled[i-splitUnit:i, 0])
    labels_close.append(closeScaled[i, 0])
features_set_close, labels_close = numpy.array(features_set_close), numpy.array(labels_close)
features_set_close = numpy.reshape(features_set_close, (features_set_close.shape[0], features_set_close.shape[1], 1))  


#Step 1d spliting the dataset into training and test set
open_x_train, open_x_test, open_y_train, open_y_test = train_test_split(features_set_open, labels_open, test_size=0.1)
high_x_train, high_x_test, high_y_train, high_y_test = train_test_split(features_set_high, labels_high, test_size=0.1)
low_x_train, low_x_test, low_y_train, low_y_test = train_test_split(features_set_low, labels_low, test_size=0.1)
close_x_train, close_x_test, close_y_train, close_y_test = train_test_split(features_set_close, labels_close, test_size=0.1)


#VARIABLES
nodes = 10
epochUnit = 50
dropout = 0.2

#Step 2a Build Model For Open
modelOpen = Sequential()
modelOpen.add(LSTM(nodes, return_sequences=True, input_shape=(features_set_open.shape[1],1)))  
modelOpen.add(Dropout(dropout))

modelOpen.add(LSTM(nodes, return_sequences=True))
modelOpen.add(Dropout(dropout))

modelOpen.add(LSTM(nodes))

modelOpen.add(Dense(1))

modelOpen.compile(loss='mse', optimizer='adam')
modelOpen.fit(open_x_train, open_y_train, batch_size=26, epochs=epochUnit)


#Step 2b Build Model For High
modelHigh = Sequential()
modelHigh.add(LSTM(nodes, return_sequences=True, input_shape=(features_set_high.shape[1], 1)))
modelHigh.add(Dropout(dropout))

modelHigh.add(LSTM(nodes, return_sequences=True))
modelHigh.add(Dropout(dropout))

modelHigh.add(LSTM(nodes))

modelHigh.add(Dense(1))

modelHigh.compile(loss='mse', optimizer='adam')
modelHigh.fit(high_x_train, high_y_train, batch_size=26, epochs=epochUnit)


#Step 2c Build Model For Low
modelLow = Sequential()
modelLow.add(LSTM(nodes, return_sequences=True, input_shape=(features_set_low.shape[1], 1)))
modelLow.add(Dropout(dropout))

modelLow.add(LSTM(nodes, return_sequences=True))
modelLow.add(Dropout(dropout))

modelLow.add(LSTM(nodes))

modelLow.add(Dense(1))

modelLow.compile(loss='mse', optimizer='adam')
modelLow.fit(low_x_train, low_y_train, batch_size=26, epochs=epochUnit)


#Step 2d Build Model For Close
modelClose = Sequential()
modelClose.add(LSTM(nodes, return_sequences=True, input_shape=(features_set_close.shape[1], 1)))
modelClose.add(Dropout(dropout))

modelClose.add(LSTM(nodes, return_sequences=True))
modelClose.add(Dropout(dropout))

modelClose.add(LSTM(nodes))

modelClose.add(Dense(1, activation="linear"))

modelClose.compile(loss='mse', optimizer='adam')
modelClose.fit(close_x_train, close_y_train, batch_size=26, epochs=epochUnit, validation_split=0.2)


#Step 3a - Predict with test data!
predictions_open = modelOpen.predict(open_x_test)
predictions_high = modelHigh.predict(high_x_test)
predictions_low = modelLow.predict(low_x_test)
predictions_close = modelClose.predict(close_x_test)


#SAVE!
numpy.savetxt('predictions_open.csv', predictions_open, delimiter=',')
numpy.savetxt('predictions_high.csv', predictions_high, delimiter=',')
numpy.savetxt('predictions_low.csv', predictions_low, delimiter=',')
numpy.savetxt('predictions_close.csv', predictions_close, delimiter=',')


#Step 3b - Inverse scale!
inversed_open = scaler.inverse_transform(predictions_open)
inversed_high = scaler.inverse_transform(predictions_high) 
inversed_low = scaler.inverse_transform(predictions_low) 
inversed_close = scaler.inverse_transform(predictions_close)


#SAVE!
numpy.savetxt('inversed_open.csv', inversed_open, delimiter=',')
numpy.savetxt('inversed_high.csv', inversed_high, delimiter=',')
numpy.savetxt('inversed_low.csv', inversed_low, delimiter=',')
numpy.savetxt('inversed_close.csv', inversed_close, delimiter=',')