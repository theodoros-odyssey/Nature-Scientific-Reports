import time
import math
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from readFileToArray import readFile

from sklearn.ensemble import RandomForestRegressor

# Evaluation Metrics for Weka Regression Models
print("Evaluation results for RF regression")

#Store evaluation values in text file
metrics = open("eval_RF_Metrics.txt", "w")

# Read input datasets
totalFiles = 17
fileIndex = 0
# Certain number of datasets for loop
for fileCounter in range(totalFiles):

    # Set start time to compute execution time
    startTime = time.time()

    fileIndex = fileCounter + 1
    fileName = str(fileIndex) + '.txt'

    inputData = readFile(fileName, 'r')
    rowsData = len(inputData)
    columnsData = len(inputData[0])

    # Define actual and predicted values arrays
    actualArray = []
    predictedSelfArray = []

    i = 1
    while i <= rowsData:
        # Training phase
        prevTrainData = inputData[0:i - 1, 0:columnsData]
        testData = inputData[i - 1:i, 0:columnsData]
        nextTrainData = inputData[i:rowsData, 0:columnsData]
        trainData = np.concatenate([prevTrainData, nextTrainData], 0)
        trainDataX = trainData[:, 0:columnsData - 1]
        trainDataY = trainData[:, columnsData - 1]

        # create an RF model with a linear kernel
        rf = RandomForestRegressor(
            n_estimators=100,  # number of trees
            random_state=42
        )

        # train the model on the data
        rf.fit(trainDataX, trainDataY)

        # Testing phase
        testDataX = testData[:, 0:columnsData - 1]

        # Make predictions on the test data
        # make predictions on the data
        predicted_rf_Self = rf.predict(testDataX)
        # predicted_knn_Self = knn_regressor.predict(testDataX)
        predictedSelf = float(predicted_rf_Self[0])
        predictedSelfArray.append(predictedSelf)

        # Actual values
        actualValue = testData[0, columnsData - 1]

        actualArray.append(actualValue)

        ## Predicted values

        # print(i)
        i += 1


    #  Define and compute evaluation metrics
    MAE = mean_absolute_error(actualArray, predictedSelfArray)
    MSE = mean_squared_error(actualArray, predictedSelfArray)
    RMSE = np.sqrt(mean_squared_error(actualArray, predictedSelfArray))
    R2 = r2_score(actualArray, predictedSelfArray)

    # Compute execution time in seconds
    endTime = time.time()
    executionTime = endTime - startTime

    # Store calculated evaluation metrics and other information
    print("File:", fileIndex, ", MAE:", MAE, ", MSE:", MSE, ", RMSE:", RMSE, ", R2:", R2, ", execution time:", executionTime)
    metrics.write(str(fileIndex) + "," + str(MAE) + "," + str(MSE) + "," + str(RMSE) + "," + str(R2) + "," + str(executionTime) + "\n")

    # Store actual and predicted values in text result files for each input file
    resultsFileName = 'eval_RF_Results_'+ fileName
    resultsFile = open(resultsFileName, "w")

    for p in range(len(actualArray)):
        resultsFile.write(str(actualArray[p]) + "," + str(predictedSelfArray[p]) + "\n")
        #print(p, ",", actualArray[p], ",", predictedSelfArray[p])

    resultsFile.close()

metrics.close()