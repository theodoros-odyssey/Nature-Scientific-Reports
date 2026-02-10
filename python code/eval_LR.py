import time
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from readFileToArray import readFile

# Evaluation Metrics for Weka Regression Models
print("Evaluation results for linear regression")

#Store evaluation values in text file
metrics = open("eval_LR_Metrics.txt", "w")

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

        # Train normal equation for linear regression
        thetaParameters = np.linalg.pinv(trainDataX.T.dot(trainDataX)).dot(trainDataX.T).dot(trainDataY)

        columnsTheta = len(thetaParameters)

        # Testing phase
        testDataX = testData[:, 0:columnsData - 1]

        # Actual values
        actualValue = testData[0, columnsData - 1]

        actualArray.append(actualValue)

        # Predicted values
        predictedSelf = 0
        for j in range(columnsTheta):
            predictedSelf += thetaParameters[j] * testDataX[0][j]

        predictedSelfArray.append(predictedSelf)

        i += 1

    # Define and compute evaluation metrics
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
    resultsFileName = 'eval_LR_Results_'+ fileName
    resultsFile = open(resultsFileName, "w")

    for p in range(len(actualArray)):
        resultsFile.write(str(actualArray[p]) + "," + str(predictedSelfArray[p]) + "\n")
        #print(p, ",", actualArray[p], ",", predictedSelfArray[p])

    resultsFile.close()

metrics.close()
