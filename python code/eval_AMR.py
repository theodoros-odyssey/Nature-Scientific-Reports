###################################################
### Evaluate Arithmetic Method Regression Model ###
###################################################

# Evaluate arithmetic method regression model
# with leave-one-out cross validation evaluation method
# and MAE, MSE, RMSE, and R2 evaluation metrics

# Import libraries and functions
import time
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from selfMultipleModel import selfMultipleModel
from readFileToArray import readFile
from selfMultiplePredict import selfMultiplePredict

# Print a message
print("Evaluation results for arithmetic method regression")

# Store evaluation metrics and other information
# for arithmetic method regression
# in text file myEvaluateMetricsFull.txt
metrics = open("eval_AMR_Metrics.txt", "w")

## Store only the evaluation metrics
## in text file myEvaluateMetricsLimited.txt
# metricsLimited = open("myEvaluateMetricsLimited.txt", "w")

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

    # Initialize minimum MAE with maximum float number
    MAE = 1.7976931348623157e+308

    # External delta loop up to the maximum number of neighbors
    # upperNeighbors = math.ceil(0.1 * rowsData)
    delta = 1  # delta: minimum value = 1, maximum value = 10 #upperNeighbors, step = 0.1
    while delta <= 10:  # upperNeighbors:
        alpha = 0.1  # alpha: minimum value = 0, maximum value = 1, step = 0.1

        # Internal alpha loop
        while alpha <= 1:

            beta = 1 - alpha  # beta: minimum value = 0, maximum value = 1
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

                modelX, modelY = selfMultipleModel(trainDataX, trainDataY)
                # print(modelY)

                # Testing phase
                testDataX = testData[:, 0:columnsData - 1]

                # print(testDataX)

                # Actual values
                actualValue = testData[0, columnsData - 1]

                actualArray.append(actualValue)

                # Predicted values
                predictedSelf, neighbors = selfMultiplePredict(modelX, modelY, testDataX, alpha, beta, delta)
                predictedSelfArray.append(predictedSelf)

                i += 1

            # Define and compute evaluation metrics
            mae = mean_absolute_error(actualArray, predictedSelfArray)
            mse = mean_squared_error(actualArray, predictedSelfArray)
            rmse = np.sqrt(mean_squared_error(actualArray, predictedSelfArray))
            r2 = r2_score(actualArray, predictedSelfArray)

            # Find minimum MAE
            if mae <= MAE:
                MAE = mae
                MSE = mse
                RMSE = rmse
                R2 = r2
                # Define optimal solution
                optimalAlpha = alpha
                optimalBeta = beta
                optimalDelta = delta
                optimalNeighbors = neighbors
                lengthModelX = len(modelX)
                datasetSize = rowsData
                minActualArray = []
                minPredictedSelfArray = []
                minActualArray = actualArray
                minPredictedSelfArray = predictedSelfArray

            alpha += 0.1

        delta += 0.1

    # Compute execution time in seconds
    endTime = time.time()
    executionTime = endTime - startTime

    # Store calculated evaluation metrics and other information
    print("File:", fileIndex, ", MAE:", MAE, ", MSE:", MSE, ", RMSE:", RMSE, ", R2:", R2, ",alpha:", optimalAlpha,
          ",beta:", optimalBeta, ", delta:", optimalDelta, ", neighbors:", optimalNeighbors, ", model size:",
          lengthModelX, ", dataset size:", datasetSize, ", execution time:", executionTime)
    metrics.write(str(fileIndex) + "," + str(MAE) + "," + str(MSE) + "," + str(RMSE) + "," + str(R2) + "," + str(
        optimalAlpha) + "," + str(optimalBeta) + "," + str(optimalDelta) + "," + str(optimalNeighbors) + "," + str(
        lengthModelX) + "," + str(datasetSize) + "," + str(executionTime) + "\n")

    ## Store only calculated of evaluation metrics
    # metricsLimited.write(str(fileIndex) + "," + str(MAE) + "," + str(MSE) + "," + str(RMSE) + "," + str(R2) + "\n")

    # Store actual and predicted values in text result files for each input file
    resultsFileName = 'eval_AMR_Results_' + fileName
    resultsFile = open(resultsFileName, "w")

    for p in range(len(minActualArray)):
        resultsFile.write(str(minActualArray[p]) + "," + str(minPredictedSelfArray[p]) + "\n")
        # print(p, ",", minActualArray[p], ",", minPredictedSelfArray[p])

    resultsFile.close()

metrics.close()