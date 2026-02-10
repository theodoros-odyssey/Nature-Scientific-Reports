import math

def selfMultiplePredict(modelX, modelY, testX, alpha, beta, delta):
    # n is the number of regressor values i.e., modelX columns
    n = len(modelX[0])

    # m is the number of training instances i.e., modelX or modelY rows
    m = len(modelX)

    p = (n / 2)

    k = len(testX[0])

    # Initialize minimum distance with maximum float number
    min = 1.7976931348623157e+308
    minIndex = 0
    distance = 0

    # neighbors selected for prediction
    neighbors = 0
    # Nearest neighbors distance
    nnDistance = 0
    flag = 0
    # Predicted x and y values
    predictedX = 0
    predictedY = 0
    predictedTrainY = 0

    for i in range(m):
        distance = 0
        j = 0
        r = 1
        while (j < p):
            # Build distance value
            distance += abs(testX[0][j] - modelX[i][r])
            r += 2
            j += 1

        distance = distance / k

        # Find minimum distance
        if abs(distance) <= abs(min):
            min = abs(distance)
            # Define minimum distance index
            minIndex = i

    for i in range(m):
        nnDistance = 0
        j = 0
        r = 1
        while (j < p):
            # Build nnDistance value
            nnDistance += abs(testX[0][j] - modelX[i][r])
            r += 2
            j += 1

        nnDistance = nnDistance / k

        # Find minimum nearest neighbors depending on certain distance threshold
        # which is defined experimentally to reduce self regression model bias
        if abs(nnDistance) <= (delta * abs(min)):

            #print(abs(nnDistance), ",", delta * abs(min))

            minIndex = i
            # neighbors selected for prediction
            neighbors += 1
            j = 0
            r = 0
            while (j < p):
                # Build predictedX vector
                predictedX += (testX[0][j] * modelX[minIndex][r])
                r += 2
                j += 1
            predictedY += modelY[minIndex][0]
            #predictedTrainY += modelY[minIndex][1]
            flag += 1

    predictedX = predictedX / flag
    predictedY = predictedY / flag
    #predictedTrainY = predictedTrainY / flag

    # alpha is defined experimentally and expresses the relation
    # between computed predicted x regressand and values and
    # y regressor value during test phase

    tuning  = (alpha * predictedX) + (beta * predictedY) # + (gamma * predictedTrainY) #modelY[minDistIndex][0])
    predictedValue = tuning

    return predictedValue,neighbors