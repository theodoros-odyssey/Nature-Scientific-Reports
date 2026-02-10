from selfMultipleCore import selfMultipleCore

def selfMultipleModel(trainX,trainY):
    # interception value in self multiple regression is equals to 0
    # dataSetX is the regressand value
    # dataSetY contains regressor values

    # n is the number of regressor values i.e., trainX or trainY columns
    n = len(trainX[0])

    # m is the number of training instances i.e., trainX or trainY rows
    m = len(trainX)

    # Create modelX array
    rows, cols = (m, 2*n)
    modelX = [[0 for i in range(cols)] for j in range(rows)]

    # Create modelY array
    rows, cols = (m, 2)
    modelY = [[0 for i in range(cols)] for j in range(rows)]

    x_m = [] # Empty x input array
    y_m = [] # Empty y input array

    i = 0
    for i in range(m):
        x_m = trainX[i,:n]
        y_m = trainY[i]
        # Invoke self multiple regression
        #y_self, beta_self, e_self = selfMultipleCore(x_m, y_m)
        y_self, beta_self = selfMultipleCore(x_m, y_m)
        j = 0
        p = 0
        while j <= (2 * n) - 1:
            # Build self regression model
            modelX[i][j] = beta_self[p]
            modelX[i][j+1] = x_m[p]
            p += 1
            j += 2

        modelY[i][0] = y_self
        modelY[i][1] = y_m

    return modelX, modelY
