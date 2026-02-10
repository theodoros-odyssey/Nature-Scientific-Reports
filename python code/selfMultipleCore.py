def selfMultipleCore(x,y):

    # interception value is considered to be equals to 0
    # y is the regressand value
    # x contains regressor values

    # n is the number of regressor values
    n = len(x)

    # Check for missing x values
    missingFlag = 0
    missing = n
    i = 1
    while i <= n:
        if x[i-1] == 0:
            missing -= 1
            missingFlag = 1
        i += 1

    # b is the regression coefficient vector
    if missingFlag == 1:
        b = []
        i = 1
        while i <= n:
            if x[i-1] == 0:
                b.append(0)
            else:
                b.append(y / (x[i - 1] * missing))
            i += 1
    else:
        b = []
        i = 1
        while i <= n:
            b.append(y / (x[i - 1] * n))
            i += 1

    # y_est is the estimated regressand value
    y_est = 0
    i = 1
    while i <= n:
        y_est += b[i - 1] * x[i - 1]
        i += 1

    return y_est, b