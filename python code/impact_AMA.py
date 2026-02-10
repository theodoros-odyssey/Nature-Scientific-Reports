import time
#import matplotlib.pyplot as plt
from numpy import *

print("###############################")
print("# Arithmetic Method Algorithm #")
print("###############################")


# Set variables for plotting results with scatter plot
iterationPlotValues = []
aggregationErrorPercentagePlotValues = []
executionTimePlotValues = []

# Set For loop from i = [1, n], where n = 1,000,000
# and ai, gamma randomly belong to [-1000.000, 1000.000]

#Store errorPercentage values in text file visual.txt
visual = open("AMA_Results.txt", "w")

# Initialize gamma and beta
gamma = 0
beta = 0

# Set j = n + 1, where n is the number of polyonym variables
# and minimum value of j = 2
j = 1000001

# Set variable iteration to iterate internal loop iteration - 1 times
# and aggregate errorPercentage to minimize random errors
for n in range(1, j):
    iterationPlotValues.append(n)
    # Set start time to compute execution time
    startTime = time.time()
    aggregationErrorPercentage = 0
    aggregationExecutionTime = 0

    # Set ai random values within a range of [-1000.000, 1000.000]s
    ai = []

    # Repeat loop which checks if at least one ai random value is not equal to zero
    zeroCheck = 0
    while zeroCheck == 0:
        i = 1
        while i <= n:
            aiValue = float(random.uniform(-1000.000, 1000.000))
            #aiValue = float(truncate(random.uniform(-1000.000, 1000.000), 3))
            ai.append(aiValue)
            i += 1

        i = 1
        zero = 0
        while i <= n:
            #print(ai[i-1])
            if ai[i - 1] == 0:
                zero += 1
            i += 1

        if zero == n:
            zeroCheck = 0
            # Since all ai values are equal to zero reset ai to empty
            ai = []
        else:
            zeroCheck = 1

    # Set random gamma value
    gamma = float(random.uniform(-1000.000, 1000.000))

    # Compute xi solutions
    xi = []

    i = 1
    xValue = 0
    while i <= n:
        xValue = gamma / (ai[i - 1] * n)
        xi.append(xValue)
        i += 1

    # Compute beta value
    i = 1
    beta = 0
    while i <= n:
        beta += ai[i - 1] * xi[i - 1]
        i += 1

    # Compute error percentage
    error = abs((beta - gamma) / gamma)
    errorPercentage = error * 100
    aggregationErrorPercentage += errorPercentage

    # Compute execution time in seconds
    endTime = time.time()
    executionTime = endTime - startTime
    aggregationExecutionTime += executionTime

    #aggregationExecutionTime = aggregationExecutionTime / (iteration - 1)
    executionTimePlotValues.append(aggregationExecutionTime)
    #aggregationErrorPercentage = aggregationErrorPercentage / (iteration - 1)
    aggregationErrorPercentagePlotValues.append(aggregationErrorPercentage)
    #aggregationErrorPercentage = float(truncate(aggregationErrorPercentage, 3))
    #print(aggregationErrorPercentage)
    visual.write(str(n) + "," + str(aggregationExecutionTime) + "," + str(gamma) + "," + str(beta) + "," + str(aggregationErrorPercentage) + "\n")

visual.close()

#xpoints = np.array([1, 8])
#ypoints = np.array([3, 10])

#fig, (ax1) = plt.subplots(1, 1)

#ax1.plot(iterationPlotValues, aggregationErrorPercentagePlotValues, 'r')
#ax1.legend(['Percentage error'])
#ax1.set(xlabel='Number of polynomial variables', ylabel='Prediction percentage error')
#ax1.set_title('Percentage error', y=1.0, pad=-20)

#fig.tight_layout()
#plt.savefig('AMA_1_Visual')
#plt.show()

#fig, (ax2) = plt.subplots(1, 1)
#ax2.plot(iterationPlotValues, executionTimePlotValues, 'b')
#ax2.legend(['Execution time'])
#ax2.set(xlabel='Number of polynomial variables', ylabel='Execution time in seconds')
#ax2.set_title('Execution time', y=1.0, pad=-20)

#fig.tight_layout()
#plt.savefig('AMA_2_Visual')
#plt.show()