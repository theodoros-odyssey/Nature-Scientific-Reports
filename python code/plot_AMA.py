import matplotlib.pyplot as plt
import numpy as np
from readFileToArray import readFile

print("\n#########################")
print("# Arithmetic Method Plots #")
print("###########################")

inputData = readFile('AMA_Results.txt', 'r')

rowsData = len(inputData)
columnsData = len(inputData[0])

x_axis = inputData[:,0:1]
time_axis = inputData[:,1:2]
error_axis = inputData[:,4:5]

fig, (ax1) = plt.subplots(1, 1)

ax1.plot(x_axis, error_axis, 'r')
ax1.legend(['Percentage error'])
ax1.set(xlabel='Number of polynomial variables', ylabel='Prediction percentage error')
#ax1.set_title('Percentage error', y=1.0, pad=-20)

fig.tight_layout()
plt.savefig('AMA_Error_Visual')
plt.show()

fig, (ax2) = plt.subplots(1, 1)
ax2.plot(x_axis, time_axis, 'b')
ax2.legend(['Execution time'])
ax2.set(xlabel='Number of polynomial variables', ylabel='Execution time in seconds')
#ax2.set_title('Execution time', y=1.0, pad=-20)

fig.tight_layout()
plt.savefig('AMA_Time_Visual')
plt.show()