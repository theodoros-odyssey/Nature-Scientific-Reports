import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from readFileToArray import readFile

# Optional: mixed precision (for GPUs with Tensor Cores)
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# Silence TensorFlow output
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.keras.utils.disable_interactive_logging()

# ===========================
# CNN Model Builder
# ===========================
def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(16, kernel_size=1, activation='relu'),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model


# ===========================
# Main Evaluation
# ===========================
print("Evaluation results for CNN regression")

metrics = open("eval_CNN_Metrics.txt", "w")
totalFiles = 17  # Adjust if multiple input files

for fileCounter in range(totalFiles):
    startTime = time.time()
    fileIndex = fileCounter + 1
    fileName = f"{fileIndex}.txt"

    inputData = np.array(readFile(fileName, 'r'))
    rowsData, columnsData = inputData.shape

    actualArray = []
    predictedSelfArray = []

    # --- Prebuild model once to avoid retracing ---
    input_shape = (columnsData - 1, 1)
    base_model = build_cnn_model(input_shape)
    initial_weights = base_model.get_weights()

    # --- Warm-up once to build the TensorFlow graph ---
    _ = base_model.predict(np.zeros((1,) + input_shape), verbose=0)

    # ===========================
    # Leave-One-Out Loop
    # ===========================
    for i in range(rowsData):
        # Reset model to initial weights
        cnn = build_cnn_model(input_shape)
        cnn.set_weights(initial_weights)

        # EarlyStopping callback
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0)

        # Use np.delete (faster than concatenating)
        trainData = np.delete(inputData, i, axis=0)
        testData = inputData[i:i + 1, :]

        # Split into X and y
        trainDataX = trainData[:, :-1]
        trainDataY = trainData[:, -1]
        testDataX = testData[:, :-1]
        actualValue = testData[0, -1]

        # Reshape for CNN input
        trainDataX = trainDataX.reshape((trainDataX.shape[0], trainDataX.shape[1], 1))
        testDataX = testDataX.reshape((testDataX.shape[0], testDataX.shape[1], 1))

        # Train CNN (silent)
        cnn.fit(trainDataX, trainDataY, epochs=200, batch_size=8, verbose=0, callbacks=[es])

        # Predict
        predictedSelf = float(cnn.predict(testDataX, verbose=0)[0][0])

        # Store results
        predictedSelfArray.append(predictedSelf)
        actualArray.append(actualValue)

    # ===========================
    # Evaluation Metrics
    # ===========================
    MAE = mean_absolute_error(actualArray, predictedSelfArray)
    MSE = mean_squared_error(actualArray, predictedSelfArray)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(actualArray, predictedSelfArray)
    executionTime = time.time() - startTime

    # Store calculated evaluation metrics and other information
    print("File:", fileIndex, ", MAE:", MAE, ", MSE:", MSE, ", RMSE:", RMSE, ", R2:", R2, ", execution time:",executionTime)
    metrics.write(str(fileIndex) + "," + str(MAE) + "," + str(MSE) + "," + str(RMSE) + "," + str(R2) + "," + str(executionTime) + "\n")

    # Store actual and predicted values in text result files for each input file
    resultsFileName = 'eval_CNN_Results_' + fileName
    resultsFile = open(resultsFileName, "w")

    for p in range(len(actualArray)):
        resultsFile.write(str(actualArray[p]) + "," + str(predictedSelfArray[p]) + "\n")
        #print(p, ",", actualArray[p], ",", predictedSelfArray[p])

    resultsFile.close()

metrics.close()