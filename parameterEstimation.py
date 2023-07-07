import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D,Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
from tensorflow.keras.models import save_model,load_model

path=os.getcwd()

data=pd.read_csv('labels.csv')

file_list=data.file

waveform=[]
param_=[data.m1,data.m2,data.d]
for i in file_list:
    temp_file=np.load(f'Data/GW/{i}')
    x_data.append(temp_file)

plt.plot(waveform[0])
plt.title('Waveform')
plt.show()
# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(waveform_, param, test_size=0.2, random_state=42)
print(x_train[0].shape)
# Define the model architecture
reg_model = Sequential()
reg_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(4096, 1)))
reg_model.add(MaxPooling1D(pool_size=2))
reg_model.add(Dropout(0.2))
reg_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
reg_model.add(MaxPooling1D(pool_size=2))
reg_model.add(Dropout(0.2))
reg_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
reg_model.add(MaxPooling1D(pool_size=2))
reg_model.add(Dropout(0.2))
reg_model.add(Flatten())
reg_model.add(Dense(64, activation='relu'))
reg_model.add(Dense(3, activation='linear'))

reg_model.summary()

# Compile the model
optimizer = Adam(lr=1e-4)
reg_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Train the model
history = reg_model.fit(x_train[..., np.newaxis], y_train, batch_size=32, epochs=50, validation_data=(x_val[..., np.newaxis], y_val),callbacks=[early_stop])

error, loss = reg_model.evaluate(x_val[..., np.newaxis], y_val)
print(f"Validation loss: {loss:.4f}, Validation error: {error:.4f}")
save_model(reg_model,'reg_model')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict the mass and distance parameters using the trained model
y_pred = reg_model.predict(x_val[..., np.newaxis])

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_val, y_pred)
print("MAE:", mae)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_val, y_pred)
print("MSE:", mse)

# Calculate root mean squared error (RMSE)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Calculate R-squared (R^2)
r2 = r2_score(y_val, y_pred)
print("R^2:", r2)

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model error')
plt.ylabel('error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()