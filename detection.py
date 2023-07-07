import glob
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

labels=[]
GW_file_list =golb.glob(path+'\Data\GW\*.npy')
for i in range(len(GW_file_list)):
    labels.append(1)

NGW_file_list =golb.glob(path+'\Data\NGW\*.npy')
for i in range(len(NGW_file_list)):
    labels.append(0)

def read_npy_file(item):
    data = np.load(item.decode())
    return data.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices([GW_file_list,NGW_file_list ])

dataset = dataset.map(lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,])))

x_train,y_train,x_Test,y_test=train_test_split([dataset,labels])

print(x_train[0].shape)

# Define the model architecture
clas_model = Sequential()
clas_model.add(Conv1D(16, 3, activation='relu', input_shape=(4096, 1)))
clas_model.add(Conv1D(16, 3, activation='relu'))
clas_model.add(MaxPooling1D(2))
clas_model.add(Conv1D(32, 3, activation='relu'))
clas_model.add(Conv1D(32, 3, activation='relu'))
clas_model.add(MaxPooling1D(2))
clas_model.add(Flatten())
clas_model.add(Dense(64, activation='relu'))
clas_model.add(Dense(1, activation='sigmoid'))

clas_model.summary()

# Compile the model
clas_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_accuracy', patience=2)

# Train the model
history=clas_model.fit(x_train[..., np.newaxis], y_train, batch_size=32, epochs=50, validation_data=(x_val[..., np.newaxis], y_val), callbacks=[early_stop])

# Evaluate the model on the validation set
loss, accuracy = clas_model.evaluate(x_val[..., np.newaxis], y_val)
print(f"Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}")

save_model(clas_model,'clas_model')

# Make predictions on the test set
y_pred = clas_model.predict(x_val[..., np.newaxis])
y_pred = (y_pred > 0.5)

# Generate the classification report
print(classification_report(y_val, y_pred))

# Generate the confusion matrix
sns.heatmap(confusion_matrix(y_val, y_pred),annot=True)

# Calculate the F1 score
f1 = f1_score(y_val, y_pred)

# Calculate the F1 score
accuracy=accuracy_score(y_val, y_pred)

print("F1 score:", f1)
print("Accuracy score:", accuracy)

"plotting the metrics"

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




