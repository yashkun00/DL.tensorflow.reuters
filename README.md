# DL.tensorflow.reuters
model trainning with tensorflow.reuters

## CODE
from tensorflow.keras.datasets import reuters
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)
model = Sequential([ Embedding(10000, 16, input_length=100),Flatten(), Dense(64
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics
model.fit(x_train, y_train, epochs=10, batch_size=256)
model.evaluate(x_test, y_test)
print("predictions =", model.predict(x_test[0:5]))
