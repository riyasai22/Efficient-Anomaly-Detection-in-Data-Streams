# Efficient-Anomaly-Detection-in-Data-Streams
Efficient Anomaly Detection in Data Streams built using Python. Performed a comparative study of existing anomaly detection methodologies and selected LSTM based Autoencoders to work best on data streams.

### Dataset:
Household Electric Power Consumption :  Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available.

Link: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

### Stream Emulation: CustomDS function 

```python
class CustomDS:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.current_idx = 0

    def get_next_sample(self):
        if self.current_idx < len(self.data):
            sample = self.data[self.current_idx]
            label = self.labels[self.current_idx]
            self.current_idx += 1
            return sample, label
        else:
            return None, None

stream = CustomDS(stream_data, stream_data_labels)
```

### Detects anomalies on the go (real-time anomaly detection). 

![image](https://github.com/riyasai22/Efficient-Anomaly-Detection-in-Data-Streams/assets/80235375/309edc4f-e153-4575-a841-4c5854243f1d)

### LSTM Autoencoder
```python
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(train_X.shape[1]))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(train_X.shape[2])))
model.compile(optimizer='adam', loss='mse')
```

## Visualtion
![image](https://github.com/riyasai22/Efficient-Anomaly-Detection-in-Data-Streams/assets/80235375/447c3a0c-8abd-4a0f-bd5a-7208a8a1081c)


