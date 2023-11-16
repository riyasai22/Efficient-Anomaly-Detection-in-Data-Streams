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

Using LSTM autoencoders for anomaly detection in data streams offers several merits and efficiencies, especially in handling high-volume, high-velocity data with temporal dependencies. Here are some key advantages and capabilities of LSTM autoencoders in this context:

1. Temporal Dependency Handling:
LSTM (Long Short-Term Memory) networks are well-suited for capturing temporal dependencies in sequential data. In anomaly detection for data streams, this capability is crucial for identifying abnormal patterns that evolve over time, such as seasonal variations and concept drift.

2. Ability to Handle Seasonal Variations:
In the context of the Household Electric Power Consumption dataset with measurements over four years, LSTM autoencoders excel at capturing seasonal patterns and long-term dependencies in multivariate time series data. They can learn and model complex temporal relationships, enabling the detection of anomalies that deviate from expected seasonal behaviors.

3. Adaptability to Concept Drift:
Concept drift refers to changes in the underlying data distribution over time. LSTM autoencoders can adapt to such changes by continuously updating their internal representations, allowing them to adjust to new data distributions. This adaptability is crucial for detecting anomalies in dynamic environments where the characteristics of normal behavior may evolve.

4. Model Efficiency and Real-time Processing:
LSTM autoencoders can efficiently process streaming data due to their sequential nature and ability to retain historical context. This capability enables real-time anomaly detection by processing incoming data samples incrementally and swiftly identifying deviations from learned patterns without the need for retraining the entire model.

5. Multivariate Anomaly Detection:
For multivariate datasets like the Household Electric Power Consumption dataset, LSTM autoencoders can effectively capture correlations and interactions among multiple variables. This allows the model to detect anomalies that involve complex relationships across different features.

6. Model Update and Online Learning:
LSTM autoencoders can be updated with new data, allowing for online learning. This feature enables the model to continuously refine its understanding of normal patterns and adapt to changes in the data distribution, enhancing its anomaly detection performance over time.

## Visualization
[1] Anomaly detection using LSTM autoencoders(Static)
![image](https://github.com/riyasai22/Efficient-Anomaly-Detection-in-Data-Streams/assets/80235375/447c3a0c-8abd-4a0f-bd5a-7208a8a1081c)

[2] Improvised LSTM autoencoders for Anomaly Detection in Real-time data streams 
![image](https://github.com/riyasai22/Efficient-Anomaly-Detection-in-Data-Streams/assets/80235375/f2325ded-0e66-4b16-82fd-46b7197e3132)

- Inital 15% data from the stream is taken for intialization of the model
- Data Stream is generated using the CustomDS function
- LSTM model runs and detects anomalies on the remaining data stream


## Experiment Findings:
Based on experiments conducted on the Household Electric Power Consumption dataset, the LSTM autoencoder model showcased its ability to efficiently identify anomalies amidst seasonal variations and concept drift. Its multivariate capabilities allowed it to capture complex patterns within the data, enabling robust anomaly detection across various scenarios present in long-term electrical consumption records. In conclusion, employing LSTM autoencoders for anomaly detection in data streams presents a powerful solution due to their ability to handle temporal dependencies, adapt to changes, process high-velocity data efficiently, and detect anomalies in multivariate datasets with complex patterns.






