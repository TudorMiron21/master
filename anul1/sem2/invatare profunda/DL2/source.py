from tensorflow.keras.datasets import boston_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#2. Reprezentarea distribuțiilor variabilelor predictive

df = pd.DataFrame(x_train, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

# df.hist(bins=30, figsize=(15, 10))
# plt.suptitle("Distribuțiile variabilelor predictive", fontsize=16)
# plt.show()



# # 3. Distribuția variabilei țintă
# sns.histplot(y_train, bins=30, kde=True)
# plt.title("Distribuția variabilei țintă (prețul casei)")
# plt.xlabel("Prețul casei ($1000s)")
# plt.ylabel("Frecvență")
# plt.grid(True)
# plt.show()

# print(pd.Series(y_train).value_counts())


# q1, q3 = np.percentile(y_train, [25, 75])
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
# outliers = y_train[(y_train < lower_bound) | (y_train > upper_bound)]

# # Create figure
# plt.figure(figsize=(10, 8))

# # Boxplot
# sns.boxplot(x=y_train, color='skyblue')

# # Annotate the bounds
# plt.axvline(q1, color='orange', linestyle='--', label='Q1')
# plt.axvline(q3, color='green', linestyle='--', label='Q3')
# plt.axvline(lower_bound, color='red', linestyle=':', label='Lower Bound')
# plt.axvline(upper_bound, color='red', linestyle=':', label='Upper Bound')

# # Titles and labels
# plt.title("Boxplot al valorilor țintă (y_train) cu IQR și outlieri")
# plt.xlabel("Valoare")
# plt.legend()
# plt.grid(True, axis='x')

# plt.show()

# 4. Set de validare (20% din antrenare)
x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# 5. Model de bază
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(13,)),
#     Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# history = model.fit(x_train_sub, y_train_sub, validation_data=(x_val, y_val), epochs=100, verbose=0)


#  6. Experiment: scalare

scaler = MinMaxScaler(feature_range=(-1, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_scaled = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(1)
])
model_scaled.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_scaled.fit(x_train_scaled, y_train, epochs=100, verbose=0)

loss, mae = model_scaled.evaluate(x_test_scaled, y_test)
print(f"Loss (MSE): {loss:.2f}, MAE: {mae:.2f}")


# 7. Experiment: funcții de 
# MSE
model_mse = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(1)
])
model_mse.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_mse.fit(x_train_scaled, y_train, epochs=100, verbose=0)

# MAE
model_mae = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(1)
])
model_mae.compile(optimizer='adam', loss='mae', metrics=['mae'])
model_mae.fit(x_train_scaled, y_train, epochs=100, verbose=0)

# Huber
model_huber = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(1)
])
model_huber.compile(optimizer='adam', loss=Huber(), metrics=['mae'])
model_huber.fit(x_train_scaled, y_train, epochs=100, verbose=0)
