from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD


#1.Încărcarea setului de date Fashion-MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



#2. Afișarea distribuției claselor
# Creăm un DataFrame pentru a vizualiza distribuția
class_counts = pd.Series(np.concatenate((y_train,y_test))).value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=class_names, y=class_counts.values)
plt.xticks(rotation=45)
plt.title("Distribuția claselor în setul de date")
plt.ylabel("Număr de imagini")
plt.show()

# 3. Normalizarea pixelilor și one-hot encoding

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

#4. Modelul de referință (Baseline)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='tanh'))
model.add(Dense(10, activation='sigmoid'))


#Optimizer: SGD = Stochastic Gradient Descent.
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='mse',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat,
                    epochs=30,
                    validation_split=0.1)


plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Acuratețea în timp')
plt.xlabel('Epoca')
plt.legend()
plt.grid(True)
plt.show()



