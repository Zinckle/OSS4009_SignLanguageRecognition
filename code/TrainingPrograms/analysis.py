import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf

# Mapping of numeric labels to sign language characters
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
             10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
             19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

test_data = "../Dataset/sign_mnist_test/sign_mnist_test.csv"

test_df = pd.read_csv(test_data)

y_test = test_df['label']
del test_df['label']

label_binarizer = LabelBinarizer()
y_test = label_binarizer.fit_transform(y_test)

x_test = test_df.values

x_test = x_test / 255

x_test = x_test.reshape(-1, 28, 28, 1)

model1 = tf.keras.models.load_model('../smnist.h5')

preds_probs = model1.predict(x_test)
preds = np.argmax(preds_probs, axis=1)  # Find the class with the highest probability for each prediction
print("Accuracy: ", accuracy_score(np.argmax(y_test, axis=1), preds))
print(model1.metrics)
cf_matrix = confusion_matrix(np.argmax(y_test, axis=1), preds)

# Replace axis labels with SMNIST equivalent
class_labels = [label_map[i] for i in range(len(label_map))]
plt.figure(figsize=(12, 10))  # Increase figure size
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, cmap='viridis', annot_kws={"fontsize": 8},
            xticklabels=class_labels, yticklabels=class_labels)  # Normalize and change colormap, adjust font size, replace axis labels
plt.title("Confusion Matrix")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)   # Rotate y-axis labels for better readability
plt.show()
