import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r'F:\Sem 5\Artificial Intelligence\Code\Product_data.csv')
print(df)
print(df.shape)

# Get the features
x = df.iloc[0:36].values

# Extract the columns
z = df['availability (%)']
e = df['price ($)']
p = df['rating']
b = df['number of reviews']
t = df['discount']

print(z)
print(e)
print(p)
print(b)
print(t)

# Create target variable
a = []
for i, j, k, r, q in zip(e, z, p, b, t):
    if (i < 150 and j >= 70 and k >= 4 and r > 100 and q == 1):
        a.append(1)
    else:
        a.append(0)

print("Target:", set(a))
print(a) 

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, a, test_size=0.11, random_state=102)
print("y_train classes: ", set(y_train))
print("y_test classes: ", set(y_test))

# Train the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Print predictions and accuracy
print("Y_Predict is: ",y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Plot test data
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
plt.xlabel('Features')
plt.ylabel('Target')
plt.title('Test data')
plt.show()

# Plot train data
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.xlabel('Features')
plt.ylabel('Target')
plt.title('Train data')
plt.show()
