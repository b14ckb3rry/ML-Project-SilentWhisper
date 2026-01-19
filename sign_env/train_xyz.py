import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data_dict = pickle.load(open("data_xyz.pickle", "rb"))

data = np.array(data_dict["data"])
labels = np.array(data_dict["labels"])

print("Feature length:", data.shape[1])  # MUST be 63

# Split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, shuffle=True
)

# Train
model = RandomForestClassifier(n_estimators=200)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Save model
with open("model_xyz.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved as model_xyz.p")
