import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        n_samples = len(y)
        self.classes = np.unique(y)
   
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / n_samples
        
        n_features = X.shape[1]
        self.feature_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.feature_probs[c] = {}

            for i in range(n_features):
                values, counts = np.unique(X_c[:, i], return_counts=True)
                total = len(X_c)
                probs = {}

                for v, count in zip(values, counts):
                    probs[v] = count / total

                self.feature_probs[c][i] = probs

    def predict(self, X):
        predictions = []

        for x in X:
            class_scores = {}

            for c in self.classes:
                score = self.class_probs[c]

                for i, value in enumerate(x):
                    if value in self.feature_probs[c][i]:
                        score *= self.feature_probs[c][i][value]
                    else:
                        score *= 1e-5  

                class_scores[c] = score
            
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)

        return predictions

df = pd.read_csv('data/train.csv')
ts = pd.read_csv('data/test.csv')

X_train = df[["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]].to_numpy()

y_train = df[["Quality"]].to_numpy().ravel()

X_test = ts[["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]].to_numpy()

model = NaiveBayes()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Hasil Prediksi:")
for i in range(len(X_test)):
    print(f"Data {i+1}: {preds[i]}")
    
