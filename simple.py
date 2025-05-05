import numpy as np

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

X_train = np.array([
    [25, 5000000, 1],
    [30, 8000000, 1],
    [35, 12000000, 1],
    [20, 3000000, 0],
    [45, 15000000, 1],
    [22, 4000000, 0],
    [28, 6000000, 1],
    [40, 10000000, 1],
    [19, 2000000, 0],
    [32, 9000000, 1]
])

y_train = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

X_test = np.array([
    [27, 5500000, 1],
    [33, 9500000, 1],
    [21, 3500000, 0],
    [38, 11000000, 1]
])

model = NaiveBayes()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Hasil Prediksi:")
for i in range(len(X_test)):
    usia = X_test[i][0]
    pendapatan = X_test[i][1]
    status = 'Bekerja' if X_test[i][2] == 1 else 'Tidak Bekerja'
    hasil = 'Membeli' if preds[i] == 1 else 'Tidak Membeli'

    print(f"Data Uji {i+1}: Usia={usia}, Pendapatan=Rp{pendapatan}, Status={status}")
    print(f"Prediksi: {hasil}\n")
