import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        for cls, count in zip(classes, counts):
            self.class_priors[cls] = count / total_samples
        
        for feature_idx in range(X.shape[1]):
            for cls in classes:
                X_cls = X[y == cls]
                
                unique_values, value_counts = np.unique(X_cls[:, feature_idx], return_counts=True)
                total_cls_samples = len(X_cls)
                
                for value, count in zip(unique_values, value_counts):
                    self.feature_probs[feature_idx][cls][value] = count / total_cls_samples
    
    def predict(self, X):
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            for cls in self.class_priors:
                score = np.log(self.class_priors[cls])
                
                for feature_idx, value in enumerate(sample):
                    if value in self.feature_probs[feature_idx][cls]:
                        score += np.log(self.feature_probs[feature_idx][cls][value])
                    else:
                        score += np.log(1e-10)
                
                class_scores[cls] = score
            
            predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])
        
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

nb = NaiveBayes()
nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

print("Hasil Prediksi:")
for i, (sample, pred) in enumerate(zip(X_test, predictions)):
    print(f"Data Uji {i+1}:")
    print(f"Usia: {sample[0]}, Pendapatan: Rp {sample[1]}, Status Pekerjaan: {'Bekerja' if sample[2] == 1 else 'Tidak Bekerja'}")
    print(f"Prediksi: {'Membeli' if pred == 1 else 'Tidak Membeli'}\n") 