import numpy as np
from sklearn.naive_bayes import GaussianNB

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

model = GaussianNB()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Hasil Prediksi:")
for i, (sample, pred) in enumerate(zip(X_test, predictions)):
    print(f"Data Uji {i+1}:")
    print(f"Usia: {sample[0]}, Pendapatan: Rp {sample[1]}, Status Pekerjaan: {'Bekerja' if sample[2] == 1 else 'Tidak Bekerja'}")
    print(f"Prediksi: {'Membeli' if pred == 1 else 'Tidak Membeli'}\n")
