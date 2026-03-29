import numpy as np

# Matris tanımlama
A = np.array([[4, 2],
              [1, 3]])

# Özdeğer ve özvektör hesaplama
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Özdeğerler:")
print(eigenvalues)

print("\nÖzvektörler:")
print(eigenvectors)
