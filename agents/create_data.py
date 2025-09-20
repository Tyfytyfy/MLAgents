from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Pobieranie zbioru danych Breast Cancer Wisconsin...")
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X_bc = breast_cancer_wisconsin_diagnostic.data.features
y_bc = breast_cancer_wisconsin_diagnostic.data.targets

bc_data = pd.concat([X_bc, y_bc], axis=1)

bc_data.to_csv('bc_data.csv', index=False)
print(f"Zapisano bc_data.csv - wymiary: {bc_data.shape}")

print("\n=== BREAST CANCER METADATA ===")
print(breast_cancer_wisconsin_diagnostic.metadata)
print("\n=== BREAST CANCER VARIABLES ===")
print(breast_cancer_wisconsin_diagnostic.variables)

print("\n" + "="*50 + "\n")

print("Pobieranie zbioru danych Heart Disease...")
heart_disease = fetch_ucirepo(id=45)

X_hd = heart_disease.data.features
y_hd = heart_disease.data.targets

hd_data = pd.concat([X_hd, y_hd], axis=1)

hd_data.to_csv('hd_data.csv', index=False)
print(f"Zapisano hd_data.csv - wymiary: {hd_data.shape}")

print("\n=== HEART DISEASE METADATA ===")
print(heart_disease.metadata)
print("\n=== HEART DISEASE VARIABLES ===")
print(heart_disease.variables)

print("\n=== PODSUMOWANIE ===")
print(f"bc_data.csv: {bc_data.shape[0]} wierszy, {bc_data.shape[1]} kolumn")
print(f"hd_data.csv: {hd_data.shape[0]} wierszy, {hd_data.shape[1]} kolumn")

print("\n=== PODGLĄD bc_data ===")
print(bc_data.head())
print("\n=== PODGLĄD hd_data ===")
print(hd_data.head())