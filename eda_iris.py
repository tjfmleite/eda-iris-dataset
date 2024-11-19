import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
file_name = "iris.csv"
urllib.request.urlretrieve(url, file_name)
print(f"Dataset baixado e salvo como {file_name}.")


column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(file_name, header=None, names=column_names)

print("\nDataset - Primeiras Linhas:")
print(data.head())


print("\nInformações Básicas do Dataset:")
print(data.info())


print("\nResumo Estatístico:")
print(data.describe())


print("\nVerificar Valores Ausentes:")
print(data.isnull().sum())


print("\nDistribuição das Classes:")
print(data['species'].value_counts())


sns.pairplot(data, hue='species', diag_kind='kde', palette='husl')
plt.suptitle("Gráfico Pairplot - Dataset Iris", y=1.02)
plt.show()


correlation_matrix = data.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlação das Variáveis")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='species', y='sepal_length', palette='Set2')
plt.title("Comprimento da Sépala por Espécie")
plt.show()

sns.boxplot(data=data, x='species', y='sepal_width', palette='Set3')
plt.title("Largura da Sépala por Espécie")
plt.show()


data.iloc[:, :-1].hist(figsize=(10, 8), bins=15, edgecolor='black')
plt.suptitle("Histogramas das Variáveis", y=0.93)
plt.show()


grouped_stats = data.groupby('species').agg(['mean', 'std'])
print("\nEstatísticas Agrupadas por Espécie (Média e Desvio Padrão):")
print(grouped_stats)


output_file = "iris_summary_stats.csv"
grouped_stats.to_csv(output_file)
print(f"Estatísticas salvas no arquivo '{output_file}'.")
