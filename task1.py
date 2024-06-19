#task 1: EXPLORATORY DATA ANALYSIS (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wine = pd.read_csv('/users/kritikap/desktop/eda/winequality-red.csv')
print(wine.head())
print(wine.info())
print(wine.isnull().sum())
print(wine.describe())

sns.set(style="whitegrid")

# Histograms 
plt.figure(figsize=(15, 10))
for i, column in enumerate(wine.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.histplot(wine[column], kde=True)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# Box plots 
plt.figure(figsize=(15, 10))
for i, column in enumerate(wine.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=wine[column])
    plt.title(f'Box plot of {column}')
plt.tight_layout()
plt.show()

# Pair plot
sns.pairplot(wine, hue='quality')
plt.show()

# Correlation matrix
corr_matrix = wine.corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Wine Dataset')
plt.show()
