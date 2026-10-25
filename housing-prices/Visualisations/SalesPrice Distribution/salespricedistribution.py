# saleprice_distribution.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('DataSet/AmesHousing.csv')


if 'SalePrice' not in data.columns:
    raise ValueError("The dataset does not contain a 'SalePrice' column.")

# Visualization: SalePrice Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True, color='blue', bins=30)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")

# Save the plot to an image
plt.savefig("SalePrice_Distribution.png")
plt.close()
print("SalePrice_Distribution.png saved.")
