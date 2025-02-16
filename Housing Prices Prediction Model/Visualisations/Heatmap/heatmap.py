import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('DataSet/AmesHousing.csv')

# Clean column names 
data.columns = data.columns.str.strip()

print(data.columns)


required_columns = ['SalePrice', 'Gr Liv Area', 'Overall Qual']  
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")


correlation_matrix = data[required_columns].corr()

# Visualization: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")

# Save the plot to an image
plt.savefig("Correlation_Heatmap.png")
plt.close()
print("Correlation_Heatmap.png saved.")

