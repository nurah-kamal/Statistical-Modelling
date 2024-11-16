import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Dataset/advertising (1).csv')

plt.figure(figsize=(8, 6))
sns.regplot(x=data['Newspaper'], y=data['Sales'], line_kws={"color": "black"})
plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.savefig("newspaper_vs_sales.png")
plt.close()

print("Newspaper vs Sales plot saved as newspaper_vs_sales.png")

