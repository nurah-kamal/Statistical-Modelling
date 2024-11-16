import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Dataset/advertising (1).csv')

plt.figure(figsize=(8, 6))
sns.regplot(x=data['TV'], y=data['Sales'], line_kws={"color": "black"})
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.savefig("tv_vs_sales.png")
plt.close()

print("TV vs Sales plot saved as tv_vs_sales.png")

