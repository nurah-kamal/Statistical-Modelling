import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Dataset/advertising (1).csv')

plt.figure(figsize=(8, 6))
sns.regplot(x=data['Radio'], y=data['Sales'], line_kws={"color": "black"})
plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.savefig("radio_vs_sales.png")
plt.close()

print("Radio vs Sales plot saved as radio_vs_sales.png")
