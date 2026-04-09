import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory
os.makedirs('diagrams', exist_ok=True)

# 1. Omission vs Framing Pie Chart
plt.figure(figsize=(8, 6))
labels = ['Fact Omission\n(Selection Bias)', 'Rhetorical Framing\n(RST Structure)']
sizes = [72, 28]
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
plt.title('Predictive Power Contribution', fontsize=16, weight='bold')
plt.axis('equal')  
plt.tight_layout()
plt.savefig('diagrams/omission_vs_framing.png', dpi=300)
plt.close()

# 2. Cluster Distribution Bar Chart
plt.figure(figsize=(10, 6))
categories = ['All 3 Sides', 'Left + Center', 'Center + Right', 'Left + Right']
percentages = [18.8, 43.6, 37.6, 0.0]
colors = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd']

bars = plt.bar(categories, percentages, color=colors)
plt.title('Distribution of Fact Clusters', fontsize=16, weight='bold')
plt.ylabel('Percentage of Clusters (%)', fontsize=14)
plt.xticks(fontsize=12, weight='bold')
plt.ylim(0, 50)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('diagrams/cluster_distribution.png', dpi=300)
plt.close()

# 3. Performance Drop Bar Chart
plt.figure(figsize=(10, 6))
models = ['Full Dataset\n(Coverage + Framing)', '3-Way Clusters Only\n(Framing Only)', 'Random Chance Baseline']
accuracies = [89.77, 61.46, 50.00]
colors = ['#17becf', '#ff7f0e', '#7f7f7f']

bars = plt.bar(models, accuracies, color=colors, width=0.6)
plt.title('Classification Accuracy Drop-off', fontsize=16, weight='bold')
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.ylim(0, 100)
plt.xticks(fontsize=12, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('diagrams/performance_drop.png', dpi=300)
plt.close()

print("Diagrams generated successfully in 'diagrams/' directory.")
