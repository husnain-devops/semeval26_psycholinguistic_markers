import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# The derived confusion matrix values
cm_data = np.array([[43, 7], 
                    [6, 21]])

# Create a DataFrame for seaborn
df_cm = pd.DataFrame(cm_data, index=["no", "yes"], columns=["no", "yes"])

# Plotting setup
plt.figure(figsize=(10, 8))
sns.set_theme(style="white") # Matches the clean background of your example

# Create the heatmap
ax = sns.heatmap(df_cm, 
                 annot=True, 
                 fmt='d', 
                 cmap='Blues', 
                 cbar_kws={'label': 'Count'})

# Labels and Title



# Labels and Title
ax.set_ylabel('True Label', fontsize=13)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_title('Confusion Matrix - Dev Set (RoBERTa-large, LoRA r=16)', 
             fontsize=16, 
             fontweight='bold')

plt.tight_layout()

# Save the figure as a high-resolution PNG (perfect for LaTeX)
plt.savefig('confusion_matrix_lora.png', dpi=300, bbox_inches='tight')

# Display the figure
plt.show()