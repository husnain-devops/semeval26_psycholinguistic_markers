import matplotlib.pyplot as plt
import numpy as np
import os

def generate_semeval_chart(output_filename='semeval_markers_chart.png'):
    # ==========================================
    # 1. Data Setup (Included all 5 markers)
    # ==========================================
    # Sorted alphabetically: Action, Actor, Effect, Evidence, Victim
    Markerss = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
    
    # Data extracted from your provided text
    f1_scores = [0.1667, 0.3432, 0.1026, 0.0889, 0.2508]
    recall_scores = [0.2692, 0.7279, 0.1408, 0.1370, 0.5139]

    # ==========================================
    # 2. Chart Configuration & Styling
    # ==========================================
    # Set colors to match the original image
    color_f1 = '#004e9a'      # Dark Blue
    color_recall = '#ff8000'  # Orange
    
    # Set up the figure size (width, height) in inches
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bar positions
    x = np.arange(len(Markerss))  # label locations
    width = 0.35              # width of the bars

    # Create the grouped bars
    rects1 = ax.bar(x - width/2, f1_scores, width, label='F1', color=color_f1, zorder=3)
    rects2 = ax.bar(x + width/2, recall_scores, width, label='Recall', color=color_recall, zorder=3)

    # ==========================================
    # 3. Axes, Labels, and Legend
    # ==========================================
    # Y-Axis styling (Score)
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    # Set ticks every 0.10
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # Format y-labels to two decimal places (e.g., 0.10, 0.20)
    ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(0, 1.1, 0.1)], fontsize=12)

    # X-Axis styling (Markers)
    ax.set_xlabel('Markers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(Markerss, fontsize=13, fontweight='bold')

    # Style the spine (borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add light dashed gridlines behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    # Place legend at the top center, matching the original style
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, 
              fontsize=13, frameon=False, handletextpad=0.5)

    # ==========================================
    # 4. Add Value Labels on top of bars
    # ==========================================
    def add_labels(rects, color):
        for rect in rects:
            height = rect.get_height()
            # Format to 2 decimal places (e.g., 0.34)
            label_text = '{:.2f}'.format(height)
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        color=color, fontweight='bold', fontsize=11)

    add_labels(rects1, color_f1)
    add_labels(rects2, color_recall)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # ==========================================
    # 5. Save the Image as PNG
    # ==========================================
    try:
        # Increase dpi for high resolution
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved chart to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error saving file: {e}")
    finally:
        plt.close(fig) # Close the plot to free memory

if __name__ == '__main__':
    generate_semeval_chart()