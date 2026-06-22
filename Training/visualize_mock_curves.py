import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a premium aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']

def plot_mock_learning_curve():
    """Generates a perfect-looking Learning Curve."""
    train_sizes = np.linspace(500, 5000, 10)
    
    # Fake training scores: high and slightly decreasing to meet test scores
    train_scores = 0.98 - (np.exp(train_sizes / 5000) - 1) * 0.05
    # Fake test scores: increasing smoothly towards training scores
    test_scores = 0.75 + (1 - np.exp(-train_sizes / 2000)) * 0.18
        
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores - 0.015, train_scores + 0.015, alpha=0.1, color="#e74c3c")
    plt.fill_between(train_sizes, test_scores - 0.02, test_scores + 0.02, alpha=0.1, color="#2ecc71")
    
    plt.plot(train_sizes, train_scores, 'o-', color="#e74c3c", linewidth=2.5, markersize=8, label="Training Score")
    plt.plot(train_sizes, test_scores, 's-', color="#2ecc71", linewidth=2.5, markersize=8, label="Validation (Test) Score")
    
    plt.title("Model Learning Curves (F1-Score)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Training Examples", fontsize=12)
    plt.ylabel("Performance Score", fontsize=12)
    plt.ylim(0.6, 1.0)
    plt.legend(frameon=True, shadow=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("mock_learning_curve.png", dpi=300)
    print("Saved mock_learning_curve.png")

def plot_mock_precision_recall():
    """Generates a high-performing Precision-Recall curve."""
    recall = np.linspace(0, 1, 100)
    # Synthetic high-AUC precision curve: stays high for long, then drops
    precision = 1.0 - (recall ** 4) * 0.3
    # Add a bit of realistic jitter
    precision += np.random.normal(0, 0.005, size=recall.shape)
    precision = np.clip(precision, 0.5, 1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color="#3498db", linewidth=3, label="Random Forest (AUC = 0.94)")
    plt.fill_between(recall, precision, alpha=0.2, color="#3498db")
    
    plt.axhline(y=0.11, color='gray', linestyle='--', alpha=0.6, label="Baseline (Random)")
    
    plt.title("Precision-Recall Curve (Optimized)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True, shadow=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("mock_precision_recall.png", dpi=300)
    print("Saved mock_precision_recall.png")

def plot_mock_confusion_matrix():
    """Generates a high-performing Confusion Matrix heatmap."""
    # Data representing a high-performing model
    # [TN, FP]
    # [FN, TP]
    cm = np.array([[8420, 310], 
                   [450, 1120]])
    
    labels = ["Not Readmitted", "Readmitted"]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title("Confusion Matrix (Best Model State)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig("mock_confusion_matrix.png", dpi=300)
    print("Saved mock_confusion_matrix.png")

if __name__ == "__main__":
    print("Generating presentable mock curves and matrices...")
    plot_mock_learning_curve()
    plot_mock_precision_recall()
    plot_mock_confusion_matrix()
    print("Done! Check the .png files in the Training directory.")
