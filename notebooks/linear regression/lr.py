
import json
import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = "/local/s3905020/notebooks/linear regression/cwe_code_before_min100.jsonl" 
RESULTS_DIR = Path("./results")
FIGURES_DIR = Path("./figures")
RANDOM_SEED = 42

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Load and Preprocess Data
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

# Load all data
all_data = load_jsonl(DATA_PATH)
print(f"Loaded {len(all_data)} samples")

# Extract texts and labels
texts = [item['code'] for item in all_data]
labels = [item['cwe'] for item in all_data]

# Get unique labels
unique_labels = sorted(list(set(labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_classes = len(unique_labels)

print(f"Number of CWE classes: {num_classes}")
print(f"Classes: {unique_labels}")

# Encode labels
encoded_labels = np.array([label2id[label] for label in labels])
# Split Data: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, encoded_labels, 
    test_size=0.3, 
    random_state=RANDOM_SEED,
    stratify=encoded_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=y_temp
)
print(f"Train: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")

# Class distribution
from collections import Counter
train_dist = Counter([id2label[y] for y in y_train])
print(f"   Train distribution: {dict(train_dist)}")

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    token_pattern=r'\b\w+\b',
    lowercase=True,
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    verbose=1
)

start_time = time.time()
model.fit(X_train_tfidf, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# Evaluate on All Sets
def evaluate_set(X, y_true, set_name):
    """Evaluate on a dataset"""
    print(f"\n{set_name} Set:")
    start_time = time.time()
    y_pred = model.predict(X)
    inference_time = (time.time() - start_time) / len(y_true) * 1000  # ms per sample
    
    # metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Inference: {inference_time:.4f} ms/sample")

    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'inference_time_ms': float(inference_time),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            id2label[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(num_classes)
        }
    }

# Evaluate on all sets
train_results = evaluate_set(X_train_tfidf, y_train, "Train")
val_results = evaluate_set(X_val_tfidf, y_val, "Validation")
test_results = evaluate_set(X_test_tfidf, y_test, "Test")

# Save Results
final_results = {
    'model_name': 'Logistic Regression',
    'total_parameters': X_train_tfidf.shape[1],  # Number of features
    'trainable_parameters': X_train_tfidf.shape[1] * num_classes,  # Features × Classes
    'training_time_s': training_time,
    'accuracy': test_results['accuracy'],
    'precision_macro': test_results['precision_macro'],
    'recall_macro': test_results['recall_macro'],
    'f1_macro': test_results['f1_macro'],
    'inference_time_ms': test_results['inference_time_ms'],
    'per_class_metrics': test_results['per_class_metrics'],
    'confusion_matrix': test_results['confusion_matrix'],
    'train_results': train_results,
    'val_results': val_results,
    'test_results': test_results
}


with open(RESULTS_DIR / 'lr_baseline_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"Saved results to {RESULTS_DIR / 'lr_baseline_results.json'}")

with open(RESULTS_DIR / 'lr_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(RESULTS_DIR / 'lr_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open(RESULTS_DIR / 'lr_label_mapping.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
print(f"Saved model to {RESULTS_DIR / 'lr_model.pkl'}")

# Generate Visualizations
# Confusion Matrix
cm = np.array(test_results['confusion_matrix'])
cwe_labels = [f'CWE-{id2label[i]}' for i in range(num_classes)]

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=cwe_labels, yticklabels=cwe_labels,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted CWE', fontweight='bold', fontsize=12)
plt.ylabel('True CWE', fontweight='bold', fontsize=12)
plt.title('Confusion Matrix - Logistic Regression', fontweight='bold', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrix_lr.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'confusion_matrix_lr.pdf', bbox_inches='tight')
print(f"Saved confusion matrix to {FIGURES_DIR / 'confusion_matrix_lr.png'}")
plt.close()

# Per-CWE Performance
per_cwe_data = []
for cwe, metrics in test_results['per_class_metrics'].items():
    per_cwe_data.append({
        'CWE': f'CWE-{cwe}',
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'Support': metrics['support']
    })

df_per_cwe = pd.DataFrame(per_cwe_data)
df_per_cwe.to_csv(RESULTS_DIR / 'lr_per_cwe_performance.csv', index=False)
print(f"Saved per-CWE performance to {RESULTS_DIR / 'lr_per_cwe_performance.csv'}")

# Per-CWE Bar Chart
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(df_per_cwe))
width = 0.25

bars1 = ax.bar(x - width, df_per_cwe['Precision'], width, label='Precision', alpha=0.8)
bars2 = ax.bar(x, df_per_cwe['Recall'], width, label='Recall', alpha=0.8)
bars3 = ax.bar(x + width, df_per_cwe['F1-Score'], width, label='F1-Score', alpha=0.8)

ax.set_xlabel('CWE Type', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Per-CWE Performance - Logistic Regression', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df_per_cwe['CWE'], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'per_cwe_performance_lr.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'per_cwe_performance_lr.pdf', bbox_inches='tight')
print(f"Saved per-CWE chart to {FIGURES_DIR / 'per_cwe_performance_lr.png'}")
plt.close()

print("SUMMARY - LOGISTIC REGRESSION BASELINE")
print(f"\nModel: Logistic Regression (TF-IDF)")
print(f"Features: {X_train_tfidf.shape[1]:,}")
print(f"Training Time: {training_time:.2f}s")
print(f"\nTest Set Performance:")
print(f"  Accuracy:  {test_results['accuracy']:.4f}")
print(f"  Precision: {test_results['precision_macro']:.4f}")
print(f"  Recall:    {test_results['recall_macro']:.4f}")
print(f"  F1-Score:  {test_results['f1_macro']:.4f}")
print(f"  Inference: {test_results['inference_time_ms']:.4f} ms/sample")

print("PER-CWE PERFORMANCE DETAILS")
print(f"{'CWE':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
for cwe, metrics in sorted(test_results['per_class_metrics'].items()):
    print(f"CWE-{cwe:<7} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
          f"{metrics['f1']:<12.4f} {metrics['support']:<10}")