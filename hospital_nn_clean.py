import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.001):
        self.lr = lr
        self.params = {}
        
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.params['b1'] = np.zeros((1, hidden_size1))
        self.params['W2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        self.params['b2'] = np.zeros((1, hidden_size2))
        self.params['W3'] = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        self.params['b3'] = np.zeros((1, output_size))
        
        self.loss_history = []
        self.val_loss_history = []
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def binary_crossentropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def forward(self, X):
        self.cache = {}
        
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = self.relu(self.cache['Z1'])
        
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = self.relu(self.cache['Z2'])
        
        self.cache['Z3'] = np.dot(self.cache['A2'], self.params['W3']) + self.params['b3']
        self.cache['A3'] = self.sigmoid(self.cache['Z3'])
        
        return self.cache['A3']
    
    def backward(self, X, y):
        m = X.shape[0]
        
        dZ3 = self.cache['A3'] - y.reshape(-1, 1)
        self.grads['dW3'] = (1/m) * np.dot(self.cache['A2'].T, dZ3)
        self.grads['db3'] = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, self.params['W3'].T)
        dZ2 = dA2 * self.relu_derivative(self.cache['Z2'])
        self.grads['dW2'] = (1/m) * np.dot(self.cache['A1'].T, dZ2)
        self.grads['db2'] = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.params['W2'].T)
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        self.grads['dW1'] = (1/m) * np.dot(X.T, dZ1)
        self.grads['db1'] = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    def update_params(self):
        self.params['W1'] -= self.lr * self.grads['dW1']
        self.params['b1'] -= self.lr * self.grads['db1']
        self.params['W2'] -= self.lr * self.grads['dW2']
        self.params['b2'] -= self.lr * self.grads['db2']
        self.params['W3'] -= self.lr * self.grads['dW3']
        self.params['b3'] -= self.lr * self.grads['db3']
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=1000, batch_size=32):
        self.grads = {}
        best_val_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            
            for i in range(0, X.shape[0], batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_params()
            
            if epoch % 10 == 0:
                y_pred_full = self.forward(X)
                train_loss = self.binary_crossentropy(y, y_pred_full.flatten())
                self.loss_history.append(train_loss)
                
                if X_val is not None:
                    y_pred_val = self.forward(X_val)
                    val_loss = self.binary_crossentropy(y_val, y_pred_val.flatten())
                    self.val_loss_history.append(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}", end="")
                    if X_val is not None:
                        print(f", Val Loss: {val_loss:.4f}")
                    else:
                        print()
    
    def predict_proba(self, X):
        return self.forward(X).flatten()
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def load_and_preprocess_data():
    df = pd.read_csv('hospital_records_cleaned.csv')
    
    df['age_squared'] = df['age'] ** 2
    df['bp_ratio'] = df['diastolic_bp'] / df['systolic_bp']
    df['med_per_diagnosis'] = df['num_medications'] / (df['num_diagnoses'] + 1)
    df['stay_med_ratio'] = df['length_of_stay_days'] / (df['num_medications'] + 1)
    
    feature_cols = ['age', 'age_squared', 'length_of_stay_days', 'systolic_bp', 'diastolic_bp', 
                   'glucose_mg_dl', 'creatinine_mg_dl', 'bmi', 'num_medications', 
                   'num_diagnoses', 'icu_stay', 'bp_ratio', 'med_per_diagnosis', 'stay_med_ratio']
    
    df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'].fillna('Unknown'))
    df['department_encoded'] = LabelEncoder().fit_transform(df['department'].fillna('Unknown'))
    df['insurance_encoded'] = LabelEncoder().fit_transform(df['insurance_type'].fillna('Unknown'))
    
    feature_cols.extend(['gender_encoded', 'department_encoded', 'insurance_encoded'])
    
    X = df[feature_cols].copy()
    y = df['readmitted_30d'].values
    
    X = X.fillna(X.median())
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, feature_cols


def train_neural_network(X_train, X_val, X_test, y_train, y_val, y_test):
    print("Training Neural Network")
    print("=" * 40)
    
    input_size = X_train.shape[1]
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 1
    learning_rate = 0.0005
    
    nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, learning_rate)
    nn.fit(X_train, y_train, X_val, y_val, epochs=2000, batch_size=16)
    
    y_pred_proba = nn.predict_proba(X_test)
    y_pred = nn.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nNeural Network Performance:")
    print(f"AUC-ROC: {auc:.4f}")
    
    return nn, y_pred_proba, y_pred


def train_sklearn_benchmark(X_train, X_test, y_train, y_test):
    print("\nTraining Scikit-learn Random Forest")
    print("=" * 40)
    
    rf = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5
    )
    rf.fit(X_train, y_train)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Random Forest AUC-ROC: {auc:.4f}")
    
    return rf, y_pred_proba, y_pred


def calculate_optimal_threshold(y_true, y_pred_proba, fn_cost=1000, fp_cost=100):
    thresholds = np.arange(0.01, 0.99, 0.01)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        costs.append(total_cost)
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    
    return optimal_threshold, min_cost, thresholds, costs


def plot_results(nn, y_test, nn_pred_proba, rf_pred_proba, thresholds, costs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(nn.loss_history, label='Training Loss', linewidth=2)
    axes[0, 0].plot(nn.val_loss_history, label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch (x10)')
    axes[0, 0].set_ylabel('Binary Crossentropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    from sklearn.metrics import roc_curve
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_pred_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    
    axes[0, 1].plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {roc_auc_score(y_test, nn_pred_proba):.3f})', linewidth=2)
    axes[0, 1].plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_pred_proba):.3f})', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    axes[0, 1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(thresholds, costs, linewidth=2)
    axes[1, 0].axvline(thresholds[np.argmin(costs)], color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold = {thresholds[np.argmin(costs)]:.2f}')
    axes[1, 0].set_title('Cost Analysis by Decision Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Decision Threshold')
    axes[1, 0].set_ylabel('Total Cost ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    optimal_pred = (nn_pred_proba >= thresholds[np.argmin(costs)]).astype(int)
    cm = confusion_matrix(y_test, optimal_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], 
                xticklabels=['No Readmission', 'Readmission'],
                yticklabels=['No Readmission', 'Readmission'])
    axes[1, 1].set_title(f'Confusion Matrix (Threshold = {thresholds[np.argmin(costs)]:.2f})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('hospital_nn_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = load_and_preprocess_data()
    
    print(f"Training set: {X_train.shape[0]} patients")
    print(f"Validation set: {X_val.shape[0]} patients")
    print(f"Test set: {X_test.shape[0]} patients")
    print(f"Features: {len(feature_cols)}")
    print(f"Readmission rate - Training: {np.mean(y_train):.2%}")
    print(f"Readmission rate - Test: {np.mean(y_test):.2%}")
    
    nn, nn_pred_proba, nn_pred = train_neural_network(X_train, X_val, X_test, y_train, y_val, y_test)
    rf, rf_pred_proba, rf_pred = train_sklearn_benchmark(X_train, X_test, y_train, y_test)
    
    optimal_threshold, min_cost, thresholds, costs = calculate_optimal_threshold(y_test, nn_pred_proba)
    
    print(f"Optimal Decision Threshold: {optimal_threshold:.2f}")
    print(f"Minimum Expected Cost: ${min_cost:,.0f} per {len(y_test)} patients")
    print(f"Cost Savings vs Treating All: ${(len(y_test) * 100) - min_cost:,.0f}")
    
    plot_results(nn, y_test, nn_pred_proba, rf_pred_proba, thresholds, costs)
    
    optimal_pred = (nn_pred_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, optimal_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print("\n" + "="*60)
    print("CLINICAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\nNEURAL NETWORK METRICS (at optimal threshold {optimal_threshold:.2f}):")
    print(f"- Sensitivity (Recall): {sensitivity:.1%}")
    print(f"- Specificity: {specificity:.1%}")
    print(f"- Precision: {precision:.1%}")
    print(f"- AUC-ROC: {roc_auc_score(y_test, nn_pred_proba):.3f}")
    
    print(f"\nECONOMIC IMPACT:")
    print(f"- Total Expected Cost: ${min_cost:,}")
    print(f"- Cost per Patient: ${min_cost/len(y_test):.2f}")
    
    print(f"\nAnalysis complete. Results saved to 'hospital_nn_results.png'")


if __name__ == "__main__":
    main()
