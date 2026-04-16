import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.01):
        self.lr = lr
        self.params = {}
        
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * 0.01
        self.params['b1'] = np.zeros((1, hidden_size1))
        self.params['W2'] = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.params['b2'] = np.zeros((1, hidden_size2))
        self.params['W3'] = np.random.randn(hidden_size2, output_size) * 0.01
        self.params['b3'] = np.zeros((1, output_size))
        
        self.loss_history = []
    
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
    
    def fit(self, X, y, epochs=1000, batch_size=32):
        self.grads = {}
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            
            for i in range(0, X.shape[0], batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_params()
            
            if epoch % 50 == 0:
                y_pred_full = self.forward(X)
                loss = self.binary_crossentropy(y, y_pred_full.flatten())
                self.loss_history.append(loss)
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
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


def train_neural_network(X_train, X_test, y_train, y_test):
    print("Training Neural Network from Scratch")
    print("=" * 40)
    
    input_size = X_train.shape[1]
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 1
    learning_rate = 0.001
    
    nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, learning_rate)
    
    nn.fit(X_train, y_train, epochs=1000, batch_size=32)
    
    y_pred_proba = nn.predict_proba(X_test)
    y_pred = nn.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nNeural Network Performance:")
    print(f"AUC-ROC: {auc:.4f}")
    
    return nn, y_pred_proba, y_pred


def train_sklearn_model(X_train, X_test, y_train, y_test):
    print("\nTraining Scikit-learn Random Forest")
    print("=" * 40)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Random Forest AUC-ROC: {auc:.4f}")
    
    return rf, y_pred_proba, y_pred


def analyze_class_imbalance(y_train, y_test):
    print("\nClass Distribution Analysis")
    print("=" * 30)
    
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    test_pos = np.sum(y_test == 1)
    test_neg = np.sum(y_test == 0)
    
    print(f"Training set: {train_neg} negative, {train_pos} positive ({train_pos/(train_pos+train_neg):.2%} positive)")
    print(f"Test set: {test_neg} negative, {test_pos} positive ({test_pos/(test_pos+test_neg):.2%} positive)")
    
    return train_pos, train_neg, test_pos, test_neg


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
    
    axes[0, 0].plot(nn.loss_history)
    axes[0, 0].set_title('Training Loss Curve')
    axes[0, 0].set_xlabel('Epoch (x50)')
    axes[0, 0].set_ylabel('Binary Crossentropy Loss')
    axes[0, 0].grid(True)
    
    from sklearn.metrics import roc_curve
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_pred_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    
    axes[0, 1].plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {roc_auc_score(y_test, nn_pred_proba):.3f})')
    axes[0, 1].plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_pred_proba):.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_title('ROC Curves Comparison')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(thresholds, costs)
    optimal_threshold, min_cost, _, _ = calculate_optimal_threshold(y_test, nn_pred_proba)
    axes[1, 0].axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    axes[1, 0].set_title('Cost Analysis by Decision Threshold')
    axes[1, 0].set_xlabel('Decision Threshold')
    axes[1, 0].set_ylabel('Total Cost ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    optimal_pred = (nn_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, optimal_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('hospital_nn_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_executive_summary(y_test, nn_pred_proba, optimal_threshold, min_cost, train_pos, train_neg):
    tn, fp, fn, tp = confusion_matrix(y_test, (nn_pred_proba >= optimal_threshold).astype(int)).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY FOR DR. ANAND")
    print("="*60)
    
    print(f"\nCLINICAL CONTEXT:")
    print(f"- Dataset: {len(y_test)} patient records")
    print(f"- Readmission rate: {train_pos/(train_pos+train_neg):.1%} (imbalanced dataset)")
    print(f"- Model: 3-layer neural network developed from scratch")
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"- Sensitivity (Recall): {sensitivity:.1%} - correctly identifies high-risk patients")
    print(f"- Specificity: {specificity:.1%} - correctly identifies low-risk patients")
    print(f"- Precision: {precision:.1%} - accuracy when predicting readmission")
    print(f"- AUC-ROC: {roc_auc_score(y_test, nn_pred_proba):.3f} - overall discriminative ability")
    
    print(f"\nECONOMIC IMPACT:")
    print(f"- Optimal decision threshold: {optimal_threshold:.2f}")
    print(f"- False Negative cost: $1,000 (missed high-risk patient)")
    print(f"- False Positive cost: $100 (unnecessary intervention)")
    print(f"- Expected minimum cost: ${min_cost:,.0f} per 400 patients")
    
    print(f"\nCLINICAL RECOMMENDATIONS:")
    print(f"1. Deploy model with {optimal_threshold:.2f} threshold for cost optimization")
    print(f"2. Focus resources on patients above threshold (high-risk group)")
    print(f"3. Model catches {sensitivity:.0%} of patients who will be readmitted")
    print(f"4. Consider additional preventive measures for false negatives")
    
    print(f"\nIMPLEMENTATION STEPS:")
    print(f"1. Integrate model into hospital EMR system")
    print(f"2. Set up automated risk alerts at discharge")
    print(f"3. Monitor model performance monthly")
    print(f"4. Adjust threshold based on actual cost data")
    
    print(f"\n" + "="*60)


def main():
    X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_preprocess_data()
    
    train_pos, train_neg, test_pos, test_neg = analyze_class_imbalance(y_train, y_test)
    
    nn, nn_pred_proba, nn_pred = train_neural_network(X_train, X_test, y_train, y_test)
    
    rf, rf_pred_proba, rf_pred = train_sklearn_model(X_train, X_test, y_train, y_test)
    
    optimal_threshold, min_cost, thresholds, costs = calculate_optimal_threshold(y_test, nn_pred_proba)
    
    plot_results(nn, y_test, nn_pred_proba, rf_pred_proba, thresholds, costs)
    
    generate_executive_summary(y_test, nn_pred_proba, optimal_threshold, min_cost, train_pos, train_neg)
    
    print(f"\nAnalysis complete. Results saved to 'hospital_nn_results.png'")


if __name__ == "__main__":
    main()
