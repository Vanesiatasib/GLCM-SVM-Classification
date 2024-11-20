import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Function to compute feature importances
def compute_feature_importances(d, t):
    dataset_path = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\03_Data_Baru\\4_GLCM\\Histogram_databaru_d_{d}_t_{t}.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Select features and target
        X = df[['contrast', 'correlation', 'dissimilarity']]
        y = df['label']

        # Train Random Forest
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Compute feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Select top 3 features
        top_features = feature_importance_df['Feature'].head(3)
        return top_features.tolist(), df
    
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return None, None

# Function to train and evaluate an SVM model using top features
def train_svm(top_features, df):
    if not top_features or df is None:
        return None, None, None

    # Scale the selected top 3 features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[top_features] = scaler.fit_transform(df[top_features])

    # Split data into training, validation, and test sets (70:20:10 split)
    X = df[top_features]
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    # Grid search parameters
    param_grid = {
        'C': [100],
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'degree': [2]
    }

    # Train SVM with GridSearchCV
    svm_model = SVC()
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate the model
    val_score = grid_search.best_estimator_.score(X_val, y_val)
    test_score = grid_search.best_estimator_.score(X_test, y_test)

    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {test_score:.4f}")

    return grid_search.best_params_, val_score, test_score

# Function to save results to a CSV file
def save_results_to_csv(data):
    df_results = pd.DataFrame(data, columns=['d', 't', 'top_features', 'best_params', 'val_accuracy', 'test_accuracy'])
    df_results.to_csv('0_Histogram_Hasil.csv', index=False)
    print("Results saved to 0_Histogram_Hasil.csv")

# Main function to process datasets and train models
def main():
    results = []
    d = 15  # Example value
    t = 0   # Example value

    top_features, df = compute_feature_importances(d, t)
    if top_features:
        best_params, val_accuracy, test_accuracy = train_svm(top_features, df)
        if best_params:
            results.append([d, t, top_features, best_params, val_accuracy, test_accuracy])

    if results:
        save_results_to_csv(results)

# Run the main function
if __name__ == "__main__":
    main()
