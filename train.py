import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
file_path = 'train_hsbc_df.csv'  # Update this with your actual file path
df = pd.read_csv(file_path)

# Convert all categorical variables to numerical using Label Encoding
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store the label encoder for inverse transforming if needed

# Define the features (X) and the target (y)
X = df.drop(columns=['fraud'])
y = df['fraud']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and target into a new DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['fraud'])], axis=1)

# Save the balanced dataset to a new CSV file
balanced_file_path = 'balanced_hsbc_df.csv'
df_resampled.to_csv(balanced_file_path, index=False)

# Split the balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train a RandomForestClassifier on the balanced dataset
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print('Classification Report:')
print(report)

# Save the trained model as a pickle file
model_file_path = 'random_forest_model.pkl'
with open(model_file_path, 'wb') as file:
    pickle.dump(rf_model, file)

# If you need to download the balanced dataset and model, they will be saved as 'balanced_hsbc_df.csv' and 'random_forest_model.pkl' in your working directory
