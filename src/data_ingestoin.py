import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Load the dataset
df = pd.read_csv('./data/raw/titanic.csv')

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Separate features and target variable
X = df.drop(columns=['Survived'])
y = df['Survived']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', ])
df_pca['Survived'] = y.values

# Ensure the directory exists
os.makedirs(os.path.join('data', 'processed'), exist_ok=True)

# Save the PCA results to a CSV file
df_pca.to_csv(os.path.join('data', 'processed', 'titanic_pca.csv'), index=False)
