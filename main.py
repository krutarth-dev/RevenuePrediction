import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("/Users/apple/Documents/Projects/Revenue prediction/online_shoppers_intention.csv")

# Select features and target variables
features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
target_revenue = 'Revenue'
target_weekend = 'Weekend'
target_info_duration = 'Informational_Duration'

# Perform data preprocessing

# Encode categorical variables
label_encoder = LabelEncoder()
data['Month'] = label_encoder.fit_transform(data['Month'])
data['VisitorType'] = label_encoder.fit_transform(data['VisitorType'])

# Split the data into training and testing sets
X_train, X_test, y_revenue_train, y_revenue_test, y_weekend_train, y_weekend_test, y_info_duration_train, y_info_duration_test = \
    train_test_split(data[features], data[target_revenue], data[target_weekend], data[target_info_duration],
                     test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model for Revenue prediction
rf_classifier.fit(X_train, y_revenue_train)

# Make predictions for Revenue
revenue_predictions = rf_classifier.predict(X_test)

# Evaluate the model for Revenue prediction
revenue_classification_report = classification_report(y_revenue_test, revenue_predictions)
print("Revenue Classification Report:")
print(revenue_classification_report)

# Train the model for Weekend prediction
rf_classifier.fit(X_train, y_weekend_train)

# Make predictions for Weekend
weekend_predictions = rf_classifier.predict(X_test)

# Evaluate the model for Weekend prediction
weekend_classification_report = classification_report(y_weekend_test, weekend_predictions)
print("Weekend Classification Report:")
print(weekend_classification_report)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model for Informational_Duration prediction
rf_regressor.fit(X_train, y_info_duration_train)

# Make predictions for Informational_Duration
info_duration_predictions = rf_regressor.predict(X_test)

# Evaluate the model for Informational_Duration prediction
info_duration_mae = mean_absolute_error(y_info_duration_test, info_duration_predictions)
print("Informational_Duration MAE:", info_duration_mae)
