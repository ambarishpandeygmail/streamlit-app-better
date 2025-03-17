import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# Generate anonymous real estate data
def generate_anonymous_real_estate_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'PropertySize_sqft': np.random.randint(800, 3000, num_samples),
        'PropertyAge_years': np.random.randint(1, 50, num_samples),
        'Location_index': np.random.choice([1, 2, 3, 4, 5], num_samples),
        'OccupancyRate': np.random.uniform(0.2, 1.0, num_samples),
        'HVAC_Age_years': np.random.randint(1, 20, num_samples),
        'RoofAge_years': np.random.randint(1, 30, num_samples),
        'EnergyConsumption_kWh': np.random.uniform(500, 5000, num_samples),
        'NumberOfUnits': np.random.randint(1, 10, num_samples),
    }
    df = pd.DataFrame(data)
    df['ServiceCost'] = (
        0.005 * df['PropertySize_sqft'] +
        0.003 * df['PropertyAge_years'] +
        0.01 * df['Location_index'] +
        0.008 * df['OccupancyRate'] * df['PropertySize_sqft'] +
        0.004 * df['HVAC_Age_years'] * df['PropertySize_sqft'] / 100 +
        0.002 * df['RoofAge_years'] * df['PropertySize_sqft'] / 100 +
        0.001 * df['EnergyConsumption_kWh'] +
        0.006 * df['NumberOfUnits'] * df['PropertySize_sqft'] / 100 +
        np.random.normal(0, 500, num_samples)  # Adding noise
    )
    return df

# Load and prepare data
data = generate_anonymous_real_estate_data()
X = data.drop(columns=['ServiceCost'])
y = data['ServiceCost']

# Feature Engineering
X['CostPerSqft'] = y / X['PropertySize_sqft']
X['EnergyPerSqft'] = X['EnergyConsumption_kWh'] / X['PropertySize_sqft']

# Apply Polynomial Features (interaction only)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model only once
@st.cache_data
def train_xgboost():
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, reg_alpha=0.1, reg_lambda=0.1)
    model.fit(X_train_scaled, y_train)
    return model

model = train_xgboost()

# Streamlit UI
st.title("Real Estate Service Cost Prediction App")

# Ideal values for reset (initial slider values)
ideal_values = {
    'PropertySize_sqft': 1500.0,
    'PropertyAge_years': 20.0,
    'Location_index': 3.0,
    'OccupancyRate': 0.8,
    'HVAC_Age_years': 10.0,
    'RoofAge_years': 15.0,
    'EnergyConsumption_kWh': 2500.0,
    'NumberOfUnits': 1.0
}

# Initialize session state for slider values
if 'slider_values' not in st.session_state:
    st.session_state['slider_values'] = {k: float(v) for k, v in ideal_values.items()}

# Sidebar sliders
input_features = {}
for column in X.columns:
    default_value = st.session_state['slider_values'].get(column, float(X[column].mean()))
    min_value = float(X[column].min())
    max_value = float(X[column].max())
    
    input_features[column] = st.sidebar.slider(
        column,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        key=f"slider_{column}",  # Unique key for each slider
        step=0.01
    )

# Convert input to model format
user_input = np.array([[input_features[feature] for feature in X.columns]])
user_input_poly = poly.transform(user_input)
user_input_scaled = scaler.transform(user_input_poly)

# Prediction
user_prediction = model.predict(user_input_scaled)[0]

# Display prediction
st.subheader("Predicted Service Cost:")
st.write(f"${user_prediction:,.2f}")

# Scatter Plot (Predicted vs Actual)
y_pred = model.predict(X_test_scaled)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.6, label="Predicted vs Actual")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal Fit")
ax.set_title("Predicted vs Actual Values")
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.legend()
st.pyplot(fig)

# Residuals Plot
residuals = y_test - y_pred
fig_res, ax_res = plt.subplots(figsize=(10, 6))
ax_res.hist(residuals, bins=20, color='purple', edgecolor='black')
ax_res.set_title('Residuals Distribution')
ax_res.set_xlabel('Residuals (Actual - Predicted)')
ax_res.set_ylabel('Frequency')
st.pyplot(fig_res)

# Feature Importance (All Features, Filtered)
feature_importance = model.feature_importances_
feature_names = poly.get_feature_names_out(X.columns)
sorted_idx = feature_importance.argsort()

# Filter out features with zero importance
non_zero_indices = feature_importance[sorted_idx] > 0
filtered_feature_names = np.array(feature_names)[sorted_idx][non_zero_indices]
filtered_feature_importance = feature_importance[sorted_idx][non_zero_indices]

# Create the plot for filtered features
fig_feat, ax_feat = plt.subplots(figsize=(12, len(filtered_feature_names) * 0.3))
ax_feat.barh(filtered_feature_names, filtered_feature_importance, color="blue", height=0.7)
ax_feat.set_title("Feature Importances (Filtered Zero)", fontsize=16)
ax_feat.set_xlabel("Importance Score", fontsize=12)
ax_feat.tick_params(axis='y', labelsize=8)
plt.tight_layout()
st.pyplot(fig_feat)