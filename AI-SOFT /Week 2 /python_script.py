#Load and Explore the Data
import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
data = pd.read_csv(url)

# Display the first 5 rows
print(data.head())
print("\nColumns:", data.columns)

#Filter for Recent Data (2020)
# Filter for 2020 and drop missing values
data_2020 = data[data['year'] == 2020].dropna(subset=['co2'])

# Select features and target
features = ['co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2']  # Example features
X = data_2020[features]
y = data_2020['co2']  # Total CO2 emissions

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data shape:", X_train.shape, X_test.shape)

#Train the Model Using  Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


#Visualize Results Plot Actual vs. Predicted Emissions
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emissions (2020)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()



