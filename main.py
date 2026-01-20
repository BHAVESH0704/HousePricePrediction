import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv("train.csv")
data = data[['GrLivArea','BedroomAbvGr','FullBath','OverallQual','SalePrice']].dropna()

# Inputs & Output
X = data[['GrLivArea','BedroomAbvGr','FullBath','OverallQual']]
y = data['SalePrice']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
    

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Predict new house (CORRECT way)
new_house = pd.DataFrame(
    [[1500, 3, 2, 7]],
    columns=['GrLivArea','BedroomAbvGr','FullBath','OverallQual']
)

price = model.predict(new_house)
print("Predicted Price:", price[0])

# Visualization
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.show()

joblib.dump(model, "house_price_model.pkl")
print("Model saved correctly!")


