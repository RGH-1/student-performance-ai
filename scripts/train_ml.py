from src.preprocessing import load_data
from src.models import train_grid_search, train_polynomial_reg, save_ml_model


X_train = load_data("X_train.csv")
y_train = load_data("y_train.csv")

X_validate = load_data("X_validate.csv")
y_validate = load_data("y_validate.csv")


random_forest_model = train_grid_search(X_train, y_train, "random_forest")
gradient_boosting_model = train_grid_search(X_train, y_train, "gradient_boost")
polynomial_regression_model = train_polynomial_reg(X_train, y_train, X_validate, y_validate, max_degree=5)

save_ml_model(random_forest_model, model_name = "rf")
save_ml_model(gradient_boosting_model, model_name="gb")
save_ml_model(polynomial_regression_model, model_name="pr")