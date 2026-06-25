from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from src.models import get_grid_config

def get_model(model_name):
    if (model_name == "random_forest"):
        return RandomForestRegressor(random_state=42)
    elif (model_name == "gradient_boost"):
        return MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42)
        )
    else:
        raise ValueError("Use a valid model name")


def train_grid_search(X_train, y_train, model_name):
    model = get_model(model_name)
    param_grid = get_grid_config(model_name)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    print("Starting fitting process")
    grid.fit(X_train, y_train)
    print("Best Parameters Found:", grid.best_params_)
    print()

    return grid.best_estimator_