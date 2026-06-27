from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def evaluate_poly_reg_model(X_train, y_train, X_validate, y_validate, degree):
    pipeline = Pipeline([
        ("polynomials", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_validate)
    R2_score = r2_score(y_predict, y_validate, multioutput='raw_values')

    return {
        "degree": degree,
        "model": pipeline,
        "scores": R2_score,
        "mean_score": R2_score.mean()
    }


def train_polynomial_reg(X_train, y_train, X_validate, y_validate, max_degree=1):
    best = None
    for degree in range(1, max_degree + 1):
        result = evaluate_poly_reg_model(X_train, y_train, X_validate, y_validate, degree)
        if best is None or result["mean_score"] > best["mean_score"]:
            best = result
    
    return best