from src.preprocessing import load_data, remove_duplicates, encode_features, split_data, fit_scaler, apply_scaler, save_data

students = load_data()

students = remove_duplicates(students)
students = encode_features(students)

X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(students)

scaler = fit_scaler(X_train)

X_train = apply_scaler(scaler, X_train)
X_validate = apply_scaler(scaler, X_validate)
X_test = apply_scaler(scaler, X_test)

save_data(X_train, "X_train.csv")
save_data(X_validate, "X_validate.csv")
save_data(X_test, "X_test.csv")
save_data(y_train, "y_train.csv")
save_data(y_validate, "y_validate.csv")
save_data(y_test, "y_test.csv")