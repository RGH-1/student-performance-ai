from sklearn.preprocessing import MinMaxScaler, StandardScaler

#possible to add any scaler you want
def get_scaler(scaler_type):
    if(scaler_type == "min_max"):
        scaler = MinMaxScaler()
    elif(scaler_type == "standard"):
        scaler = StandardScaler()
    
    return scaler


def fit_scaler(X_train, scaler_type="min_max"):
    scaler = get_scaler(scaler_type)
    scaler = scaler.fit(X_train)
    return scaler


def apply_scaler(scaler, X):
    return scaler.transform(X)
    #this was made into a fuction in case other libraries use other methods for transforming