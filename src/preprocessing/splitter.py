from sklearn.model_selection import train_test_split

def split_data(students, train_percent=0.6, validate_percent=0.2, random_state=42):
    X = students.drop(['focus_index', 'burnout_level', 'productivity_score', 'exam_score'], axis=1)
    y = students[['focus_index', 'burnout_level', 'productivity_score', 'exam_score']]

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=(1-train_percent), random_state=random_state)
    X_validate, X_test, y_validate, y_test = train_test_split(X_valtest, y_valtest, test_size=(validate_percent/(1-train_percent)), random_state=random_state)
    #since train_test_split only does training and testing we do 2 splits to get out 3 sets

    return X_train, X_validate, X_test, y_train, y_validate, y_test