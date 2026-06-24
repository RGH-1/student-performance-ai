import pandas as pd

def remove_duplicates(students: pd.DataFrame):
    if(students.duplicated().any()): #checking for any duplicated rows
        mod_students = students.drop_duplicates()
        print("Duplicated Rows Removed")
        return mod_students
    else:
        print("No Duplicated Rows")
        return students


def encode_features(students: pd.DataFrame):
    # Since Academic Level and Internet Quality are on a scale, we use Label Encoding. As for Gender we use One-Hot encoding.
    students['academic_level_encoded'] = students['academic_level'].map({
        'High School':0,
        'Undergraduate':1,
        'Postgraduate':2
    })

    students['internet_quality_encoded'] = students['internet_quality'].map({
        'Good':2,
        'Average':1,
        'Poor':0
    })

    students.drop(['academic_level', 'internet_quality'], axis=1, inplace=True) #Removing the non encoded features

    students = pd.get_dummies(students, columns=['gender'], drop_first=False)

    return students