import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


data= "Data\\CTU-IoT-Malware-Capture-34-1\\bro\\conn.log.labeled"

def load_data():
    columns_data = pd.read_csv(data, sep='\t', skiprows=6, nrows=1, header=None).iloc[0][1:]
    df = pd.read_csv(data, sep='\t', comment="#", header=None)
    df.columns = columns_data
    return df

def process_data(data):
    #Split the "tunnel_parents   label   detailed-label" column into three columns
    tunnel_parents_column = data.iloc[:,-1].apply(lambda x: x.split()[0])
    label_column = data.iloc[:,-1].apply(lambda x: x.split()[1])
    detailed_label_column = data.iloc[:,-1].apply(lambda x: x.split()[2])
    data.drop(["tunnel_parents   label   detailed-label"], axis=1, inplace=True)
    data["tunnel_parents"] = tunnel_parents_column
    data["label"] = label_column
    data["detailed_label"] = detailed_label_column
    #drop the uid column
    data.drop("uid", axis=1, inplace=True)
    #for "local_resp","local_orig","tunnel_parents" columns, convert the values to 0 and 1
    data["local_resp"] = data["local_resp"].apply(lambda x: 1 if x=="-" else 0)
    data["local_orig"] = data["local_orig"].apply(lambda x: 1 if x=="-" else 0)
    data["tunnel_parents"] = data["tunnel_parents"].apply(lambda x: 1 if x=="-" else 0)
    # IP addresses should maybe be deleted?
    data.drop(columns=["id.orig_h","id.resp_h"], inplace=True)
    # Detailed_label doesn't contribute to the data analysis, duo to it being extra information.
    data.drop(columns="detailed_label", inplace=True)
    #Remove all the rows with missing values
    data.replace({'-':np.nan, "(empty)":np.nan}, inplace=True)
    # Fix data types of the misinterpreted columns
    float_dict = {
        "duration": float,
        "orig_bytes": float,
        "resp_bytes": float
    }
    data= data.astype(float_dict)                                                  
    return data

def data_check(data):
    #Check the target attribute
    print("Number of targets thats not Maliious/benign:",data["label"].isna().sum())
    #Plot the target attribute as a countplot as a pop up window
    sns.countplot(data["label"])
    sns.heatmap(data=data.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    numerical_features = ["duration", "orig_bytes",	"resp_bytes", "missed_bytes", "orig_pkts",	"orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    data[numerical_features].describe()

def outliers(data):
    # Replace outliers using IQR (Inter-quartile Range)
    outliers_columns = ['duration']
    for col_name in outliers_columns:
        #Calculate first and third quartiles
        q1, q3 = np.nanpercentile(data[col_name],[25,75])
        #Calculate the inter-quartile range
        intr_qr = q3-q1
        #Calculate lower and higher bounds
        iqr_min_val = q1-(1.5*intr_qr)
        iqr_max_val = q3+(1.5*intr_qr)
        #Replace values that are less than min or larger then max with np.nan
        data.loc[data[col_name] < iqr_min_val, col_name] = np.nan
        data.loc[data[col_name] > iqr_max_val, col_name] = np.nan
    return data
def missing_values(data):
    #Select specific columns to be used for the classification, here we initially select the numerical attributes with no missing values.
    srv_training_columns = ["id.orig_p","id.resp_p","missed_bytes","orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes"] 
    #Split the rows into two datasets containing rows with/without "service"
    data_df_with_service = data[data["service"].notna()]
    data_df_no_service = data[data["service"].isna()]
    #Split the service dataset into dependent and independent features
    srv_X = data_df_with_service[srv_training_columns]
    srv_y = data_df_with_service["service"].values
    #Split into train/test subsets
    srv_X_train, srv_X_test, srv_y_train, srv_y_test = train_test_split(srv_X, srv_y, test_size=0.2, random_state=0)
    #Create KNN estimator and fit it
    srv_knn = KNeighborsClassifier(n_neighbors=3)
    srv_knn.fit(srv_X_train, srv_y_train)
    #Predict "service" for missing values
    srv_predictions = srv_knn.predict(data_df_no_service[srv_training_columns])
    #Update the original data set with predicted "service" values
    data.loc[data["service"].isna(), "service"] = srv_predictions
    numerical_features = data.drop("label", axis=1).select_dtypes(include="number").columns
    knn_imputer = KNNImputer()
    data_df_after_imputing = knn_imputer.fit_transform(data[numerical_features])
    data[numerical_features] = data_df_after_imputing
    return data

def encoding_labels(data):
    #Initialize encoder with default parameters
    target_le = LabelEncoder()
    #Fit the encoder to the target attribute
    encoded_attribute = target_le.fit_transform(data["label"])
    #Replace target attribute with encoded values
    data["label"] = encoded_attribute
    #Check mapped labels
    dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))
    return data

def encoding_onehot(data):
    categorical_features = ["proto","service","conn_state","history"]
    #Initialize the encoder
    ohe = OneHotEncoder()
    #Fit the encoder to categorical features in the dataset
    encoded_features = ohe.fit_transform(data[categorical_features])
    #Create a dataframe of encoded features
    encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=ohe.get_feature_names_out())
    data = pd.concat([data, encoded_features_df], axis=1).drop(categorical_features, axis=1)
    return data

def normalization(data):
    #Normalize the data
    numerical_features = ["id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes",	"resp_pkts", "resp_ip_bytes"]
    min_max_scaler = MinMaxScaler()
    data[numerical_features] = min_max_scaler.fit_transform(data[numerical_features])
    return data

def save_data(data):
    data.to_csv("Data\\processed_data.csv", index=False)

data = load_data()
data = process_data(data)
data = outliers(data)
data = missing_values(data)
data = encoding_labels(data)
data = encoding_onehot(data)
data = normalization(data)
save_data(data)

#show all of the columns
pd.set_option('display.max_columns', None)

print(data)