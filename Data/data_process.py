import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


data= "Data\\CTU-IoT-Malware-Capture-34-1\\bro\\conn.log.labeled"

def load_data():
    columns_data = pd.read_csv(data, sep='\t', skiprows=6, nrows=1, header=None).iloc[0][1:]
    df = pd.read_csv(data, sep='\t', comment="#", header=None)
    df.columns = columns_data
    return df

data = load_data()

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

data = process_data(data)

def data_check(data):
    #Check the target attribute
    print("Number of targets thats not Maliious/benign:",data["label"].isna().sum())
    #Plot the target attribute as a countplot as a pop up window
    sns.countplot(data["label"])
    sns.heatmap(data=data.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    numerical_features = ["duration", "orig_bytes",	"resp_bytes", "missed_bytes", "orig_pkts",	"orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    data[numerical_features].describe()

data_check(data)

def outliers(data):
    # Replace outliers using IQR (Inter-quartile Range)
    outliers_columns = ['duration']
    for col_name in outliers_columns:
        # Calculate first and third quartiles
        q1, q3 = np.nanpercentile(data[col_name],[25,75])
        
        # Calculate the inter-quartile range
        intr_qr = q3-q1

        # Calculate lower and higher bounds
        iqr_min_val = q1-(1.5*intr_qr)
        iqr_max_val = q3+(1.5*intr_qr)
        print(f"(min,max) bounds for \"{col_name}\": ({iqr_min_val},{iqr_max_val})")
        
        # Replace values that are less than min or larger then max with np.nan
        data.loc[data[col_name] < iqr_min_val, col_name] = np.nan
        data.loc[data[col_name] > iqr_max_val, col_name] = np.nan
    return data

data = outliers(data)

def encoding(data):
    #Initialize encoder with default parameters
    target_le = LabelEncoder()
    #Fit the encoder to the target attribute
    encoded_attribute = target_le.fit_transform(data["label"])
    #Replace target attribute with encoded values
    data["label"] = encoded_attribute
    #Check mapped labels
    dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))

    categorical_features = ["proto","service","conn_state","history"]
    for c in categorical_features:
        print(f"Column ({c}) has ({data[c].nunique()}) distinct values.")
    #Initialize the encoder with its default parameters
    ohe = OneHotEncoder()
    #Fit the encoder to categorical features in the dataset
    encoded_features = ohe.fit_transform(data[categorical_features])
    #Create a dataframe of encoded features
    encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=ohe.get_feature_names_out())
    #Check the results of encoding
    encoded_features_df
    data = pd.concat([data, encoded_features_df], axis=1).drop(categorical_features, axis=1)
    return data

data = encoding(data)
print(data)