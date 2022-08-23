import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA


#--------------------------------------------------------------------------------------------------------------------------------------------------------


start_time=time.time()  #Program Start time
#Titles
tit1,tit2 = st.columns((4, 1))
tit1.markdown("<h1 style='text-align: center;'><u>Perbandingan Akurasi Penyakit Liver</u> </h1>",unsafe_allow_html=True)
st.sidebar.title("Dataset and Classifier")

dataset_name=st.sidebar.selectbox("Select Dataset: ",('Liver',"none"))
classifier_name = st.sidebar.selectbox("Select Classifier: ",("KNN","Random Forest"))

LE=LabelEncoder()
def get_dataset(dataset_name):
    if dataset_name=="Liver":
        data=pd.read_csv("https://raw.githubusercontent.com/baharandili/Liverdisease/master/Csv/liverV1.csv")
        st.header("Liver  Dataset")
        return data


data = get_dataset(dataset_name)

def clean_dataset(dataset_name):
    assert isinstance(dataset_name, pd.DataFrame), "df needs to be a pd.DataFrame"
    dataset_name.dropna(inplace=True)
    indices_to_keep = ~dataset_name.isin([np.nan, np.inf, -np.inf]).any(1)
    return dataset_name[indices_to_keep].astype(np.float64)

def selected_dataset(dataset_name):
    if dataset_name == "Liver":
        X=data.drop(["gender","result"],axis=1)
        Y=data.result
        return X,Y  

X,Y=selected_dataset(dataset_name)

#Plot output variable
def plot_op(dataset_name):
    col1, col2 = st.columns((1, 5))
    plt.figure(figsize=(12, 3))
    plt.title("Classes in 'Not Diseased vs  Diseased' ")
    if dataset_name == "Liver":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()


st.write(data)
st.write("Shape of dataset: ",data.shape)
st.write("Number of classes: ",Y.nunique())
plot_op(dataset_name)


def add_parameter_ui(clf_name):
    params={}
    st.sidebar.write("Select values: ")


    if clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors",1,35,step=1)
        params["K"] = K

    elif clf_name == "Random Forest":
        N = st.sidebar.slider("n_estimators",5,250,step=5,value=250)
        M = st.sidebar.slider("max_depth",1,20)
        params["N"] = N
        params["M"] = M


    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    global clf

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["N"],max_depth=params["M"])

    return clf
    
clf = get_classifier(classifier_name,params)


#Build Model
def model():
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=50,test_size=0.2)

    #MinMax Scaling / Normalization of data
    Std_scaler = StandardScaler()
    X_train = Std_scaler.fit_transform(X_train)
    X_test = Std_scaler.transform(X_test)

    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)

    return Y_pred,Y_test

Y_pred,Y_test=model()

#Plot Output
def compute(Y_pred,Y_test):
    #Plot PCA
    pca=PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:,0]
    x2 = X_projected[:,1]
    plt.figure(figsize=(16,8))
    plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot()

    c1, c2 = st.columns((4,3))
  
   

    #Calculate Metrics
    acc=accuracy_score(Y_test,Y_pred)
    st.subheader("Accuracy of the model: ")
    st.text('\nAccuracy: {} %'.format(
        round((acc),2)))

st.markdown("<hr>",unsafe_allow_html=True)
st.header(f"1) Model for Prediction of {dataset_name}")
st.subheader(f"Classifier Used: {classifier_name}")
compute(Y_pred,Y_test)

#Execution Time
end_time=time.time()
st.info(f"Total execution time: {round((end_time - start_time),4)} seconds")

#@st.cache(suppress_st_warning=True)
def user_predict():
    global U_pred

    if dataset_name == "Liver":
        X = data.drop(["gender","result"], axis=1)

user_predict()  #Predict the status of user.


#-------------------------------------------------------------------------END------------------------------------------------------------------------#
