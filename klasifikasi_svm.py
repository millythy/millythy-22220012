import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def main():
    st.title("Brain Stroke Classification Web App")
    st.sidebar.title("Brain Stroke Classification Web App")
    st.markdown("Predict if a person is likely to have a stroke 🧠")
    st.sidebar.markdown("Predict if a person is likely to have a stroke 🧠")

    @st.cache_data(persist=True)
    def load_data():
        # Load the brain stroke dataset
        data = pd.read_csv('brain_stroke.csv')

        # Handle categorical variables encoding
        label = LabelEncoder()
        data['gender'] = label.fit_transform(data['gender'])
        data['ever_married'] = label.fit_transform(data['ever_married'])
        data['work_type'] = label.fit_transform(data['work_type'])
        data['Residence_type'] = label.fit_transform(data['Residence_type'])
        data['smoking_status'] = label.fit_transform(data['smoking_status'])

        return data

    @st.cache_data(persist=True)
    def split(df):
        # Set the target variable (stroke) and features
        y = df['stroke']
        X = df.drop(columns=['stroke'])

        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test, display_labels=['No Stroke', 'Stroke'])
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve") 
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    # Load dataset and prepare the data
    df = load_data()

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = split(df)

    # Define class names for target
    class_names = ['No Stroke', 'Stroke']

    # Sidebar for classifier selection
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # SVM Classifier
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("SVM Classifier Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))
            plot_metrics(metrics, model, x_test, y_test)

    # Logistic Regression Classifier
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))
            plot_metrics(metrics, model, x_test, y_test)

    # Random Forest Classifier
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))
            plot_metrics(metrics, model, x_test, y_test)

    # Option to display raw data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Brain Stroke Dataset")
        st.write(df)

if __name__ == '__main__':
    main()