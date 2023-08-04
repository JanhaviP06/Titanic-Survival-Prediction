import numpy as np
#from urllib.request import urlopen
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
  
RF_model = pickle.load(open('C:/Users/janha/OneDrive/Desktop/ML/titanicmodel.pkl','rb'))
train=pd.read_csv('C:\\Users\\janha\\OneDrive\\Desktop\\ML\\Titanic Survival Prediction\\train.csv')

def titanic_prediction(data):
    input_arr = np.asarray(data)
    res = input_arr.astype(float)
    input_arr_reshaped = res.reshape(1,-1)
    op = RF_model.predict(input_arr_reshaped)
    if (op[0] == 0):
        return 'The Person not  Survived '
    else:
        return'The Person Survived '

def main():


    menu=st.sidebar.radio("Menu",['Titanic Dataset Information','Titanic Survival Prediction'])

    if menu=='Titanic Survival Prediction':
        st.title("Titanic Survival Prediction")

        image = Image.open('C:/Users/janha/OneDrive/Desktop/ML/Titanic Survival Prediction/titanic.jpg')
        st.image(image,width=600)
        st.write('')
        st.write('Enter the details below :')
        Pclass = st.text_input("Passenger Class"," Class : 1 / 2 / 3")
        Sex = st.text_input("Sex","Female : 0  Male : 1")
        Age= st.text_input("Age","Type Here")
        SibSp = st.text_input(" No. of Siblings or Spouse Aboard","Type Here")
        Parch = st.text_input("No. of Parents or Children Aboard","Type Here")
        Fare = st.text_input("Ticket Fare","Type Here")
        EmbarkedC = st.text_input("Port of Embarkation Cherbourg :","if applies 1 else 0")
        EmbarkedQ = st.text_input("Port of Embarkation Queenstown :","if applies 1 else 0")
        EmbarkedS = st.text_input("Port of Embarkation Southampton :","if applies 1 else 0")
        data=[[Pclass,Sex,Age,SibSp,Parch,Fare,EmbarkedC,EmbarkedQ,EmbarkedS]]
        
        result=""
        if st.button('Predict'):
            result = titanic_prediction(data)
            
        st.success(result)
        
    if menu=='Titanic Dataset Information':
        
        st.title("Titanic Survival Dataset :")
        st.write('')
        col1, col2= st.columns([1,3])

        with col1:
           st.subheader("Variable")
           st.write('survival')
           st.write('pclass')
           st.write('sex')
           st.write('Age')
           st.write('sibsp')
           st.write('parch')
           st.write('ticket')
           st.write('fare')
           st.write('cabin')
           st.write('embarked')
           
        with col2:
           st.subheader("Definition")
           st.write('Survival of Passenger ( 0 : No, 1 : Yes )')
           st.write('''A proxy for socio-economic status (SES)
           1st = Upper
           2nd = Middle
           3rd = Lower''')
           st.write('Sex ( 0 : Female 1 : Male)')
           st.write('Age in years')
           st.write('No of siblings / spouses aboard the Titanic')
           st.write('No of parents / children aboard the Titanic')
           st.write('Ticket number')
           st.write('Passenger fare')
           st.write('Cabin number')
           st.write('Port of Embarkation ( C = Cherbourg, Q = Queenstown, S = Southampton )')

        st.write('')
        st.subheader('Tabular Data :')
        st.table(train.head())

if __name__=='__main__':
    main()
