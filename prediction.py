import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from neuralprophet import NeuralProphet



def app3():
    st.subheader("Out of Stock Forecasting")
    st.caption("Forecasting out of stock to improve future stock planning")

    df = pd.read_excel('Inputs\Preprocessed_files\OOS.xlsx')
    st.write(df)
    data = df.copy()

    fig = px.line(df, x='Date', y="OOS")
    st.plotly_chart(fig)

    List_of_Site_Code = list(data.Site_Code.unique())
    List_of_Material = list(data.Material_Number.unique())
    form = st.form(key="my-form")
    c1, c2 = st.columns(2)
    with c1:
        site_code = form.selectbox("Site Code", List_of_Site_Code)
    with c2:
        material_no = form.selectbox("Material Number", List_of_Material)

    check = form.form_submit_button("Generate Report")

    if check == True:
        
        new_data = data[(data['Site_Code'] == site_code) & (data['Material_Number'] == material_no)]
        new_data = new_data[['Date' , 'OOS']]
        st.write(new_data)
        train = new_data.reset_index(drop = True)
        train.rename(columns = {'Date':'ds' , "OOS" : "y"}, inplace = True)
        with st.spinner('Generating Report....'):
            m = NeuralProphet()
            m.fit(train,freq='D')
            forecast = m.predict(train)
            fig = m.plot(forecast)
            st.pyplot(fig)

            no_days = st.number_input("Enter the number of days you want to Predict for" , min_value=30)


            future=m.make_future_dataframe(train,periods=no_days)
            forecast=m.predict(future)
            st.write(forecast)
            fig = m.plot(forecast)
            st.pyplot(fig)

        
