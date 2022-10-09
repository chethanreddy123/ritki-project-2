import streamlit as st
import pandas as pd
import os


def app2():

    st.subheader("Model Deployment")
    st.caption("Optimised solution for maintaining a balanced inventory across all distribution centers in a region ; Maximizing the benefits vs the cost to the company")
    st.cache(suppress_st_warning=True)
    def Output1():
        df = pd.read_csv('Outputs/2022-03-01/Output1_new.csv')
        
        st.write(df)

        ReID = list(df['Re-deployment ID'])
        Sk =  list(df['SKU'])

        form = st.form(key="my-form")
        c1, c2 = st.columns(2)
        with c1:
            s1 = form.selectbox("Re-deloyment ID" , ReID)
        with c2:
            tr = form.selectbox("SKU" , Sk)
        submit = form.form_submit_button("Search")

        if submit:
            with st.spinner('Generating Report....'):
                result = df[(df['Re-deployment ID'] == s1) & (df['SKU'] == tr)]
                print(s1, tr)
                st.write(result)


    st.cache(suppress_st_warning=True)


    def Output2():
        st.subheader('Benefits VS Cost(USD)')
        df = pd.read_csv('Outputs/2022-03-01/Output2_new.csv')
        st.write(df)

        Source = list(df['Source'])
        Des = list(df['Destination'])
        NSK = list(df['No of SKUs'])


        form = st.form(key="my-form1")
        c1, c2, c3 = st.columns(3)
        with c1:
            sel1 = form.selectbox("Source" , Source)
        with c2:
            track = form.selectbox("Destination", Des)
        with c3:
            track1 = form.selectbox("No of SKUs", NSK)
        submit = form.form_submit_button("Search")

        if submit:
            with st.spinner('Generating Report....'):
                result = df[(df['Source'] == sel1) & (
                    df['Destination'] == track) & (df['No of SKUs'] == track1)]
                print(sel1, track, track1)
                st.write(result)
        def InputAndFinal():
            pass
    # Outputs\2022-03-01\

    options  = st.selectbox("Choose the Ouput File" , ("Benefits Vs Cost" , "Trip Cost", "Final Stock Calculation" , "Input to Model"))
    if options == "Benefits Vs Cost":
        Output1()
    if options == "Trip Cost":
        Output2()
    if options == "Final Stock Calculation":
        df = pd.read_excel('final_stock_cal_file.xlsx')
        st.write(df)
        sku = list(df['SKU'])
        sc = list(df['Site_Code'])

        form = st.form(key="my-form1")
        c1, c2 = st.columns(2)
        with c1:
            s1 = form.selectbox("SKU" , sku)
        with c2:
            tr = form.selectbox("Site_Code" ,sc)
        submit = form.form_submit_button("Search")

        if submit:
            with st.spinner('Generating Report....'):
                result = df[(df['SKU'] == s1) & (df['Site_Code'] == tr)]
                print(s1, tr)
                st.write(result)

    if options == "Input to Model":
        df = pd.read_csv('Input to Model_new.csv')
        sku = list(df['SKU'])
        sc = list(df['Source'])

        st.write(df)

        form = st.form(key="my-form1")
        c1, c2 = st.columns(2)
        with c1:
            s1 = form.selectbox("SKU" , sku)
        with c2:
            tr = form.selectbox("Source" ,sc)
        submit = form.form_submit_button("Search")

        if submit:
            with st.spinner('Generating Report....'):
                result = df[(df['SKU'] == s1) & (df['Source'] == tr)]
                print(s1, tr)
                st.write(result)




