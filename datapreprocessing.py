import streamlit as st
import pandas as pd
import os
import plotly.express as px


def app1():
    #st.cache(suppress_st_warning=True)
    st.subheader("Data Preprocesing and Analysis")
    st.caption("Preprocessing consists of outlier treatment; working with duplicates and null values and date-time.")
    def fileUploader():
        file = st.file_uploader("Please choose a file")

        if file is not None:
            df= pd.read_excel(file)

            st.write(df)
            return file

    #st.cache(suppress_st_warning=True)
    def RunScript(file):
        # os.system("MainModel.py") Un Comment this for new file
        pass

    # st.cache(suppress_st_warning=True)
    def InputProcessed():
        option = st.selectbox(
        'Choose the Pre-Processed File',
        ( 'Customer_order.xlsx' , 'Intransit_Volume.xlsx' , 'MACO_wo_VLC.xlsx' , 
        'OOS.xlsx' , 'Opening_Stock.xlsx' , 'planned_prod_df.xlsx' , 'Reorder_Point.xlsx' , 'Trip_Cost.xlsx' 
        , 'Average_Demand.xlsx'))

    
        if option == 'Average_Demand.xlsx':
            df= pd.read_excel(f'Inputs/Preprocessed_files/{option}')
            st.write(df)

            ListSC = list(df['Site_Code'])
    
            form = st.form(key="my-form1")
            c1, c2, c3 = st.columns(3)
            with c1:
               Month = form.selectbox("Month" ,  ("March" , "April" , "May"))
            with c2:
               Site_Code = form.selectbox("Site_Code" , list(set(ListSC)))
            with c3:
                TopNo = form.number_input("Select No Top: " , min_value= 3, max_value=10)
            submit = form.form_submit_button("Search")

            if submit:
                with st.spinner('Generating Report....'):
                    pass
            


            march = df[(df['Date'] < pd.Timestamp("2022-03-31")) & (df['Site_Code'] == Site_Code) ]
            april = df[(df['Date'] > pd.Timestamp("2022-03-31")) & (df['Date'] < pd.Timestamp("2022-04-30")) & (df['Site_Code'] == Site_Code)] 
            may = df[(df['Date'] > pd.Timestamp("2022-04-30")) & (df['Date'] < pd.Timestamp("2022-05-31") ) & (df['Site_Code'] == Site_Code)]

            if Month == 'March':
                march = march[['Material_Number' , 'Daily_Demand' , 'Site_Code']]
                st.write(march.iloc[:TopNo])
            if Month == 'April':
                april = april[['Material_Number' , 'Daily_Demand' , 'Site_Code']]
                st.write(april.iloc[:TopNo])
            if Month == 'May':
                may = may[['Material_Number' , 'Daily_Demand' , 'Site_Code']]
                st.write(may.iloc[:TopNo])
                
            st.subheader("Graphical Analysis")
            fig = px.line(df, x='Date', y="Daily_Demand")
            st.plotly_chart(fig)

        # <------------------------>

        elif option == 'Customer_order.xlsx':
            df= pd.read_excel(f'Inputs/Preprocessed_files/{option}')
            st.write(df)

            ListSC = list(df['Site_Code'])
    
            form = st.form(key="my-form2")
            c1, c2, c3 = st.columns(3)
            with c1:
               Month = form.selectbox("Month" ,  ("March" , "April" , "May"))
            with c2:
               Site_Code = form.selectbox("Site_Code" , list(set(ListSC)))
            with c3:
                TopNo = form.number_input("Select No Top: " , min_value= 3, max_value=10)
            submit = form.form_submit_button("Search")

            if submit:
                with st.spinner('Generating Report....'):
                    pass
            


            march = df[(df['Date'] < pd.Timestamp("2022-03-31")) & (df['Site_Code'] == Site_Code) ]
            april = df[(df['Date'] > pd.Timestamp("2022-03-31")) & (df['Date'] < pd.Timestamp("2022-04-30")) & (df['Site_Code'] == Site_Code)] 
            may = df[(df['Date'] >pd.Timestamp("2022-04-30")) & (df['Date'] < pd.Timestamp("2022-05-31") ) & (df['Site_Code'] == Site_Code)]

            if Month == 'March':
                march = march[['Material_Number' , 'Customer_Orders' , 'Site_Code']]
                st.write(march.iloc[:TopNo])
            if Month == 'April':
                april = april[['Material_Number' , 'Customer_Orders' , 'Site_Code']]
                st.write(april.iloc[:TopNo])
            if Month == 'May':
                may = may[['Material_Number' , 'Customer_Orders' , 'Site_Code']]
                st.write(may.iloc[:TopNo])

            st.subheader("Graphical Analysis")
            fig = px.line(df, x='Date', y="Customer_Orders")
            st.plotly_chart(fig)

        # <--------------------------------------------------->
        elif option == 'Intransit_Volume.xlsx':
            df= pd.read_excel(f'Inputs/Preprocessed_files/{option}')
            st.write(df)

            ListSC = list(df['Site_Code'])
    
            form = st.form(key="my-form3")
            c1, c2, c3 = st.columns(3)
            with c1:
               Month = form.selectbox("Month" ,  ("March" , "April" , "May"))
            with c2:
               Site_Code = form.selectbox("Site_Code" , list(set(ListSC)))
            with c3:
                TopNo = form.number_input("Select No Top: " , min_value= 3, max_value=10)
            submit = form.form_submit_button("Search")

            if submit:
                with st.spinner('Generating Report....'):
                    pass 
            


            march = df[(df['Date'] < pd.Timestamp("2022-03-31")) & (df['Site_Code'] == Site_Code) ]
            april = df[(df['Date'] > pd.Timestamp("2022-03-31")) & (df['Date'] <  pd.Timestamp("2022-04-30")) & (df['Site_Code'] == Site_Code)] 
            may = df[(df['Date'] > pd.Timestamp("2022-04-30")) & (df['Date'] < pd.Timestamp("2022-05-31") ) & (df['Site_Code'] == Site_Code)]

            if Month == 'March':
                march = march[['Material_Number' , 'In_Transit' , 'Site_Code']]
                st.write(march.iloc[:TopNo])
            if Month == 'April':
                april = april[['Material_Number' , 'In_Transit' , 'Site_Code']]
                st.write(april.iloc[:TopNo])
            if Month == 'May':
                may = may[['Material_Number' , 'In_Transit' , 'Site_Code']]
                st.write(may.iloc[:TopNo])

            st.subheader("Graphical Analysis")
            fig = px.line(df, x='Date', y="In_Transit")
            st.plotly_chart(fig)

            

        # <----------------------------------->

        elif option == 'Opening_Stock.xlsx':
            df= pd.read_excel(f'Inputs/Preprocessed_files/{option}')
            st.write(df)

            ListSC = list(df['Site_Code'])
    
            form = st.form(key="my-form3")
            c1, c2, c3 = st.columns(3)
            with c1:
               Month = form.selectbox("Month" ,  ("March" , "April" , "May"))
            with c2:
               Site_Code = form.selectbox("Site_Code" , list(set(ListSC)))
            with c3:
                TopNo = form.number_input("Select No Top: " , min_value= 3, max_value=10)
            submit = form.form_submit_button("Search")

            if submit:
                with st.spinner('Generating Report....'):
                    pass
            


            march = df[(df['Date'] < pd.Timestamp("2022-03-31")) & (df['Site_Code'] == Site_Code) ]
            april = df[(df['Date'] > pd.Timestamp("2022-03-31")) & (df['Date'] < pd.Timestamp("2022-04-30")) & (df['Site_Code'] == Site_Code)] 
            may = df[(df['Date'] > pd.Timestamp("2022-04-30")) & (df['Date'] < pd.Timestamp("2022-05-31")) & (df['Site_Code'] == Site_Code)]

            if Month == 'March':
                march = march[['Material_Number' , 'Opening_Stock' , 'Site_Code']]
                st.write(march.iloc[:TopNo])
            if Month == 'April':
                april = april[['Material_Number' , 'Opening_Stock' , 'Site_Code']]
                st.write(april.iloc[:TopNo])
            if Month == 'May':
                may = may[['Material_Number' , 'Opening_Stock' , 'Site_Code']]
                st.write(may.iloc[:TopNo])

            st.subheader("Graphical Analysis")
            fig = px.line(df, x='Date', y="Opening_Stock")
            st.plotly_chart(fig)


        #>------------------------------------>
        elif option == 'OOS.xlsx':
            df= pd.read_excel(f'Inputs/Preprocessed_files/{option}')
            st.write(df)

            ListSC = list(df['Site_Code'])
    
            form = st.form(key="my-form3")
            c1, c2, c3 = st.columns(3)
            with c1:
               Month = form.selectbox("Month" ,  ("March" , "April" , "May"))
            with c2:
               Site_Code = form.selectbox("Site_Code" , list(set(ListSC)))
            with c3:
                TopNo = form.number_input("Select No Top: " , min_value= 3, max_value=10)
            submit = form.form_submit_button("Search")

            if submit:
                with st.spinner('Generating Report....'):
                    pass
            


            march = df[(df['Date'] < pd.Timestamp("2022-03-31")) & (df['Site_Code'] == Site_Code) ]
            april = df[(df['Date'] > pd.Timestamp("2022-03-31")) & (df['Date'] < pd.Timestamp("2022-04-30")) & (df['Site_Code'] == Site_Code)] 
            may = df[(df['Date'] > pd.Timestamp("2022-04-30")) & (df['Date'] < pd.Timestamp("2022-05-31") ) & (df['Site_Code'] == Site_Code)]

            if Month == 'March':
                march = march[['Material_Number' , 'OOS' , 'Site_Code']]
                st.write(march.iloc[:TopNo])
            if Month == 'April':
                april = april[['Material_Number' , 'OOS' , 'Site_Code']]
                st.write(april.iloc[:TopNo])
            if Month == 'May':
                may = may[['Material_Number' , 'OOS' , 'Site_Code']]
                st.write(may.iloc[:TopNo])

            st.subheader("Graphical Analysis")
            fig = px.line(df, x='Date', y="OOS")
            st.plotly_chart(fig)

        else:
            df= pd.read_excel(f'Inputs/Preprocessed_files/{option}')
            st.write(df)



        
    file = fileUploader()
    if file:
        # RunScript(file) # un comment this for new excel
        InputProcessed()



        