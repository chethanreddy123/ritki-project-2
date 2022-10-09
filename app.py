import os
import streamlit as st
import numpy as np
from PIL import  Image
from pymongo.mongo_client import MongoClient
import time

# Custom imports 
from multipage import MultiPage
import  datapreprocessing, modeldep, prediction  # import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('logo.png')
display = np.array(display)
# st.image(display, width = 400)
# st.title("Data Storyteller Application")
col1, col2 = st.columns(2)
col1.image(display, width = 400)
st.title("Inventory Management")


Data = MongoClient("mongodb://InventoryManagement:Sama_12345@ac-dwqmg3b-shard-00-00.4yncqir.mongodb.net:27017,ac-dwqmg3b-shard-00-01.4yncqir.mongodb.net:27017,ac-dwqmg3b-shard-00-02.4yncqir.mongodb.net:27017/?ssl=true&replicaSet=atlas-742a1l-shard-0&authSource=admin&retryWrites=true&w=majority")
Data = Data['Test']['Test']




app.add_page("Data-Preprocessing", datapreprocessing.app1)
app.add_page("Model - Deployment", modeldep.app2)
app.add_page("Prediction", prediction.app3)


app.run()