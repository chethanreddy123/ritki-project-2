import inspect
import glob
import numpy as np
import pandas as pd
import random, os
import time
import datetime as dt
from datetime import datetime, timedelta
# import lightgbm as lgb

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
import warnings
warnings.filterwarnings('ignore')
import calendar
from pathlib import Path
from itertools import combinations
import math
from math import sin, cos, sqrt, atan2, radians
import copy
from sklearn import preprocessing
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

TimeStart = time.time()


#Input paths 
root_path = Path(os.getcwd())
select_country = 'Dominican_Republic'

input_path = root_path / "Inputs"
preprocessed_path = input_path/"Preprocessed_files" 

if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
    print("created")
else:
    print("exists")

#Updated currency conversion rate
usd_to_pesos = 57.40

################################
direc= ""
load_path= Path(direc)

planned_prod_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
planned_procurement_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
maco_wo_vlc_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
avg_demand_forecast_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
customer_orders_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
oos_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
reorder_point_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
dc_mapping_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
uom_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
avg_trip_cost_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
opening_stock_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
intransit_volume_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'
planned_in_filename = load_path/'09.09 - Data Requirement Dominican Republic - MKTP.xlsx'



#UOM Sheet
uom_df=pd.read_excel(uom_filename,sheet_name='UOM')
print(uom_df.shape)
print(uom_df.SKU.nunique())
uom_df.columns= ['Material_Number','SKU Name', 'CS_PAL','SU_CS','SU_PAL','KG_SU','HL_SU','CS_HL','CS_KG','HLKG_SU','HL_PAL']
uom_df = uom_df.drop_duplicates(subset='Material_Number')
uom_df['CS_KG']= uom_df['CS_KG'].replace(0.0,np.NaN).fillna(uom_df['CS_KG'].mean())
uom_df['KG_CS'] = 1/uom_df['CS_KG']
uom_df['HL_CS'] = uom_df['CS_HL']
uom_df['KG_PAL'] = uom_df['CS_PAL']/uom_df['CS_KG']
cols_list = ['Material_Number','SKU Name','HL_PAL','KG_CS','HL_CS','CS_PAL','KG_PAL','HL_SU','SU_CS']
uom_df = uom_df[cols_list]

uom_df['HL_SU'] = uom_df['HL_CS']/(uom_df['SU_CS'])
uom_df.drop(columns={'SU_CS'},inplace=True)
############################################################

cols_list = ['HL_PAL','KG_CS','HL_CS','CS_PAL','KG_PAL','HL_SU']
for cols in cols_list:
    uom_df[cols] = uom_df[cols].replace(0.0,np.NaN).fillna(uom_df[cols].mean())
    uom_df[cols] = 1/uom_df[cols]
    
uom_df = uom_df.melt(id_vars = ['Material_Number','SKU Name'])
uom_df.rename(columns={'variable':'Convert Unit','value':'Conversion Factor'},inplace=True)

uom_df[['Base Unit','Convert Unit']] = pd.DataFrame(uom_df['Convert Unit'].str.split(pat='_', n=-1, expand=False).to_list(), 
                                                    columns=['Base Unit','Convert Unit'])

uom_df['Material_Number'] = uom_df['Material_Number'].str.upper()
sku_list = list(uom_df['Material_Number'].unique())
print("Number of unique SKUs :",uom_df['Material_Number'].nunique())

uom_df.head()



#DC Info
add_data_df=pd.read_excel(dc_mapping_filename,sheet_name='Aditional Data Required')
print(add_data_df.shape)
dc_mapping = add_data_df.loc[6:18,['Active SKUs','Unnamed: 9','Unnamed: 10']].reset_index(drop= True)
dc_mapping.columns= ['Depot Code','Depot Description','DESCRIPTION']
dc_mapping['DC_FLAG']= np.where(dc_mapping.DESCRIPTION.isin(['DDC','DDC/HUB']),1,0)
dc_mapping['HUB_FLAG']= np.where(dc_mapping.DESCRIPTION.isin(['HUB','DDC/HUB']),1,0)

#Filtering for only DCs
dc_mapping= dc_mapping[dc_mapping['DC_FLAG']==1]
dc_mapping['Depot Code'] = dc_mapping['Depot Code'].str.upper()

depot_code_dict=dict(list(zip(dc_mapping['Depot Code'],dc_mapping['Depot Description']))) 
print("Number of unique Sites :",dc_mapping['Depot Code'].nunique())

depot_code_list = []
for col in depot_code_dict.keys():
    depot_code_list.append(col)

dc_mapping



#Trip Cost
trip_cost_df=pd.read_excel(avg_trip_cost_filename,sheet_name='Trip Cost')
trip_cost_df.columns= ['ORIGIN','DESTINATION', 'PAL_TRUCK','DR_CURRENCY','USD','ROUND_TRIP_KMS','USD_KM','USD_PAL']
trip_cost_df= trip_cost_df.loc[3:,].reset_index(drop= True)

#Filtering for Haina & Santiago Orgins
trip_cost_df= trip_cost_df[trip_cost_df['ORIGIN'].isin(['Haina','Santiago'])]
trip_cost_df['DESTINATION']= trip_cost_df['DESTINATION'].apply(lambda x: x.split('/'))
trip_cost_df= trip_cost_df.explode(['DESTINATION']).reset_index(drop=True)[['ORIGIN','DESTINATION','PAL_TRUCK',
                                                                            'ROUND_TRIP_KMS','USD']]

#One way distance 
trip_cost_df['DISTANCE_KMS']= trip_cost_df['ROUND_TRIP_KMS']/2
trip_cost_df.drop(['PAL_TRUCK','ROUND_TRIP_KMS'], inplace= True, axis=1)
print(trip_cost_df.shape)
trip_cost_df= trip_cost_df.drop_duplicates(['ORIGIN','DESTINATION'], keep='last').reset_index(drop=True)
print(trip_cost_df.shape)
trip_cost_df.columns= ['ORIGIN','DESTINATION','DESTINATION_USD','DESTINATION_DISTANCE_KMS']

#Merging again on origin to connect source & destination depot - Triangulation Distance
trip_cost_df2= trip_cost_df.copy()
trip_cost_df2.columns= ['ORIGIN','SOURCE','SOURCE_USD','SOURCE_DISTANCE_KMS']
print(trip_cost_df.shape, trip_cost_df2.shape)
trip_cost_df_final= trip_cost_df.merge(trip_cost_df2, on=['ORIGIN'], how='outer')
print(trip_cost_df_final.shape)
trip_cost_df_final['SAME_FLAG']= np.where(trip_cost_df_final['SOURCE']==trip_cost_df_final['DESTINATION'],1,0)
trip_cost_df_final= trip_cost_df_final[trip_cost_df_final['SAME_FLAG']!=1].drop(['SAME_FLAG'], axis=1).reset_index(drop=True)
trip_cost_df_final['TRIP_DISTANCE']= trip_cost_df_final['SOURCE_DISTANCE_KMS']+trip_cost_df_final['DESTINATION_DISTANCE_KMS']
trip_cost_df_final['TRIP_COST']= trip_cost_df_final['SOURCE_USD']+trip_cost_df_final['DESTINATION_USD']
trip_cost_df_final= trip_cost_df_final[['SOURCE','ORIGIN','DESTINATION','TRIP_DISTANCE','TRIP_COST']]

#Renaming DCs
rename_dc_list= {'CND CD San Pedro de Macorís':'CND CD San Pedro de Macoris',
'CND CD Bávaro':'CND CD Bavaro',
'CND CD San Fco. de Macorís':'CND CD San Fco. de Macoris',
'CND CD Samaná':'CND CD Samana'}

trip_cost_df_final =trip_cost_df_final.replace({"SOURCE": rename_dc_list})
trip_cost_df_final= trip_cost_df_final.replace({"DESTINATION": rename_dc_list})

#Getting DCs codes
print(trip_cost_df_final.shape)
trip_cost_df_final = trip_cost_df_final.rename(columns={'SOURCE':'Depot Description'}).merge(
    dc_mapping[['Depot Description','Depot Code']],
    on='Depot Description',how='left').rename(columns={'Depot Code':
                                                        'Site_Code_Source code'}).drop(columns={'Depot Description'})


trip_cost_df_final = trip_cost_df_final.rename(columns={'DESTINATION':'Depot Description'}).merge(
    dc_mapping[['Depot Description','Depot Code']],
    on='Depot Description',how='left').rename(columns={'Depot Code':
                                                       'Site_Code_Destination code'}).drop(columns={'Depot Description'})

trip_cost_df_final = trip_cost_df_final.rename(columns={'Site_Code_Source code':'Site_Code_Source','Site_Code_Destination code':'Site_Code_Destination'})
trip_cost_df_final['Site_Code_Source'] = trip_cost_df_final['Site_Code_Source'].str.upper()
trip_cost_df_final['Site_Code_Destination'] = trip_cost_df_final['Site_Code_Destination'].str.upper()
trip_cost_df_final = trip_cost_df_final[trip_cost_df_final['Site_Code_Source'].isin(depot_code_list)].reset_index(drop=True)
trip_cost_df_final = trip_cost_df_final[trip_cost_df_final['Site_Code_Destination'].isin(depot_code_list)].reset_index(drop=True)
print("Number of unique Source Sites :",trip_cost_df_final['Site_Code_Source'].nunique())
print("Number of unique Destination Sites :",trip_cost_df_final['Site_Code_Destination'].nunique())

print(trip_cost_df_final.shape)
trip_cost_df= trip_cost_df_final.copy()
trip_cost_df['Trip Cost per km']= trip_cost_df['TRIP_COST']/trip_cost_df['TRIP_DISTANCE']
trip_cost_df= trip_cost_df.rename(columns={'TRIP_COST':'Trip Cost','TRIP_DISTANCE':'trip_distance'})
trip_cost_df= trip_cost_df[['Site_Code_Source','ORIGIN','Site_Code_Destination','trip_distance','Trip Cost','Trip Cost per km']]
print(trip_cost_df.shape)
trip_cost_df.head()


trip_cost_df['Trip Cost']= trip_cost_df['Trip Cost'].astype(np.float64)
trip_cost_df['Trip Cost']= trip_cost_df['Trip Cost']/trip_cost_df['Trip Cost'].mean()




#Opening Stock
opening_stock_df=pd.read_excel(opening_stock_filename,sheet_name='Daily Opening Stock March-May',skiprows=1)
print(opening_stock_df.shape)

opening_stock_df = opening_stock_df.rename(columns={'SKU':'Material_Number', 
                                                    "DDC":'Site_Code'}).reset_index(drop=True)

opening_stock_df= opening_stock_df[['Site_Code','Material_Number','Description','Date','Stock Available']].reset_index(drop=True)
# print(opening_stock_df.Date.unique())
opening_stock_df['Site_Code'] = opening_stock_df['Site_Code'].str.upper()
opening_stock_df['Material_Number'] = opening_stock_df['Material_Number'].str.upper()
print('DCs: ',opening_stock_df['Site_Code'].nunique())
print('SKUs: ',opening_stock_df['Material_Number'].nunique())
opening_stock_df = opening_stock_df[opening_stock_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
opening_stock_df = opening_stock_df[opening_stock_df['Material_Number'].isin(sku_list)].reset_index(drop=True)
print("Number of unique Sites :",opening_stock_df['Site_Code'].nunique())
print("Number of unique SKUs :",opening_stock_df['Material_Number'].nunique())
print(opening_stock_df['Date'].min(), opening_stock_df['Date'].max())

opening_stock_df.head() # pointers Excel



#Intransit Volume
intransit_volume_columns = ['Planned Date','DDC','Destination Site','SKU','Quantity (Sales Unit)']
intransit_volume_df=pd.read_excel(intransit_volume_filename,sheet_name='In-Transit Volume')

intransit_volume_df=intransit_volume_df[['Planned Date','Origin Site','Destination Site','SKU','Quantity (Sales Unit)']]
intransit_volume_df.columns=['Date','Origin Site','Destination Site','Material_Number','Quantity (Sales Unit)']
print(intransit_volume_df['Destination Site'].nunique())


intransit_depot_dict= {'Azua':'CND CD Azua',
                      'San Juan':'CND CD San Juan',
                      'San Pedro':'CND CD San Pedro de Macoris',
                      'Bohemia':'CND CD Bohemia',
                      'Santiago':'CND CD Santiago',
                      'Mao':'CND CD Mao',
                      'Puerto Plata':'CND CD Puerto Plata',
                      'Samana':'CND CD Samana',
                      'Bavaro':'CND CD Bavaro',
                      'San Martin':'CND CD Central',
                      'San Isidro':'CND CD Margen Oriental',
                      'San Francisco':'CND CD San Fco. de Macoris'}

intransit_volume_df =intransit_volume_df.replace({"Destination Site": intransit_depot_dict})

intransit_volume_df = intransit_volume_df.rename(columns={'Destination Site':'Depot Description'}).merge(
    dc_mapping[['Depot Description','Depot Code']],
    on='Depot Description',how='left').rename(columns={'Depot Code':
                                                        'Site_Code'})#.drop(columns={'Depot Description'})

print(intransit_volume_df.shape)
intransit_volume_df.isnull().sum()
intransit_volume_df['Date'] = pd.to_datetime(intransit_volume_df['Date'], format='%Y-%m-%d')
intransit_volume_df = intransit_volume_df[~intransit_volume_df['Quantity (Sales Unit)'].isna()]
intransit_volume_df = intransit_volume_df[~intransit_volume_df['Material_Number'].isna()]
print(intransit_volume_df.shape)

##########Combining Outflow from DCs#############
intransit_volume_df_return = intransit_volume_df[intransit_volume_df['Site_Code'].isna()][['Date',
                                                                                           'Origin Site','Depot Description',
                                                                                           'Material_Number',
                                                                                           'Quantity (Sales Unit)']]
intransit_volume_df_return.columns= ['Date','Depot Description','Origin Site','Material_Number','Quantity (Sales Unit)']
intransit_volume_df_return['Quantity (Sales Unit)']= -intransit_volume_df_return['Quantity (Sales Unit)']

intransit_depot_dict= {'Azua':'CND CD Azua',
                      'San Juan':'CND CD San Juan',
                      'San Pedro':'CND CD San Pedro de Macoris',
                      'Bohemia':'CND CD Bohemia',
                      'Santiago':'CND CD Santiago',
                      'Mao':'CND CD Mao',
                      'Puerto Plata':'CND CD Puerto Plata',
                      'Samana':'CND CD Samana',
                      'Bavaro':'CND CD Bavaro',
                      'San Martin':'CND CD Central',
                      'San Isidro':'CND CD Margen Oriental',
                      'San Francisco':'CND CD San Fco. de Macoris'}

intransit_volume_df_return =intransit_volume_df_return.replace({"Depot Description": intransit_depot_dict})

intransit_volume_df_return = intransit_volume_df_return.merge(
    dc_mapping[['Depot Description','Depot Code']],
    on='Depot Description',how='left').rename(columns={'Depot Code':
                                                        'Site_Code'})
intransit_volume_df_return= intransit_volume_df_return[~intransit_volume_df_return['Site_Code'].isna()]
intransit_volume_df_return= intransit_volume_df_return[['Date','Origin Site','Depot Description','Material_Number',
                                                       'Quantity (Sales Unit)','Site_Code']]
print(intransit_volume_df_return.shape)
#########################################

intransit_volume_df = intransit_volume_df[~intransit_volume_df['Site_Code'].isna()]
print(intransit_volume_df.shape)

#Appending in transit & outlfow from DCs
intransit_volume_df= intransit_volume_df.append(intransit_volume_df_return)
print(intransit_volume_df.shape)

intransit_volume_df['Site_Code'] = intransit_volume_df['Site_Code'].str.upper()
intransit_volume_df['Material_Number'] = intransit_volume_df['Material_Number'].str.upper()
print('DCs: ',intransit_volume_df['Site_Code'].nunique())
print('SKUs: ',intransit_volume_df['Material_Number'].nunique())
intransit_volume_df = intransit_volume_df[intransit_volume_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
intransit_volume_df = intransit_volume_df[intransit_volume_df['Material_Number'].isin(sku_list)].reset_index(drop=True)

#Aggregating in transit data
intransit_volume_df = intransit_volume_df.rename(columns={'Quantity (Sales Unit)':
                                                         'In_Transit'}).groupby(['Date',
                                                                                 'Site_Code',
                                                                                 'Material_Number'])[['In_Transit']].sum().reset_index()

print('Date available from {} to {}'.format(intransit_volume_df['Date'].min(),intransit_volume_df['Date'].max()))
print("Number of unique Sites :",intransit_volume_df['Site_Code'].nunique())
print("Number of unique SKUs :",intransit_volume_df['Material_Number'].nunique())

intransit_volume_df.head() # Pointer Excel


#Average Demand Forecast
avg_demand_df = pd.read_excel(avg_demand_forecast_filename, sheet_name="Avg Demand Forecast (7 days)")
avg_demand_df= avg_demand_df.loc[1:,]
avg_demand_df.columns= ['DDC','SKU','Description','Category','M03','M04','M05']
avg_demand_df = avg_demand_df[['DDC','SKU','M03','M04','M05']]
print(avg_demand_df.shape)
avg_demand_df= avg_demand_df.drop_duplicates()
print(avg_demand_df.shape)
avg_demand_df = avg_demand_df.melt(id_vars = ['DDC','SKU'])
print(avg_demand_df.shape)

avg_demand_df['value']= avg_demand_df['value'].astype(np.float64)
######################Deriving avg daily demand from avg monthly demand-> Assumption 25 working days######################
avg_demand_df['value']= avg_demand_df['value']/25
############################################################################################################################

avg_demand_df = avg_demand_df.rename(columns={'DDC':'Site_Description', "SKU":'Material_Number','variable':'Date','value':'Daily_Demand'})

month_dict= {'M03':'March','M04':'April','M05':'May'}
avg_demand_df= avg_demand_df.replace({'Date':month_dict})

avg_demand_df = avg_demand_df.merge(
    dc_mapping[['Depot Code','Depot Description']].rename(columns={'Depot Description':'Site_Description'})
    ,on='Site_Description',how='left').drop(columns={'Site_Description'}).rename(columns={'Depot Code':'Site_Code'})
print(avg_demand_df.shape)

avg_demand_df['Site_Code'] = avg_demand_df['Site_Code'].str.upper()
avg_demand_df['Material_Number'] = avg_demand_df['Material_Number'].str.upper()

print("Sites :",avg_demand_df['Site_Code'].nunique())
print("SKUs :",avg_demand_df['Material_Number'].nunique())

avg_demand_df = avg_demand_df[avg_demand_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
avg_demand_df = avg_demand_df[avg_demand_df['Material_Number'].isin(sku_list)].reset_index(drop=True)
print("Number of unique Sites :",avg_demand_df['Site_Code'].nunique())
print("Number of unique SKUs :",avg_demand_df['Material_Number'].nunique())

######################## Capping negative daily demand to 0 ##########################
avg_demand_df['Daily_Demand']= np.where(avg_demand_df['Daily_Demand']<0,0,avg_demand_df['Daily_Demand'])
###############################################################################################
 
avg_demand_df.head() ## pointer excel

avg_demand_df.head()
print(avg_demand_df.shape)
avg_demand_df= avg_demand_df.drop_duplicates(subset=['Date','Material_Number','Site_Code'])
print(avg_demand_df.shape)
avg_demand_df.head() ## pointer excel


#Getting list of dates
intransit_volume_df['Date'].min()
intransit_volume_df['Date'].max()

from datetime import timedelta, date

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

start_dt = date(2022, 3, 1)
end_dt = date(2022, 5, 31)

dates_list=[]

for dt in daterange(start_dt, end_dt):
    dates_list.append(dt.strftime("%Y-%m-%d"))
    
date_df= pd.DataFrame({'DATE':dates_list})
date_df['KEY']=1

date_df['DATE']= pd.to_datetime(date_df['DATE'], format='%Y-%m-%d')
date_df['MONTH']= date_df['DATE'].dt.month

date_df


print(date_df.shape)
date_df= date_df.drop_duplicates(subset=['DATE'])
print(date_df.shape)
date_df.head()


############# Merging avg demand forecast with dates to get daily demand for all dates ##############
avg_demand_df['KEY']=1
print(avg_demand_df.shape, date_df.shape)
month_dict= {'March':3,'April':4,'May':5}
avg_demand_df['MONTH']= avg_demand_df['Date'].map(month_dict)
avg_demand_df.head() # pointer excel

avg_demand_df= avg_demand_df.drop(columns={'Date'}).merge(date_df, on=['KEY','MONTH'], 
                                                          how='left').drop(columns={'KEY'}).rename(columns={'DATE':'Date'})
print(avg_demand_df.shape)
avg_demand_df= avg_demand_df[['Date','Material_Number','Daily_Demand','Site_Code']]
avg_demand_df.head() # pointer excel

print(avg_demand_df.shape)
avg_demand_df= avg_demand_df.drop_duplicates(subset=['Date','Material_Number','Site_Code'])
print(avg_demand_df.shape)
avg_demand_df.head() # pointer excel


#Re-order point
reorder_point_df = pd.read_excel(reorder_point_filename, sheet_name="Reorder Point DOC")
reorder_point_df= reorder_point_df[['DDC','SKU','Reorder']]

reorder_point_df = reorder_point_df.merge(
    dc_mapping[['Depot Code','Depot Description']].rename(columns={'Depot Description':'DDC'})
    ,on='DDC',how='left').drop(columns={'DDC'}).rename(columns={'Depot Code':'Site_Code'})

reorder_point_df.rename(columns={'SKU':'Material_Number', 'Reorder':'Reorder_Point_SU'}, inplace=True)
reorder_point_df['Site_Code'] = reorder_point_df['Site_Code'].str.upper()
reorder_point_df['Material_Number'] = reorder_point_df['Material_Number'].str.upper()

print("Sites :",reorder_point_df['Site_Code'].nunique())
print("SKUs :",reorder_point_df['Material_Number'].nunique())

reorder_point_df = reorder_point_df[reorder_point_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
reorder_point_df = reorder_point_df[reorder_point_df['Material_Number'].isin(sku_list)].reset_index(drop=True)
reorder_point_df = reorder_point_df[~reorder_point_df['Site_Code'].isna()]
reorder_point_df = reorder_point_df[~reorder_point_df['Material_Number'].isna()]

print("Number of unique Sites :",reorder_point_df['Site_Code'].nunique())
print("Number of unique SKUs :",reorder_point_df['Material_Number'].nunique())
print(reorder_point_df.shape)

reorder_point_df= reorder_point_df[reorder_point_df.Reorder_Point_SU!=0]
print('Checking duplicates', reorder_point_df[reorder_point_df.duplicated(['Site_Code','Material_Number'])].shape)
print('Number of DC SKU combinations with 0 reorder point', reorder_point_df[reorder_point_df.Reorder_Point_SU==0].shape)
print(reorder_point_df.shape)

reorder_point_df.head() # pointer excel


#######################Outlier Treatment for Reorder point####################
##################### Might get changed later on ###########################

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

reorder_point_df_IQR= reorder_point_df.groupby(['Material_Number'], as_index=False).agg({'Reorder_Point_SU':[q25,q75]})
reorder_point_df_IQR.columns=['Material_Number','Reorder_Point_25','Reorder_Point_75']
reorder_point_df_IQR['IQR']= reorder_point_df_IQR['Reorder_Point_75']- reorder_point_df_IQR['Reorder_Point_25']
reorder_point_df_IQR['LOWER_WHISKER']= reorder_point_df_IQR['Reorder_Point_25']-1.5*reorder_point_df_IQR['IQR']
reorder_point_df_IQR['UPPER_WHISKER']= reorder_point_df_IQR['Reorder_Point_75']+1.5*reorder_point_df_IQR['IQR']

############# Negative Reorder points means-> excess as per inventory policy,hence no reorder for SKU ##############
reorder_point_df_IQR['LOWER_WHISKER']= np.where(reorder_point_df_IQR['LOWER_WHISKER']<0,0,
                                                reorder_point_df_IQR['LOWER_WHISKER'])
#####################################################################################################################

print(reorder_point_df_IQR.shape)
reorder_point_df_IQR= reorder_point_df_IQR[['Material_Number','LOWER_WHISKER','UPPER_WHISKER']]
print(reorder_point_df.shape, reorder_point_df_IQR.shape)

reorder_point_df= reorder_point_df.merge(reorder_point_df_IQR, on=['Material_Number'], how='left')
reorder_point_df['Reorder_Point_SU']= np.where(reorder_point_df['Reorder_Point_SU']<reorder_point_df['LOWER_WHISKER'],
                                               reorder_point_df['LOWER_WHISKER'],
                                               np.where(reorder_point_df['Reorder_Point_SU']>reorder_point_df['UPPER_WHISKER'],
                                                       reorder_point_df['UPPER_WHISKER'],
                                                       reorder_point_df['Reorder_Point_SU']))

reorder_point_df=reorder_point_df[['Material_Number','Reorder_Point_SU','Site_Code']]
print(reorder_point_df.shape)
# reorder_point_df_IQR
reorder_point_df.head() # pointer excel


#MACO Mirror SKUs 
maco_wo_mirror = pd.read_excel(maco_wo_vlc_filename, sheet_name="Mirror SKUs", skiprows=1)
print(maco_wo_mirror.shape)
maco_wo_mirror.head() # pointer excel


#MACO wo VLC
maco_wo_vlc_df = pd.read_excel(maco_wo_vlc_filename, sheet_name="MACO wo VLC")
maco_wo_vlc_df= maco_wo_vlc_df[['Unnamed: 1','See comment','Unnamed: 8','Unnamed: 10','Unnamed: 12','Unnamed: 13']]
maco_wo_vlc_df.columns= ['Month','CATEGORY','SKU','VOL_HL','MACO_LC','MACO_HL']
maco_wo_vlc_df= maco_wo_vlc_df.loc[1:]
print(maco_wo_vlc_df.shape)
print(maco_wo_vlc_df.CATEGORY.unique())

maco_wo_vlc_df['VOL_HL']= maco_wo_vlc_df['VOL_HL'].astype(np.float64)
maco_wo_vlc_df['MACO_LC']= maco_wo_vlc_df['MACO_LC'].astype(np.float64)
maco_wo_vlc_df['MACO_HL']= maco_wo_vlc_df['MACO_HL'].astype(np.float64)

print(maco_wo_vlc_df.SKU.nunique())

#Filtering for positive volume
maco_wo_vlc_df= maco_wo_vlc_df[maco_wo_vlc_df['VOL_HL']>0]
print(maco_wo_vlc_df.shape)
print(maco_wo_vlc_df.CATEGORY.nunique())

maco_wo_vlc_df['MACO_LC']= np.round(maco_wo_vlc_df['MACO_LC'],1)
maco_wo_vlc_df['MACO_HL']= np.round(maco_wo_vlc_df['MACO_HL'],1)

maco_wo_vlc_df.head() # pointer excel

#Merging Mirror SKUs for SKU Dia B column
maco_wo_mirror2_sku1= maco_wo_mirror[['SKU Dia B','SKU']].drop_duplicates()
maco_wo_mirror2_sku1= maco_wo_mirror2_sku1[maco_wo_mirror2_sku1['SKU Dia B'].isnull()==False].reset_index(drop=True)
print(maco_wo_mirror2_sku1.shape)
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df= maco_wo_vlc_df.merge(maco_wo_mirror2_sku1.rename(columns={'SKU':'MAIN_SKU_1',
                                                                         'SKU Dia B':'SKU'}), on=['SKU'], how='left')
print(maco_wo_vlc_df.shape)


#Merging Mirror SKUs for SKU Club B column
maco_wo_mirror2_sku2= maco_wo_mirror[['SKU Club B','SKU']].drop_duplicates()
maco_wo_mirror2_sku2= maco_wo_mirror2_sku2[maco_wo_mirror2_sku2['SKU Club B'].isnull()==False].reset_index(drop=True)
print(maco_wo_mirror2_sku2.shape)
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df= maco_wo_vlc_df.merge(maco_wo_mirror2_sku2.rename(columns={'SKU':'MAIN_SKU_2',
                                                                         'SKU Club B':'SKU'}), on=['SKU'], how='left')
print(maco_wo_vlc_df.shape)

#Merging Mirror SKUs for SKU Promo column
maco_wo_mirror2_sku3= maco_wo_mirror[['SKU Promo','SKU']].drop_duplicates()
maco_wo_mirror2_sku3= maco_wo_mirror2_sku3[maco_wo_mirror2_sku3['SKU Promo'].isnull()==False].reset_index(drop=True)
print(maco_wo_mirror2_sku3.shape)
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df= maco_wo_vlc_df.merge(maco_wo_mirror2_sku3.rename(columns={'SKU':'MAIN_SKU_3',
                                                                         'SKU Promo':'SKU'}), on=['SKU'], how='left')
print(maco_wo_vlc_df.shape)


maco_wo_vlc_df.head() # pointer excel


print(maco_wo_mirror2_sku1.shape)
print(maco_wo_vlc_df[maco_wo_vlc_df['MAIN_SKU_1'].isnull()==False]['SKU'].nunique())

print(maco_wo_mirror2_sku2.shape)
print(maco_wo_vlc_df[maco_wo_vlc_df['MAIN_SKU_2'].isnull()==False]['SKU'].nunique())

print(maco_wo_mirror2_sku3.shape)
print(maco_wo_vlc_df[maco_wo_vlc_df['MAIN_SKU_3'].isnull()==False]['SKU'].nunique())

print(maco_wo_vlc_df.SKU.nunique())
maco_wo_vlc_df['REVISED_SKU']= np.where(maco_wo_vlc_df['MAIN_SKU_1'].isnull()==False, maco_wo_vlc_df['MAIN_SKU_1'],
                                       np.where(maco_wo_vlc_df['MAIN_SKU_2'].isnull()==False, maco_wo_vlc_df['MAIN_SKU_2'],
                                               np.where(maco_wo_vlc_df['MAIN_SKU_3'].isnull()==False, maco_wo_vlc_df['MAIN_SKU_3'],
                                                       maco_wo_vlc_df['SKU'])))
print(maco_wo_vlc_df.REVISED_SKU.nunique())

maco_wo_vlc_df= maco_wo_vlc_df[['Month','CATEGORY','REVISED_SKU','VOL_HL','MACO_LC','MACO_HL']].rename(columns={'REVISED_SKU':
                                                                                                               'SKU'})
print(maco_wo_vlc_df.shape)
print(maco_wo_vlc_df[maco_wo_vlc_df['VOL_HL']==0].shape)
maco_wo_vlc_df= maco_wo_vlc_df.groupby(['Month','CATEGORY','SKU'], as_index=False).agg({'VOL_HL':'sum',
                                                                                       'MACO_LC':'sum',
                                                                                       'MACO_HL':'mean'})
print(maco_wo_vlc_df[maco_wo_vlc_df['VOL_HL']==0].shape)
print(maco_wo_vlc_df.shape)
print(maco_wo_vlc_df.SKU.nunique())
maco_wo_vlc_df.head() # pointer excel


#Deriving MACO/HL 
maco_wo_vlc_df['DERIVED_MACO_HL']= maco_wo_vlc_df['MACO_LC']/maco_wo_vlc_df['VOL_HL']
maco_wo_vlc_df['DERIVED_MACO_HL']= np.round(maco_wo_vlc_df['DERIVED_MACO_HL'],1)
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df= maco_wo_vlc_df[['Month','CATEGORY','SKU','VOL_HL',
                                'MACO_LC','DERIVED_MACO_HL']].rename(columns={'DERIVED_MACO_HL':'MACO_HL'})
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df.head() # pointer excel


#######################Outlier Treatment for MACO ####################
##################### Might get changed later on- Specially higher quantile & negative values doubt ###########################
print(maco_wo_vlc_df.shape)
print(maco_wo_vlc_df.CATEGORY.nunique())

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)


maco_wo_vlc_df_IQR= maco_wo_vlc_df.groupby(['CATEGORY'], as_index= False).agg({'MACO_HL':[q25,q75]})
maco_wo_vlc_df_IQR.columns=['CATEGORY','MACO_HL_25','MACO_HL_75']
maco_wo_vlc_df_IQR['IQR']= maco_wo_vlc_df_IQR['MACO_HL_75']- maco_wo_vlc_df_IQR['MACO_HL_25']
maco_wo_vlc_df_IQR['LOWER_WHISKER']= maco_wo_vlc_df_IQR['MACO_HL_25']-1.5*maco_wo_vlc_df_IQR['IQR']
maco_wo_vlc_df_IQR['UPPER_WHISKER']= maco_wo_vlc_df_IQR['MACO_HL_75']+1.5*maco_wo_vlc_df_IQR['IQR']

# maco_wo_vlc_df_IQR

print(maco_wo_vlc_df_IQR.shape)
maco_wo_vlc_df_IQR= maco_wo_vlc_df_IQR[['CATEGORY','LOWER_WHISKER','UPPER_WHISKER']]
print(maco_wo_vlc_df.shape, maco_wo_vlc_df_IQR.shape)

maco_wo_vlc_df= maco_wo_vlc_df.merge(maco_wo_vlc_df_IQR, on=['CATEGORY'], how='left')
maco_wo_vlc_df['MACO_HL']= np.where(maco_wo_vlc_df['MACO_HL']<maco_wo_vlc_df['LOWER_WHISKER'],
                                               maco_wo_vlc_df['LOWER_WHISKER'],
                                               np.where(maco_wo_vlc_df['MACO_HL']>maco_wo_vlc_df['UPPER_WHISKER'],
                                                       maco_wo_vlc_df['UPPER_WHISKER'],
                                                       maco_wo_vlc_df['MACO_HL']))

maco_wo_vlc_df=maco_wo_vlc_df[['Month','SKU','VOL_HL','MACO_LC','MACO_HL']]
print(maco_wo_vlc_df.shape)

maco_wo_vlc_df.head() # pointer excel


maco_wo_vlc_df = maco_wo_vlc_df.rename(columns={'SKU':'Material_Number','MACO_HL':'MACOwoVLC/HL'})
maco_wo_vlc_df['MACOwoVLC/HL'] = maco_wo_vlc_df['MACOwoVLC/HL']*(1/usd_to_pesos)
maco_wo_vlc_df['Material_Number'] = maco_wo_vlc_df['Material_Number'].str.upper()

print("SKUs :",maco_wo_vlc_df['Material_Number'].nunique())
maco_wo_vlc_df = maco_wo_vlc_df[maco_wo_vlc_df['Material_Number'].isin(sku_list)].reset_index(drop=True)
print("Number of unique SKUs :",maco_wo_vlc_df['Material_Number'].nunique())
maco_wo_vlc_df.isnull().sum()

maco_wo_vlc_df.head() # pointer excel



print(maco_wo_vlc_df.shape)
maco_wo_vlc_df = maco_wo_vlc_df.groupby(['Material_Number'])[['MACOwoVLC/HL']].mean().reset_index()
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df.head() # pointer excel


#Customer Orders 
customer_orders_df = pd.read_excel(customer_orders_filename, sheet_name="Customer Orders ")
customer_orders_df.rename(columns={'DDC':'Site_Description', 
                                   'SKU':'Material_Number', 'Qty ':'Customer_Orders',
                                  'Dispatch Date':'Date'},inplace=True)

customer_orders_df['Date']= pd.to_datetime(customer_orders_df['Date'], format='%Y-%m-%d')
customer_orders_df= customer_orders_df[['Site_Description','Material_Number','Date','Customer_Orders']]

customer_orders_df = customer_orders_df.merge(
    dc_mapping[['Depot Code','Depot Description']].rename(columns={'Depot Description':'Site_Description'}),
    on='Site_Description',how='left').drop(columns={'Site_Description'}).rename(
    columns={'Depot Code':'Site_Code'})

customer_orders_df['Site_Code'] = customer_orders_df['Site_Code'].str.upper()
customer_orders_df['Material_Number'] = customer_orders_df['Material_Number'].str.upper()

print("Sites :",customer_orders_df['Site_Code'].nunique())
print("SKUs :",customer_orders_df['Material_Number'].nunique())

customer_orders_df = customer_orders_df[customer_orders_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
customer_orders_df = customer_orders_df[customer_orders_df['Material_Number'].isin(sku_list)].reset_index(drop=True)

print(customer_orders_df.shape)
# customer_orders_df= customer_orders_df.drop_duplicates()
# print(customer_orders_df.shape)
customer_orders_df = customer_orders_df.groupby(['Date','Site_Code','Material_Number'])[['Customer_Orders']].sum().reset_index()
print(customer_orders_df.shape)

print('Date available from {} to {}'.format(customer_orders_df['Date'].min(),customer_orders_df['Date'].max()))
print("Number of unique Sites :",customer_orders_df['Site_Code'].nunique())
print("Number of unique SKUs :",customer_orders_df['Material_Number'].nunique())

customer_orders_df.head() # excel pointer


############# Dummy Data ##################
print(customer_orders_df.shape)
print(customer_orders_df.Date.min(), customer_orders_df.Date.max())
customer_orders_df= customer_orders_df[(customer_orders_df['Date']>='2022-03-01')&
                                         (customer_orders_df['Date']<='2022-05-31')]
print(customer_orders_df.shape)
print(customer_orders_df.Date.min(), customer_orders_df.Date.max())
####################################################
customer_orders_df.head() # excel pointer


customer_orders_df_dc = customer_orders_df.groupby(['Site_Code']).agg({'Customer_Orders':'sum'}).sort_values(by='Customer_Orders',ascending=False).reset_index()
customer_orders_df_dc # excel pointer

customer_orders_df_sku = customer_orders_df.groupby(['Material_Number']).agg({'Customer_Orders':'sum'}).sort_values(by='Customer_Orders',ascending=False).reset_index()
customer_orders_df_sku


#OOS
oos_df = pd.read_excel(oos_filename, sheet_name="Out of Stock",skiprows=1)
oos_df= oos_df[['DDC','Delivery Date','Order ID','SKU','OOS']]

oos_df = oos_df.merge(dc_mapping[['Depot Code',
                                  'Depot Description']].rename(columns={'Depot Description':'DDC'}),
                      on='DDC',how='left').drop(columns={'DDC'}).rename(
    columns={'Depot Code':'Site_Code'})

oos_df.rename(columns={'SKU':'Material_Number', 'Delivery Date':'Date'},inplace=True)

oos_df['Site_Code'] = oos_df['Site_Code'].str.upper()
oos_df['Material_Number'] = oos_df['Material_Number'].str.upper()
oos_df.head() # pointer excel


print("Sites :",oos_df['Site_Code'].nunique())
print("SKUs :",oos_df['Material_Number'].nunique())
print(oos_df.OOS.sum())

oos_df = oos_df[oos_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
oos_df = oos_df[oos_df['Material_Number'].isin(sku_list)].reset_index(drop=True)

print('Date available from {} to {}'.format(oos_df['Date'].min(),oos_df['Date'].max()))
print("Number of unique Sites :",oos_df['Site_Code'].nunique())
print("Number of unique SKUs :",oos_df['Material_Number'].nunique())
print(oos_df.shape)
print(oos_df.OOS.sum())

oos_df.head() # pointer excel


oos_df = oos_df.drop_duplicates(subset=['Date','Order ID','Site_Code', 'Material_Number']).reset_index(drop=True)
print(oos_df.shape)
print(oos_df.OOS.sum())


#Filtering only positive OOS
print(oos_df.shape)
oos_df= oos_df[oos_df['OOS']>=0]
print(oos_df.shape)

oos_df = oos_df.groupby(['Date','Site_Code','Material_Number'])['OOS'].sum().reset_index()
print(oos_df.shape)
print(oos_df.OOS.sum())

oos_df.head()


oos_df_sku = oos_df.groupby(['Material_Number']).agg({'OOS':'sum'}).sort_values(by='OOS',ascending=False).reset_index()
print(oos_df_sku.shape)
oos_df_sku = oos_df_sku.merge(customer_orders_df_sku,on='Material_Number',how='left').sort_values(by='Customer_Orders',ascending=False).reset_index(drop=True)
print(oos_df_sku.shape)
oos_df_sku['PERCENT']=np.round(oos_df_sku['OOS']*100/oos_df_sku['Customer_Orders'],2)
oos_df_sku= oos_df_sku.sort_values(['PERCENT'], ascending= False)
oos_df_sku.head(20)


oos_df_dc = oos_df.groupby(['Site_Code']).agg({'OOS':'sum'}).sort_values(by='OOS',ascending=False).reset_index()
oos_df_dc = oos_df_dc.merge(customer_orders_df_dc,on='Site_Code',how='left').sort_values(by='Customer_Orders',ascending=False).reset_index(drop=True)
# oos_df_dc
oos_df_dc['PERCENT']=np.round(oos_df_dc['OOS']*100/oos_df_dc['Customer_Orders'],2)
oos_df_dc= oos_df_dc.sort_values(['PERCENT'], ascending= False)

# oos_df_dc['ratio']=oos_df_dc['OOS']/oos_df_dc['Customer_Orders']
oos_df_dc


#Planned Procurement
planned_prod_df = pd.read_excel(planned_prod_filename, sheet_name="Planned Procurement March-Aug",skiprows=1)
planned_prod_df= planned_prod_df[['DDC','SKU','Received Date','Qty']].rename(columns={'DDC':'Site_Code','Received Date':'PRODUCTIONDATE','SKU':'Material_Number','Qty':'Production'})
planned_prod_df['Material_Number'] = planned_prod_df['Material_Number'].str.upper()
planned_prod_df['Site_Code'] = planned_prod_df['Site_Code'].str.upper()
planned_prod_df['PRODUCTIONDATE'] = pd.to_datetime(planned_prod_df['PRODUCTIONDATE'])
planned_prod_df = planned_prod_df[planned_prod_df['Material_Number'].isin(sku_list)].reset_index(drop=True)
planned_prod_df = planned_prod_df[planned_prod_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
planned_prod_df['Production'] = planned_prod_df['Production']/2
print('Date available from {} to {}'.format(planned_prod_df['PRODUCTIONDATE'].min(),planned_prod_df['PRODUCTIONDATE'].max()))
print("Number of unique Sites :",planned_prod_df['Site_Code'].nunique())
print("Number of unique SKUs :",planned_prod_df['Material_Number'].nunique())
planned_prod_df.head()


#Planned Procurement
planned_in_df = pd.read_excel(planned_in_filename, sheet_name="Planned Procurement March-Aug",skiprows=1)
planned_in_df= planned_in_df[['DDC','SKU','Received Date','Qty']].rename(columns={'DDC':'Site_Code','Received Date':'Date','SKU':'Material_Number','Qty':'Planned_In'})
planned_in_df['Material_Number'] = planned_in_df['Material_Number'].str.upper()
planned_in_df['Site_Code'] = planned_in_df['Site_Code'].str.upper()
planned_in_df['Date'] = pd.to_datetime(planned_in_df['Date'])
planned_in_df = planned_in_df[planned_in_df['Material_Number'].isin(sku_list)].reset_index(drop=True)
planned_in_df = planned_in_df[planned_in_df['Site_Code'].isin(depot_code_list)].reset_index(drop=True)
planned_in_df['Planned_In']=planned_in_df['Planned_In']/2
print('Date available from {} to {}'.format(planned_in_df['Date'].min(),planned_in_df['Date'].max()))
print("Number of unique Sites :",planned_in_df['Site_Code'].nunique())
print("Number of unique SKUs :",planned_in_df['Material_Number'].nunique())
planned_in_df.head()


print(opening_stock_df[opening_stock_df.duplicated(['Date','Site_Code','Material_Number'])].shape)
opening_stock_df[opening_stock_df.duplicated(['Date','Site_Code','Material_Number'])].head()


############## MIGHT CHANGE- MANIPULATION- Dummy Data ####################
# print(opening_stock_df['Date'].unique())
# opening_stock_df= opening_stock_df[opening_stock_df['Date']=='03/2022']
# opening_stock_df['Date_mod'] = pd.to_datetime("2022" + "-" + "03" + "-" + "08")
print(opening_stock_df.shape)
print(opening_stock_df.Material_Number.nunique())
opening_stock_df= opening_stock_df.drop_duplicates(['Date','Site_Code','Material_Number','Stock Available'])
print(opening_stock_df.shape)

# opening_stock_df= opening_stock_df.drop(columns=['Date','Description']).rename(columns={'Date_mod':'Date'})
opening_stock_df= opening_stock_df[['Date','Site_Code','Material_Number',
                                    'Stock Available']].rename(columns={'Stock Available':
                                                                        'Opening_Stock'})

#Aggregating same level opening stock
opening_stock_df= opening_stock_df.groupby(['Date','Site_Code','Material_Number'], as_index=False).agg({'Opening_Stock':
                                                                                                       'sum'})
print(opening_stock_df.shape)
############################################################

opening_stock_df.head()

customer_orders_df = customer_orders_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
customer_orders_df['Customer_Orders'] = customer_orders_df['Customer_Orders']/customer_orders_df['Conversion Factor']
customer_orders_df.drop(columns={'Conversion Factor'},inplace=True)
customer_orders_df.head()


oos_df = oos_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
oos_df['OOS'] = oos_df['OOS']/oos_df['Conversion Factor']
oos_df.drop(columns={'Conversion Factor'},inplace=True)
oos_df.head()


planned_in_df = planned_in_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
planned_in_df['Planned_In'] = planned_in_df['Planned_In']/planned_in_df['Conversion Factor']
planned_in_df.drop(columns={'Conversion Factor'},inplace=True)
planned_in_df.head()

intransit_volume_df = intransit_volume_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
intransit_volume_df['In_Transit'] = intransit_volume_df['In_Transit']/intransit_volume_df['Conversion Factor']
intransit_volume_df.drop(columns={'Conversion Factor'},inplace=True)
intransit_volume_df.head()


opening_stock_df = opening_stock_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
opening_stock_df['Opening_Stock'] = opening_stock_df['Opening_Stock']/opening_stock_df['Conversion Factor']
opening_stock_df.drop(columns={'Conversion Factor'},inplace=True)
opening_stock_df.head()

avg_demand_df_new = avg_demand_df.copy()
avg_demand_df_new['Daily_Demand'] = avg_demand_df_new['Daily_Demand']*15
avg_demand_df_new.head()
reorder_point_df_new = avg_demand_df.groupby(['Material_Number','Site_Code'])['Daily_Demand'].mean().reset_index().rename(columns={'Daily_Demand':'Reorder_Point_Hl'})
reorder_point_df_new.head()



planned_prod_df = planned_prod_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
planned_prod_df['Production'] = planned_prod_df['Production']/planned_prod_df['Conversion Factor']
planned_prod_df.drop(columns={'Conversion Factor'},inplace=True)
planned_prod_df.head()



reorder_point_df = reorder_point_df.merge(uom_df.loc[(uom_df['Convert Unit']=='SU') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
reorder_point_df['Reorder_Point_SU'] = reorder_point_df['Reorder_Point_SU']/reorder_point_df['Conversion Factor']
reorder_point_df.drop(columns={'Conversion Factor'},inplace=True)
reorder_point_df.rename(columns={'Reorder_Point_SU':'Reorder_Point_Hl'},inplace=True)
reorder_point_df
# reorder_point_df_new = pd.concat([reorder_point_df[['Material_Number','Reorder_Point_Hl','Site_Code']],reorder_point_df_new[['Material_Number','Reorder_Point_Hl','Site_Code']]],axis=0).drop_duplicates(subset=['Material_Number','Site_Code'],keep='first')
reorder_point_df_new = pd.concat([reorder_point_df_new[['Material_Number','Reorder_Point_Hl','Site_Code']],reorder_point_df[['Material_Number','Reorder_Point_Hl','Site_Code']]],axis=0).drop_duplicates(subset=['Material_Number','Site_Code'],keep='first')
print(reorder_point_df_new.Reorder_Point_Hl.median())
print(reorder_point_df_new.describe([0,0.1,0.2,0.25,0.3,0.4,0.5,
                                     0.6,0.7,0.75,0.8,0.9,0.99,1]).apply(lambda s: s.apply('{0:.3f}'.format)))



#Reorder point median
reorder_point_df_new.loc[reorder_point_df_new.Reorder_Point_Hl==0,'Reorder_Point_Hl']= reorder_point_df_new.Reorder_Point_Hl.median()


maco_wo_vlc_df.to_excel(preprocessed_path/"MACO_wo_VLC.xlsx", index=False)
customer_orders_df.to_excel(preprocessed_path/"Customer_order.xlsx", index=False)
oos_df.to_excel(preprocessed_path/"OOS.xlsx", index=False)
planned_in_df.to_excel(preprocessed_path/"Planned_In.xlsx", index=False)
avg_demand_df.to_excel(preprocessed_path/"Average_Demand.xlsx", index=False)
intransit_volume_df.to_excel(preprocessed_path/"Intransit_Volume.xlsx", index=False)
opening_stock_df.to_excel(preprocessed_path/"Opening_Stock.xlsx", index=False)
trip_cost_df.to_excel(preprocessed_path/"Trip_Cost.xlsx", index=False)
reorder_point_df_new.to_excel(preprocessed_path/"Reorder_Point.xlsx", index=False)
dc_mapping.to_excel(preprocessed_path/"DC_mapping.xlsx", index=False)
uom_df.to_excel(preprocessed_path/"UOM_Conversion.xlsx", index=False)
planned_prod_df.to_excel(preprocessed_path/"planned_prod_df.xlsx", index=False)


##### MACO file changes #####
maco_wo_vlc_df = maco_wo_vlc_df.loc[maco_wo_vlc_df['MACOwoVLC/HL'] > 0, :].reset_index(drop=True)
maco_wo_vlc_df['Material_Number']=maco_wo_vlc_df['Material_Number'].astype(str)
maco_wo_vlc_df = maco_wo_vlc_df.groupby(['Material_Number'])[['MACOwoVLC/HL']].mean().reset_index()
# maco_wo_vlc_df = outliter_treatment(maco_wo_vlc_df)
all_skus=maco_wo_vlc_df[maco_wo_vlc_df['MACOwoVLC/HL']>0]
median_maco=all_skus['MACOwoVLC/HL'].median()
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df = uom_df[['Material_Number']].drop_duplicates().merge(maco_wo_vlc_df,on='Material_Number',how='left')
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df['MACOwoVLC/HL'] = maco_wo_vlc_df['MACOwoVLC/HL'].fillna(median_maco)


final_stock_cal_file = pd.read_excel('final_stock_cal_file.xlsx')
print(final_stock_cal_file.shape)
final_stock_cal_file.head()

final_stock_cal_file = final_stock_cal_file.drop_duplicates(subset=['Site_Code', 'Site Name', 'SKU'],keep='first')

min_date = pd.to_datetime('2022-03-01')
recommendation_week = list(pd.date_range(min_date,end=min_date+timedelta(days=6),freq='D'))

oos_df_sku_dc_new = oos_df[oos_df['Date'].isin(recommendation_week)].merge(maco_wo_vlc_df,on='Material_Number',how='left').groupby(['Material_Number','Site_Code']).agg({'OOS':'sum','MACOwoVLC/HL':'mean'}).sort_values(by='OOS',ascending=False).reset_index()

final_stock_cal_file = final_stock_cal_file.rename(columns={'SKU':'Material_Number'}).merge(oos_df_sku_dc_new,on=['Material_Number','Site_Code'],how='left')
final_stock_cal_file['MACOwoVLC/HL'].fillna(final_stock_cal_file['MACOwoVLC/HL'].mean(),inplace=True)
# final_stock_cal_file = final_stock_cal_file[~((final_stock_cal_file['Volume required to reach Nat Avg']==0)&(final_stock_cal_file['OOS']==0)&(final_stock_cal_file['Vol available for Redeployment wrt ROP']==0))]
final_stock_cal_file = final_stock_cal_file.fillna(0)

final_stock_cal_file.head(5)

final_stock_cal_file_oos = final_stock_cal_file.copy()
final_stock_cal_file_oos = final_stock_cal_file_oos.groupby(['Material_Number']).agg({'Volume required to reach Nat Avg':'sum','OOS':'sum','MACOwoVLC/HL':'mean'}).reset_index()



final_stock_cal_file_obs = final_stock_cal_file.copy()
final_stock_cal_file_obs = final_stock_cal_file_obs.groupby(['Material_Number']).agg({'Vol available for Redeployment wrt ROP':'sum','MACOwoVLC/HL':'mean'}).reset_index()


final_stock_cal_file_calc = final_stock_cal_file_oos.merge(final_stock_cal_file_obs.drop(columns={'MACOwoVLC/HL'}),on='Material_Number',how='left')
final_stock_cal_file_calc = final_stock_cal_file_calc.fillna(0)

final_stock_cal_file_calc['OOS_min'] = np.where(final_stock_cal_file_calc['OOS']<final_stock_cal_file_calc['Vol available for Redeployment wrt ROP'],final_stock_cal_file_calc['OOS'],final_stock_cal_file_calc['Vol available for Redeployment wrt ROP'])
# final_stock_cal_file_calc['Obs_min'] = np.where(final_stock_cal_file_calc['Vol']<final_stock_cal_file_calc['Volume required to reach Nat Avg'],final_stock_cal_file_calc['Vol'],final_stock_cal_file_calc['Volume required to reach Nat Avg'])


final_stock_cal_file_calc['OOS_USD'] = final_stock_cal_file_calc['OOS_min']*final_stock_cal_file_calc['MACOwoVLC/HL']
# final_stock_cal_file_calc['Obs_USD'] = final_stock_cal_file_calc['Obs_min']*final_stock_cal_file_calc['MACOwoVLC/HL']


oos_df_calc = oos_df.merge(maco_wo_vlc_df,on='Material_Number',how='left')
oos_df_calc['MACOwoVLC/HL'] = oos_df_calc['MACOwoVLC/HL'].fillna(median_maco)
oos_df_calc['OOS_USD'] = oos_df_calc['OOS']*oos_df_calc['MACOwoVLC/HL']


oos_df_calc = oos_df_calc[oos_df_calc['Date'].isin(recommendation_week)]
oos_df_calc.OOS_USD.sum()

oos_df_calc = oos_df_calc.merge(uom_df.loc[(uom_df['Convert Unit']=='PAL') & (uom_df['Base Unit']=='HL'),['Material_Number','Conversion Factor']],on='Material_Number',how='left')
oos_df_calc['OOS pal'] = oos_df_calc['OOS']*oos_df_calc['Conversion Factor']
oos_df_calc.drop(columns={'Conversion Factor'},inplace=True)
oos_df_calc.head()


trip_cost_df['Trip Cost per pal'] = trip_cost_df['Trip Cost']/15 #(15 pallets is the average pallet capacity of truck)






import os
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
import warnings
warnings.filterwarnings('ignore')
import pyodbc
from datetime import date
from pulp import *
import math
from threading import Thread
import time
from itertools import combinations


#Input paths 
root_path = Path(os.getcwd())
input_path = root_path / "Inputs"
min_date = pd.to_datetime('2022-03-01')
input_path = root_path / "Inputs"
preprocessed_path = input_path/"Preprocessed_files"

#Updated currency conversion rate
usd_to_pesos = 57.40
    
output_path = root_path / "Outputs" /str(min_date)[:-9]
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("created")
else:
    print("exists")
    
recommendation_week = list(pd.date_range(min_date,end=min_date+timedelta(days=6),freq='D'))
recommendation_week_old = list(pd.date_range(min_date+timedelta(days=-7),end=min_date+timedelta(days=-1),freq='D'))



maco_wo_vlc_df = pd.read_excel(preprocessed_path/"MACO_wo_VLC.xlsx")
customer_orders_df = pd.read_excel(preprocessed_path/"Customer_order.xlsx")
oos_df = pd.read_excel(preprocessed_path/"OOS.xlsx")
# stock_age_df = pd.read_excel(preprocessed_path/"Obsolescence_Report.xlsx")
planned_in_df = pd.read_excel(preprocessed_path/"Planned_In.xlsx")
avg_demand_df = pd.read_excel(preprocessed_path/"Average_Demand.xlsx")
intransit_volume_df = pd.read_excel(preprocessed_path/"Intransit_Volume.xlsx")
opening_stock_df = pd.read_excel(preprocessed_path/"Opening_Stock.xlsx")
trip_cost_df = pd.read_excel(preprocessed_path/"Trip_Cost.xlsx")
reorder_point_df = pd.read_excel(preprocessed_path/"Reorder_Point.xlsx")
dc_mapping = pd.read_excel(preprocessed_path/"DC_mapping.xlsx")
uom_df = pd.read_excel(preprocessed_path/"UOM_Conversion.xlsx")
planned_prod_df = pd.read_excel(preprocessed_path/"planned_prod_df.xlsx")


maco_wo_vlc_df['Material_Number'] = maco_wo_vlc_df['Material_Number'].astype(str)
customer_orders_df['Material_Number'] = customer_orders_df['Material_Number'].astype(str)
oos_df['Material_Number'] = oos_df['Material_Number'].astype(str)
# stock_age_df['Material_Number'] = stock_age_df['Material_Number'].astype(str)
planned_in_df['Material_Number'] = planned_in_df['Material_Number'].astype(str)
avg_demand_df['Material_Number'] = avg_demand_df['Material_Number'].astype(str)
avg_demand_df['Date'] = pd.to_datetime(avg_demand_df['Date'])
intransit_volume_df['Material_Number'] = intransit_volume_df['Material_Number'].astype(str)
opening_stock_df['Material_Number'] = opening_stock_df['Material_Number'].astype(str)
reorder_point_df['Material_Number'] = reorder_point_df['Material_Number'].astype(str)
uom_df['Material_Number'] = uom_df['Material_Number'].astype(str)
planned_prod_df['Material_Number'] = planned_prod_df['Material_Number'].astype(str)



# uom_df['Conversion Factor'].mean()
uom_df = uom_df[~(uom_df['Material_Number']=='nan')]


#Constraining the dates
#CHECK
customer_orders_df = customer_orders_df[customer_orders_df['Date'].isin([recommendation_week[0]])].reset_index(drop=True)
avg_demand_df = avg_demand_df[avg_demand_df['Date'].isin(recommendation_week)].reset_index(drop=True)
planned_in_df = planned_in_df[planned_in_df['Date'].isin(recommendation_week)].reset_index(drop=True)
planned_prod_df = planned_prod_df[planned_prod_df['PRODUCTIONDATE'].isin(recommendation_week)].reset_index(drop=True)
oos_df = oos_df[oos_df['Date'].isin(recommendation_week)].reset_index(drop=True)
# oos_df = oos_df[oos_df['Date'].isin(recommendation_week_old)].reset_index(drop=True)
intransit_volume_df = intransit_volume_df[intransit_volume_df['Date'].isin([recommendation_week[0]])].reset_index(drop=True)
#CHECK
opening_stock_df = opening_stock_df[opening_stock_df['Date'].isin([recommendation_week[0]])].reset_index(drop=True)


# ### CHECK AND CHANGE ###
# all_dates_df = pd.DataFrame(pd.date_range(pd.to_datetime('2022-02-01'),end=pd.to_datetime('2022-03-09'),freq='D')).rename(columns={0:'Date'})
# avg_demand_df['comb'] = avg_demand_df['Site_Code'].astype(str) + '_' + avg_demand_df['Material_Number'].astype(str)
# avg_demand_df_new = pd.DataFrame(columns=avg_demand_df.columns) 
# for pair in avg_demand_df['comb'].unique():
#     avg_demand_df_iter = avg_demand_df[avg_demand_df['comb']==pair]
#     avg_demand_df_iter = all_dates_df.merge(avg_demand_df_iter,on='Date',how='left')
#     avg_demand_df_iter['Site_Code'] = avg_demand_df_iter['Site_Code'].ffill()
#     avg_demand_df_iter['Material_Number'] = avg_demand_df_iter['Material_Number'].ffill()
#     avg_demand_df_iter['comb'] = avg_demand_df_iter['comb'].ffill()
#     avg_demand_df_iter['Site_Code'] = avg_demand_df_iter['Site_Code'].bfill()
#     avg_demand_df_iter['Material_Number'] = avg_demand_df_iter['Material_Number'].bfill()
#     avg_demand_df_iter['comb'] = avg_demand_df_iter['comb'].bfill()
#     avg_demand_df_iter['Daily_Demand'] = avg_demand_df_iter['Daily_Demand'].fillna(0)
#     avg_demand_df_new = avg_demand_df_new.append(avg_demand_df_iter)
# avg_demand_df_new.drop(columns='comb',inplace=True)
# inventory_policy = all_df.merge(opening_stock_df,on=['Site_Code','Material_Number','Date'],how='outer')
inventory_policy = opening_stock_df.merge(avg_demand_df,on=['Site_Code','Material_Number','Date'],how='outer')
inventory_policy = inventory_policy.merge(customer_orders_df,on=['Site_Code','Material_Number','Date'],how='outer')
inventory_policy = inventory_policy.merge(reorder_point_df,on=['Site_Code','Material_Number'],how='left')



# inventory_policy = avg_demand_df.merge(customer_orders_df,on=['Site_Code','Material_Number','Date'],how='outer')
# inventory_policy = inventory_policy.merge(opening_stock_df,on=['Site_Code','Material_Number','Date'],how='outer')
# inventory_policy = inventory_policy.merge(reorder_point_df,on=['Site_Code','Material_Number'],how='outer')
inventory_policy['Daily_Demand']=inventory_policy['Daily_Demand'].fillna(0)
inventory_policy['Customer_Orders']=inventory_policy['Customer_Orders'].fillna(0)
inventory_policy['Opening_Stock']=inventory_policy['Opening_Stock'].fillna(0)
inventory_policy['Reorder_Point_Hl']=inventory_policy['Reorder_Point_Hl'].fillna(inventory_policy['Reorder_Point_Hl'].median())
inventory_policy = inventory_policy[~inventory_policy['Date'].isna()]
# inventory_policy.isnull().sum()
# inventory_policy = inventory_policy.fillna(0).reset_index(drop=True)


inventory_policy = inventory_policy[~((inventory_policy['Material_Number'].isna())|(inventory_policy['Material_Number']=='nan'))]
inventory_policy = inventory_policy[~((inventory_policy['Site_Code'].isna())|(inventory_policy['Site_Code']=='nan'))]


cols=['Site_Code']
inventory_policy[cols]=inventory_policy[cols].astype(str)
planned_in_df[cols]=planned_in_df[cols].astype(str)
planned_prod_df[cols]=planned_prod_df[cols].astype(str)
# planned_out_volume_df[cols]=planned_out_volume_df[cols].astype(str)
intransit_volume_df[cols]=intransit_volume_df[cols].astype(str)
# inventory_policy['Material_Number']=inventory_policy['Material_Number'].astype(str)
# planned_in_df['Material_Number']=planned_in_df['Material_Number'].astype(str)
# planned_out_volume_df['Material_Number']=planned_out_volume_df['Material_Number'].astype(str)
# intransit_volume_df['Material_Number']=intransit_volume_df['Material_Number'].astype(str)
stock_at_hand = pd.merge(inventory_policy,planned_in_df, how='left',on=['Date','Site_Code', 'Material_Number'])
# stock_at_hand = pd.merge(stock_at_hand,planned_out_volume_df, how='left',on=['Date','Site_Code', 'Material_Number'])
stock_at_hand = pd.merge(stock_at_hand,planned_prod_df.rename(columns={'PRODUCTIONDATE':'Date'}), how='left',on=['Date','Site_Code', 'Material_Number'])
stock_at_hand = pd.merge(stock_at_hand,intransit_volume_df, how='left',on=['Date','Site_Code', 'Material_Number'])
stock_at_hand = stock_at_hand.fillna(value=0)



stock_at_hand['Final Demand']=np.where(stock_at_hand['Customer_Orders']>stock_at_hand['Daily_Demand'],stock_at_hand['Customer_Orders'],stock_at_hand['Daily_Demand'])
stock_at_hand['Free_Stock'] = stock_at_hand['Opening_Stock']
stock_at_hand_7days=stock_at_hand.groupby(['Site_Code','Material_Number']).agg({'Planned_In':'sum',
#                                                                                 'Planned_Out' : 'sum',
                                                                                'Production' : 'sum',
                                                                                'In_Transit' : 'sum',
                                                                                'Final Demand':'sum'
                                                      }).reset_index().rename(columns = {'Planned_In' : 'Planned_In_7days',
#                                                                                          'Planned_Out' : 'Planned_Out_3days',
                                                                                         
                                                                                         'In_Transit' : 'In_Transit_7days',
                                                                                         'Final Demand': 'Demand_7days'
                                                                                         })



stock_at_hand_7days = stock_at_hand_7days[~((stock_at_hand_7days['Material_Number'].isna())|(stock_at_hand_7days['Material_Number']=='nan'))]
stock_at_hand_7days = stock_at_hand_7days[~((stock_at_hand_7days['Site_Code'].isna())|(stock_at_hand_7days['Site_Code']=='nan'))]


depot_code_dict=dict(list(zip(dc_mapping['Depot Code'],dc_mapping['Depot Description']))) 
sku_name_dict=dict(list(zip(uom_df['Material_Number'],uom_df['SKU Name']))) 



# stock_at_hand_today1=stock_at_hand[stock_at_hand['Date']==pd.to_datetime('2022-05-31')]
stock_at_hand_today1=stock_at_hand.copy()
stock_at_hand_Final1=pd.merge(stock_at_hand_today1.drop(columns={'Production'}),stock_at_hand_7days, how='left',on=['Site_Code', 'Material_Number'])    

# cols=['Site_Code']
# stock_at_hand_Final1=stock_at_hand_Final1.merge(stock_age_df3[['Site_Code','Material_Number','Obs (USD)','Vol']],on=['Site_Code', 'Material_Number'],how='left')

stock_at_hand_Final1.head()



###################### Incorporating production ###############################################3

stock_at_hand_today=stock_at_hand[stock_at_hand['Date']==min_date]
stock_at_hand_Final=pd.merge(stock_at_hand_today.drop(columns={'Production'}),stock_at_hand_7days, how='left',on=['Site_Code', 'Material_Number'])
# production_data=planned_prod_df.copy(deep=True)
# production_data_3_days=production_data.copy(deep=True)
# production_data_total=production_data_3_days.groupby(['Material_Number','Site_Code'])['Production'].sum().reset_index()
# production_data_total['Material_Number']=production_data_total['Material_Number'].astype(str)
stock_at_hand_Final.head()



############################### Final stock at hand,stock on floor for 7 days, Predicted Variance and National Average calculation############################


# stock_at_hand_Final=stock_at_hand_Final.merge(production_data_total[['Material_Number','Site_Code','Production']],left_on=['Material_Number','Site_Code'],right_on=['Material_Number','Site_Code'],how='left')
# stock_at_hand_Final['Production']=stock_at_hand_Final['Production'].fillna(0)
stock_at_hand_Final['UID']=stock_at_hand_Final['Site_Code'].astype(str)+"_"+stock_at_hand_Final['Material_Number'].astype(str)

stock_at_hand_Final['%Variance_vs_ROP']=((stock_at_hand_Final['Free_Stock']-stock_at_hand_Final['Reorder_Point_Hl'])/stock_at_hand_Final['Reorder_Point_Hl'])*100

stock_at_hand_Final['Planned_Loads_7days']=stock_at_hand_Final['Planned_In_7days']+stock_at_hand_Final['In_Transit_7days']

stock_at_hand_Final['ATD_7Days']=stock_at_hand_Final['Free_Stock']+stock_at_hand_Final['Planned_Loads_7days']+stock_at_hand_Final['Production']-stock_at_hand_Final['Demand_7days']

stock_at_hand_Final['%Predicted_Var_vs_ROP_7days']=((stock_at_hand_Final['ATD_7Days']-stock_at_hand_Final['Reorder_Point_Hl'])/stock_at_hand_Final['Reorder_Point_Hl'])*100

Final_Dataframe=stock_at_hand_Final[['Material_Number','Site_Code','Free_Stock','Reorder_Point_Hl','%Variance_vs_ROP',
                                     'Planned_Loads_7days','Demand_7days','ATD_7Days','%Predicted_Var_vs_ROP_7days','Production'
                                    ]]

Final_Dataframe['%Predicted_Var_vs_ROP_7days'] = Final_Dataframe['%Predicted_Var_vs_ROP_7days'].apply(lambda x: '{:.2f}'.format(x))

Final_Dataframe_Display=Final_Dataframe.rename(columns = {'Material_Number' : 'SKU',
                                                'Free_Stock' : 'Stock_on_Floor',
                                                '%Predicted_Var_vs_ROP_7days': '%Predicted_Var_vs ROP 7days',
                                                })




cols = list(set(list(Final_Dataframe_Display.columns)) - set(['SKU', 'Site_Code']))
Final_Dataframe_Display[cols] = Final_Dataframe_Display[cols].apply(pd.to_numeric, errors='coerce')

Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'] = Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'].apply(lambda x: '{:.2f}'.format(x))
National_Sum=Final_Dataframe_Display.groupby(['SKU']).agg({'Stock_on_Floor':'sum','Reorder_Point_Hl' : 'sum',
                                                            'ATD_7Days' : 'sum','Demand_7days':'sum'                                                                    
                                                      }).reset_index().rename(columns = {'Stock_on_Floor' : 'Stock_on_Floor_National',
                                                                                         'Reorder_Point_Hl' : 'Reorder_Point_Hl_National',
                                                                                         'ATD_7Days' : 'ATD_7Days_National',
                                                                                         'Demand_7days': 'Demand_7days_National'
                                                                                         })

National_Sum['National Average']=((National_Sum['ATD_7Days_National']-National_Sum['Reorder_Point_Hl_National'])/National_Sum['Reorder_Point_Hl_National'])*100
Final_Dataframe_Display=Final_Dataframe_Display.merge(National_Sum[['SKU','National Average']],left_on='SKU',right_on='SKU',how='left')
Final_Dataframe_Display['%Predicted_Var_vs ROP 7days']=Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'].astype(float)



# Final_Dataframe_Display['HL Above/Below']=((Final_Dataframe_Display['National Average']-Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'])/100)*Final_Dataframe_Display['Reorder_Point_Hl']

# ##CHECK##
# Final_Dataframe_Display['HL Above/Below new']=((-Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'])/100)*Final_Dataframe_Display['Reorder_Point_Hl']
# # Final_Dataframe_Display['HL Above/Below'] = Final_Dataframe_Display['HL Above/Below new'].copy()
# Final_Dataframe_Display['HL Above/Below']=np.where(Final_Dataframe_Display['HL Above/Below']>Final_Dataframe_Display['HL Above/Below new'],Final_Dataframe_Display['HL Above/Below'],Final_Dataframe_Display['HL Above/Below new'])
    
Final_Dataframe_Display['HL Above/Below'] =  ((-Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'])/100)*Final_Dataframe_Display['Reorder_Point_Hl']
Final_Dataframe_Display['HL Above/Below new']=((Final_Dataframe_Display['National Average']-Final_Dataframe_Display['%Predicted_Var_vs ROP 7days'])/100)*Final_Dataframe_Display['Reorder_Point_Hl']
Final_Dataframe_Display['HL Above/Below']=np.where(Final_Dataframe_Display['HL Above/Below']<=0,Final_Dataframe_Display['HL Above/Below'],
                                                   np.where(Final_Dataframe_Display['HL Above/Below']>Final_Dataframe_Display['HL Above/Below new'],Final_Dataframe_Display['HL Above/Below'],Final_Dataframe_Display['HL Above/Below new']))
Final_Dataframe_Display = Final_Dataframe_Display.drop(columns={'HL Above/Below new'})
    
# Final_Dataframe_Display.drop(columns={'HL Above/Below new'},inplace=True)
#Final_Dataframe_Display.to_excel(path+'Data for Solution before exclusion.xlsx',index=False)
# exclude_dc_skus=pd.read_excel(static_path+'Excluded SKUs.xlsx',sheet_name='Sheet1')
# exclude_dc_skus['UID']=exclude_dc_skus['Depot Code'].astype(str)+"_"+exclude_dc_skus['SKU Code'].astype(str)
Final_Dataframe_Display1=Final_Dataframe_Display.copy(deep=True)
Final_Dataframe_Display1['UID']=Final_Dataframe_Display['Site_Code'].astype(str)+"_"+Final_Dataframe_Display['SKU'].astype(str)
# Final_Dataframe_Display1=Final_Dataframe_Display1[~(Final_Dataframe_Display1['UID'].isin(list(exclude_dc_skus['UID'].unique())))]



############################## Prep of stock calculation sheet for the output file##################

def stock_cal(inventory_policy,planned_prod_df,Planned_in,In_Transit,sku_name_dict,depot_code_dict,Final_Dataframe_Display):
    
    min_date=min(inventory_policy['Date'])
    date_2=min_date+timedelta(days=1)
    date_3=min_date+timedelta(days=2)
    date_4=min_date+timedelta(days=3)
    date_5=min_date+timedelta(days=4)
    date_6=min_date+timedelta(days=5)
    date_7=min_date+timedelta(days=6)
    
    transit_current=In_Transit[In_Transit['Date']==min_date]
    transit_current['UID']=transit_current['Site_Code'].astype(str)+"_"+transit_current['Material_Number'].astype(str)
    transit_current_grouped=transit_current.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_current_grouped_dict=dict(list(zip(transit_current_grouped['UID'],transit_current_grouped['In_Transit'])))
    
    transit_day2=In_Transit[In_Transit['Date']==date_2]
    transit_day2['UID']=transit_day2['Site_Code'].astype(str)+"_"+transit_day2['Material_Number'].astype(str)
    transit_day2_grouped=transit_day2.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_day2_grouped_dict=dict(list(zip(transit_day2_grouped['UID'],transit_day2_grouped['In_Transit'])))
    
    transit_day3=In_Transit[In_Transit['Date']==date_3]
    transit_day3['UID']=transit_day3['Site_Code'].astype(str)+"_"+transit_day3['Material_Number'].astype(str)
    transit_day3_grouped=transit_day3.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_day3_grouped_dict=dict(list(zip(transit_day3_grouped['UID'],transit_day3_grouped['In_Transit'])))
        
    transit_day4=In_Transit[In_Transit['Date']==date_4]
    transit_day4['UID']=transit_day4['Site_Code'].astype(str)+"_"+transit_day4['Material_Number'].astype(str)
    transit_day4_grouped=transit_day4.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_day4_grouped_dict=dict(list(zip(transit_day4_grouped['UID'],transit_day4_grouped['In_Transit'])))
    
    transit_day5=In_Transit[In_Transit['Date']==date_5]
    transit_day5['UID']=transit_day5['Site_Code'].astype(str)+"_"+transit_day5['Material_Number'].astype(str)
    transit_day5_grouped=transit_day5.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_day5_grouped_dict=dict(list(zip(transit_day5_grouped['UID'],transit_day5_grouped['In_Transit'])))
    
    transit_day6=In_Transit[In_Transit['Date']==date_6]
    transit_day6['UID']=transit_day6['Site_Code'].astype(str)+"_"+transit_day6['Material_Number'].astype(str)
    transit_day6_grouped=transit_day6.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_day6_grouped_dict=dict(list(zip(transit_day6_grouped['UID'],transit_day6_grouped['In_Transit'])))
    
    transit_day7=In_Transit[In_Transit['Date']==date_7]
    transit_day7['UID']=transit_day7['Site_Code'].astype(str)+"_"+transit_day7['Material_Number'].astype(str)
    transit_day7_grouped=transit_day7.groupby(['UID','Site_Code','Material_Number'])['In_Transit'].sum().reset_index()
    transit_day7_grouped_dict=dict(list(zip(transit_day7_grouped['UID'],transit_day7_grouped['In_Transit'])))
            
    Planned_in_current=Planned_in[Planned_in['Date']==min_date]
    Planned_in_current['UID']=Planned_in_current['Site_Code'].astype(str)+"_"+Planned_in_current['Material_Number'].astype(str)
    planned_in_c_dict=dict(list(zip(Planned_in_current['UID'],Planned_in_current['Planned_In'])))
    
    Planned_in_day1=Planned_in[Planned_in['Date']==date_2]
    Planned_in_day1['UID']=Planned_in_day1['Site_Code'].astype(str)+"_"+Planned_in_day1['Material_Number'].astype(str)
    planned_in_day1_dict=dict(list(zip(Planned_in_day1['UID'],Planned_in_day1['Planned_In'])))
    
    Planned_in_day2=Planned_in[Planned_in['Date']==date_3]
    Planned_in_day2['UID']=Planned_in_day2['Site_Code'].astype(str)+"_"+Planned_in_day2['Material_Number'].astype(str)
    planned_in_day2_dict=dict(list(zip(Planned_in_day2['UID'],Planned_in_day2['Planned_In'])))
    
    Planned_in_day3=Planned_in[Planned_in['Date']==date_4]
    Planned_in_day3['UID']=Planned_in_day3['Site_Code'].astype(str)+"_"+Planned_in_day3['Material_Number'].astype(str)
    planned_in_day3_dict=dict(list(zip(Planned_in_day3['UID'],Planned_in_day3['Planned_In'])))
    
    Planned_in_day4=Planned_in[Planned_in['Date']==date_5]
    Planned_in_day4['UID']=Planned_in_day4['Site_Code'].astype(str)+"_"+Planned_in_day4['Material_Number'].astype(str)
    planned_in_day4_dict=dict(list(zip(Planned_in_day4['UID'],Planned_in_day4['Planned_In'])))

    Planned_in_day5=Planned_in[Planned_in['Date']==date_6]
    Planned_in_day5['UID']=Planned_in_day5['Site_Code'].astype(str)+"_"+Planned_in_day5['Material_Number'].astype(str)
    planned_in_day5_dict=dict(list(zip(Planned_in_day5['UID'],Planned_in_day5['Planned_In'])))
    
    Planned_in_day6=Planned_in[Planned_in['Date']==date_7]
    Planned_in_day6['UID']=Planned_in_day6['Site_Code'].astype(str)+"_"+Planned_in_day6['Material_Number'].astype(str)
    planned_in_day6_dict=dict(list(zip(Planned_in_day6['UID'],Planned_in_day6['Planned_In'])))
        
#     Planned_out_current=Planned_out[Planned_out['Date']==min_date]
#     Planned_out_current['UID']=Planned_out_current['Site_Code'].astype(str)+"_"+Planned_out_current['Material_Number'].astype(str)
#     planned_out_c_dict=dict(list(zip(Planned_out_current['UID'],Planned_out_current['Planned_Out'])))
    
#     Planned_out_day1=Planned_out[Planned_out['Date']==date_2]
#     Planned_out_day1['UID']=Planned_out_day1['Site_Code'].astype(str)+"_"+Planned_out_day1['Material_Number'].astype(str)   
#     planned_out_day1_dict=dict(list(zip(Planned_out_day1['UID'],Planned_out_day1['Planned_Out'])))
      
#     Planned_out_day2=Planned_out[Planned_out['Date']==date_3]
#     Planned_out_day2['UID']=Planned_out_day2['Site_Code'].astype(str)+"_"+Planned_out_day2['Material_Number'].astype(str)
#     planned_out_day2_dict=dict(list(zip(Planned_out_day2['UID'],Planned_out_day2['Planned_Out'])))
  
    date_list=list(planned_prod_df['PRODUCTIONDATE'].unique())
    production_as_on_date=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==min_date]
    production_as_on_date['UID']=production_as_on_date['Site_Code'].astype(str)+"_"+production_as_on_date['Material_Number'].astype(str)
    production_as_on_date_grouped=production_as_on_date.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_date_dict=dict(list(zip(production_as_on_date_grouped['UID'],production_as_on_date_grouped['Production'])))
    
    production_as_on_day7=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==date_7]
    production_as_on_day7['UID']=production_as_on_day7['Site_Code'].astype(str)+"_"+production_as_on_day7['Material_Number'].astype(str)
    production_as_on_day7_grouped=production_as_on_day7.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_day7_dict=dict(list(zip(production_as_on_day7_grouped['UID'],production_as_on_day7_grouped['Production'])))
    
    production_as_on_day2=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==date_2]
    production_as_on_day2['UID']=production_as_on_day2['Site_Code'].astype(str)+"_"+production_as_on_day2['Material_Number'].astype(str)
    production_as_on_day2_grouped=production_as_on_day2.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_day2_dict=dict(list(zip(production_as_on_day2_grouped['UID'],production_as_on_day2_grouped['Production'])))

    production_as_on_day3=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==date_3]
    production_as_on_day3['UID']=production_as_on_day3['Site_Code'].astype(str)+"_"+production_as_on_day3['Material_Number'].astype(str)
    production_as_on_day3_grouped=production_as_on_day3.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_day3_dict=dict(list(zip(production_as_on_day3_grouped['UID'],production_as_on_day3_grouped['Production'])))

    production_as_on_day4=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==date_4]
    production_as_on_day4['UID']=production_as_on_day4['Site_Code'].astype(str)+"_"+production_as_on_day4['Material_Number'].astype(str)
    production_as_on_day4_grouped=production_as_on_day4.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_day4_dict=dict(list(zip(production_as_on_day4_grouped['UID'],production_as_on_day4_grouped['Production'])))
    
    production_as_on_day5=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==date_5]
    production_as_on_day5['UID']=production_as_on_day5['Site_Code'].astype(str)+"_"+production_as_on_day5['Material_Number'].astype(str)
    production_as_on_day5_grouped=production_as_on_day5.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_day5_dict=dict(list(zip(production_as_on_day5_grouped['UID'],production_as_on_day5_grouped['Production'])))

    production_as_on_day6=planned_prod_df[planned_prod_df['PRODUCTIONDATE']==date_6]
    production_as_on_day6['UID']=production_as_on_day6['Site_Code'].astype(str)+"_"+production_as_on_day6['Material_Number'].astype(str)
    production_as_on_day6_grouped=production_as_on_day6.groupby(['UID'])['Production'].sum().reset_index()
    production_as_on_day6_dict=dict(list(zip(production_as_on_day6_grouped['UID'],production_as_on_day6_grouped['Production'])))

    ################## Stock on floor#################################################
    
    on_floor=inventory_policy[inventory_policy['Date']==min_date]
    on_floor_day7=inventory_policy[inventory_policy['Date']==date_7]
    
    
    ##CHECK##
    on_floor_day2 = inventory_policy[inventory_policy['Date']==date_2][['Site_Code','Material_Number','Daily_Demand']]
    on_floor_day2['UID']=on_floor_day2['Site_Code'].astype(str)+"_"+on_floor_day2['Material_Number'].astype(str)
    demand_day2_dict=dict(list(zip(on_floor_day2['UID'],on_floor_day2['Daily_Demand'])))

    on_floor_day3 = inventory_policy[inventory_policy['Date']==date_3][['Site_Code','Material_Number','Daily_Demand']]
    on_floor_day3['UID']=on_floor_day3['Site_Code'].astype(str)+"_"+on_floor_day3['Material_Number'].astype(str)
    demand_day3_dict=dict(list(zip(on_floor_day3['UID'],on_floor_day3['Daily_Demand'])))
    
    on_floor_day4 = inventory_policy[inventory_policy['Date']==date_4][['Site_Code','Material_Number','Daily_Demand']]
    on_floor_day4['UID']=on_floor_day4['Site_Code'].astype(str)+"_"+on_floor_day4['Material_Number'].astype(str)
    demand_day4_dict=dict(list(zip(on_floor_day4['UID'],on_floor_day4['Daily_Demand'])))
    
    on_floor_day5 = inventory_policy[inventory_policy['Date']==date_5][['Site_Code','Material_Number','Daily_Demand']]
    on_floor_day5['UID']=on_floor_day5['Site_Code'].astype(str)+"_"+on_floor_day5['Material_Number'].astype(str)
    demand_day5_dict=dict(list(zip(on_floor_day5['UID'],on_floor_day5['Daily_Demand'])))
    
    on_floor_day6 = inventory_policy[inventory_policy['Date']==date_6][['Site_Code','Material_Number','Daily_Demand']]
    on_floor_day6['UID']=on_floor_day6['Site_Code'].astype(str)+"_"+on_floor_day6['Material_Number'].astype(str)
    demand_day6_dict=dict(list(zip(on_floor_day6['UID'],on_floor_day6['Daily_Demand'])))
    
    ##################
    
    on_floor=on_floor[['Date','Site_Code','Material_Number','Reorder_Point_Hl','Opening_Stock','Daily_Demand','Customer_Orders']]
    on_floor['UID']=on_floor['Site_Code'].astype(str)+"_"+on_floor['Material_Number'].astype(str)
    demand_current_day_dict=dict(list(zip(on_floor['UID'],on_floor['Daily_Demand'])))
    on_floor_day7['UID']=on_floor_day7['Site_Code'].astype(str)+"_"+on_floor_day7['Material_Number'].astype(str)
    demand_day7_dict=dict(list(zip(on_floor_day7['UID'],on_floor_day7['Daily_Demand'])))
    
    
    #####CHECKKKKK#####
    demand_day2_dict = demand_day2_dict
    demand_day3_dict = demand_day3_dict
    demand_day4_dict = demand_day4_dict
    demand_day5_dict = demand_day5_dict
    demand_day6_dict = demand_day6_dict
    demand_day7_dict = demand_day7_dict
    #####CHECKKKKK#####    
    
    
    on_floor['Planned In Day1']=on_floor['UID'].map(planned_in_c_dict)
    on_floor['In Transit Day1']=on_floor['UID'].map(transit_current_grouped_dict)
    on_floor['Production Day1']=on_floor['UID'].map(production_as_on_date_dict)
#     on_floor['Planned Out Day1']=on_floor['UID'].map(planned_out_c_dict)
    on_floor['Final Demand']=np.where(on_floor['Customer_Orders']>on_floor['Daily_Demand'],on_floor['Customer_Orders'],on_floor['Daily_Demand'])
    
    on_floor=on_floor.fillna(0)
    on_floor['Stock on floor']=on_floor['Opening_Stock']
    on_floor['Site Name']=on_floor['Site_Code'].map(depot_code_dict)
    on_floor['Material_Number']=on_floor['Material_Number'].astype(str)
    on_floor['SKU Name']=on_floor['Material_Number'].map(sku_name_dict)    
    
    
    ################################ Stock for 7 days calculation ######################################3
    
    final=on_floor[['Date','Site_Code','Material_Number','UID','Reorder_Point_Hl','Stock on floor','Production Day1','Planned In Day1','In Transit Day1','Customer_Orders','Daily_Demand']]
    final['Planned In Day2']=final['UID'].map(planned_in_day1_dict)
    final['Planned In Day3']=final['UID'].map(planned_in_day2_dict)
    final['Planned In Day4']=final['UID'].map(planned_in_day3_dict)
    final['Planned In Day5']=final['UID'].map(planned_in_day4_dict)
    final['Planned In Day6']=final['UID'].map(planned_in_day5_dict)
    final['Planned In Day7']=final['UID'].map(planned_in_day6_dict)
    
    ###CHECKKK###
#     final['Planned In Day2']=final['UID'].map(planned_in_c_dict)
#     final['Planned In Day3']=final['UID'].map(planned_in_c_dict)
#     final['Planned In Day4']=final['UID'].map(planned_in_c_dict)
#     final['Planned In Day5']=final['UID'].map(planned_in_c_dict)
#     final['Planned In Day6']=final['UID'].map(planned_in_c_dict)
#     final['Planned In Day7']=final['UID'].map(planned_in_c_dict)
    
    final['In Transit Day2']=final['UID'].map(transit_day2_grouped_dict)
    final['In Transit Day3']=final['UID'].map(transit_day3_grouped_dict)
    final['In Transit Day4']=final['UID'].map(transit_day4_grouped_dict)
    final['In Transit Day5']=final['UID'].map(transit_day5_grouped_dict)
    final['In Transit Day6']=final['UID'].map(transit_day6_grouped_dict)
    final['In Transit Day7']=final['UID'].map(transit_day7_grouped_dict)
    final['Production Day2']=final['UID'].map(production_as_on_day2_dict)
    final['Production Day3']=final['UID'].map(production_as_on_day3_dict)
    final['Production Day4']=final['UID'].map(production_as_on_day4_dict)
    final['Production Day5']=final['UID'].map(production_as_on_day5_dict)
    final['Production Day6']=final['UID'].map(production_as_on_day6_dict)
    final['Production Day7']=final['UID'].map(production_as_on_day7_dict)
#     final['Planned Out Day2']=final['UID'].map(planned_out_day1_dict)
#     final['Planned Out Day3']=final['UID'].map(planned_out_day2_dict)
    final.rename(columns={'Daily_Demand': 'Daily Demand Day1'},inplace=True)
    final['Daily Demand Day1']=final['UID'].map(demand_current_day_dict)
    final['Daily Demand Day2']=final['UID'].map(demand_day2_dict)
    final['Daily Demand Day3']=final['UID'].map(demand_day3_dict)
    final['Daily Demand Day4']=final['UID'].map(demand_day4_dict)
    final['Daily Demand Day5']=final['UID'].map(demand_day5_dict)
    final['Daily Demand Day6']=final['UID'].map(demand_day6_dict)
    final['Daily Demand Day7']=final['UID'].map(demand_day7_dict)
    final['Day1 final demand']=np.where(final['Customer_Orders']>final['Daily Demand Day1'],final['Customer_Orders'],final['Daily Demand Day1'])
    final=final.fillna(0)
    
    final['Total 7Days Planned In']=final['Planned In Day1']+final['Planned In Day2']+final['Planned In Day3']+final['Planned In Day4']+final['Planned In Day5']+final['Planned In Day6']+final['Planned In Day7']
    final['Total 7Days In Transit']=final['In Transit Day1']+final['In Transit Day2']+final['In Transit Day3']+final['In Transit Day4']+final['In Transit Day5']+final['In Transit Day6']+final['In Transit Day7']
    final['Total 7Days Production']=final['Production Day1']+final['Production Day2']+final['Production Day3']+final['Production Day4']+final['Production Day5']+final['Production Day6']+final['Production Day7']
#     final['Total 3Days Planned Out']=final['Planned Out Day1']+final['Planned Out Day2']+final['Planned Out Day3']
    final['Total 7Days Demand']=final['Day1 final demand']+final['Daily Demand Day2']+final['Daily Demand Day3']+final['Daily Demand Day4']+final['Daily Demand Day5']+final['Daily Demand Day6']+final['Daily Demand Day7']
    final['Total Planned Load 7Days']=final['Total 7Days Planned In']+final['Total 7Days In Transit']
    
    final.rename(columns={'Material_Number':'SKU'},inplace=True)
    
    final['Stock on floor']=final['Stock on floor'].astype(float)
    final['Stock on floor 7Days']=final['Stock on floor']+final['Total 7Days Planned In']+final['Total 7Days In Transit']+final['Total 7Days Production']-final['Total 7Days Demand']
    final['%Variance_vs_ROP']=(final['Stock on floor']-final['Reorder_Point_Hl'])/final['Reorder_Point_Hl']
    final['%Predicted_Var_vs ROP 7days']=(final['Stock on floor 7Days']-final['Reorder_Point_Hl'])/final['Reorder_Point_Hl']
    
    Final_Dataframe_Display['SKU']=Final_Dataframe_Display['SKU'].astype(str)
    nat=dict(list(zip(Final_Dataframe_Display['SKU'],Final_Dataframe_Display['National Average'])))
    final['SKU']=final['SKU'].astype(str)
    final['National Average']=final['SKU'].map(nat)
    final['National Average']=final['National Average'].astype(float)
    final['National Average']=final['National Average']/100
    
#     final['Volume available for Redeployment/Volume required to reach Nat Avg']=(final['National Average']-final['%Predicted_Var_vs ROP 7days'])*final['Reorder_Point_Hl']
    
#     ##CHECK##
#     final['Volume available for Redeployment/Volume required to reach Nat Avg new']=(-final['%Predicted_Var_vs ROP 7days'])*final['Reorder_Point_Hl']
    final['Volume available for Redeployment/Volume required to reach Nat Avg']=(-final['%Predicted_Var_vs ROP 7days'])*final['Reorder_Point_Hl']    
    final['Volume available for Redeployment/Volume required to reach Nat Avg new']=((final['National Average']-final['%Predicted_Var_vs ROP 7days']))*final['Reorder_Point_Hl']
    final['Volume available for Redeployment/Volume required to reach Nat Avg']=np.where(final['Volume available for Redeployment/Volume required to reach Nat Avg']<=0,final['Volume available for Redeployment/Volume required to reach Nat Avg'],
                                                                                         np.where(final['Volume available for Redeployment/Volume required to reach Nat Avg']>final['Volume available for Redeployment/Volume required to reach Nat Avg new'],final['Volume available for Redeployment/Volume required to reach Nat Avg'],final['Volume available for Redeployment/Volume required to reach Nat Avg new']))
    final = final.drop(columns={'Volume available for Redeployment/Volume required to reach Nat Avg new'})
    


    final['%Predicted_Var_vs ROP 7days']=np.where(final['Reorder_Point_Hl']==0,0,final['%Predicted_Var_vs ROP 7days'])
    final['%Predicted_Var_vs ROP 7days']=final['%Predicted_Var_vs ROP 7days'].astype(float).map("{:.2%}".format)
    final['%Variance_vs_ROP']=final['%Variance_vs_ROP'].astype(float).map("{:.2%}".format)
    final['National Average']=final['National Average'].astype(float).map("{:.2%}".format)
    final['SKU']=final['SKU'].astype(str)
    final['Site_Code']=final['Site_Code'].astype(str)
    final['Site Name']=final['Site_Code'].map(depot_code_dict)
    
    final['SKU Name']=final['SKU'].map(sku_name_dict)
    
    final_file=final[['Site_Code','Site Name','SKU','SKU Name','Stock on floor','Planned In Day1','Planned In Day2','Planned In Day3','Planned In Day4','Planned In Day5','Planned In Day6','Planned In Day7','In Transit Day1','In Transit Day2','In Transit Day3','In Transit Day4','In Transit Day5','In Transit Day6','In Transit Day7','Production Day1','Production Day2','Production Day3','Production Day4','Production Day5','Production Day6','Production Day7','Total 7Days Production','Total Planned Load 7Days','Customer_Orders','Daily Demand Day1','Daily Demand Day2','Daily Demand Day3','Daily Demand Day4','Daily Demand Day5','Daily Demand Day6','Daily Demand Day7','Total 7Days Demand','Stock on floor 7Days','Reorder_Point_Hl','%Predicted_Var_vs ROP 7days','National Average','Volume available for Redeployment/Volume required to reach Nat Avg']]
    round_cols=['Stock on floor','Planned In Day1','Planned In Day2','Planned In Day3','Planned In Day4','Planned In Day5','Planned In Day6','Planned In Day7','In Transit Day1','In Transit Day2','In Transit Day3','In Transit Day4','In Transit Day5','In Transit Day6','In Transit Day7','Total Planned Load 7Days','Customer_Orders','Daily Demand Day1','Daily Demand Day2','Daily Demand Day3','Daily Demand Day4','Daily Demand Day5','Daily Demand Day6','Daily Demand Day7','Total 7Days Demand','Production Day1','Production Day2','Production Day3','Production Day4','Production Day5','Production Day6','Production Day7','Total 7Days Production','Stock on floor 7Days','Volume available for Redeployment/Volume required to reach Nat Avg']
#     final_file[round_cols]=final_file[round_cols].round(3)
    final_file[round_cols]=final_file[round_cols]
    final_file.rename(columns={'Reorder_Point_Hl':'Reorder Point (Hl)'},inplace=True)
    ##CHECK##
#     final_file['Vol available for Redeployment wrt ROP']=np.where(final_file['Volume available for Redeployment/Volume required to reach Nat Avg']<0,np.where((final_file['Stock on floor 7Days']+final_file['Volume available for Redeployment/Volume required to reach Nat Avg'])>=final_file['Reorder Point (Hl)'],final_file['Volume available for Redeployment/Volume required to reach Nat Avg'],np.where((final_file['Stock on floor 7Days']-final_file['Reorder Point (Hl)'])>0,final_file['Stock on floor 7Days']-final_file['Reorder Point (Hl)'],0)),0)
    final_file['Vol available for Redeployment wrt ROP'] = np.where(final_file['Volume available for Redeployment/Volume required to reach Nat Avg']<0,final_file['Volume available for Redeployment/Volume required to reach Nat Avg'],0)
    final_file['Vol available for Redeployment wrt ROP']=final_file['Vol available for Redeployment wrt ROP'].abs()
    final_file['Volume required to reach Nat Avg']=np.where(final_file['Volume available for Redeployment/Volume required to reach Nat Avg']>0,final_file['Volume available for Redeployment/Volume required to reach Nat Avg'],0)
    
    ##CHECK##
#     final_file['Volume required to reach Nat Avg new']=np.where(final_file['Volume available for Redeployment/Volume required to reach Nat Avg new']>0,final_file['Volume available for Redeployment/Volume required to reach Nat Avg new'],0)
#     final_file['Volume required to reach Nat Avg']=np.where(final_file['Volume required to reach Nat Avg']>final_file['Volume required to reach Nat Avg new'],final_file['Volume required to reach Nat Avg'],final_file['Volume required to reach Nat Avg new'])
#     final_file = final_file.drop(columns={'Volume required to reach Nat Avg new','Volume available for Redeployment/Volume required to reach Nat Avg new'})    
    
    return final_file

final_stock_cal_file=stock_cal(inventory_policy,planned_prod_df,planned_in_df,intransit_volume_df,sku_name_dict,depot_code_dict,Final_Dataframe_Display)



final_stock_cal_file.to_excel(input_path/'final_stock_cal_file.xlsx',index=False)


# Q1=maco_wo_vlc_df['MACOwoVLC/HL'].quantile(0.5)
# Q3=maco_wo_vlc_df['MACOwoVLC/HL'].quantile(0.9)
# Lower_Whisker = Q1
# maco_wo_vlc_df.loc[maco_wo_vlc_df['MACOwoVLC/HL'] < Lower_Whisker, 'MACOwoVLC/HL'] = Lower_Whisker
# print(Lower_Whisker)
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df = maco_wo_vlc_df.loc[maco_wo_vlc_df['MACOwoVLC/HL'] > 0, :].reset_index(drop=True)
maco_wo_vlc_df


#####################################################################################################
################### preparation of input for the solver#############################################
####################################################################################################

##### MACO file #####
maco_wo_vlc_df['Material_Number']=maco_wo_vlc_df['Material_Number'].astype(str)
maco_wo_vlc_df = maco_wo_vlc_df.groupby(['Material_Number'])[['MACOwoVLC/HL']].mean().reset_index()
# maco_wo_vlc_df = outliter_treatment(maco_wo_vlc_df)
all_skus=maco_wo_vlc_df[maco_wo_vlc_df['MACOwoVLC/HL']>0]
median_maco=all_skus['MACOwoVLC/HL'].median()
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df = uom_df[['Material_Number']].drop_duplicates().merge(maco_wo_vlc_df,on='Material_Number',how='left')
print(maco_wo_vlc_df.shape)
maco_wo_vlc_df['MACOwoVLC/HL'] = maco_wo_vlc_df['MACOwoVLC/HL'].fillna(median_maco)



maco_dict=dict(list(zip(maco_wo_vlc_df['Material_Number'],maco_wo_vlc_df['MACOwoVLC/HL'])))



### UOM file ###
uom_df['Material_Number']=uom_df['Material_Number'].astype(str)
conversion_file=uom_df[(uom_df['Base Unit']=='HL')&(uom_df['Convert Unit']=='PAL')].rename(columns={'Conversion Factor':'1HL_TO_Pal'})
conversion_file.rename(columns={'Material_Number':'SKU'},inplace=True)
conversion_file.head()


### Trip Cost ######

L = dc_mapping['Depot Code'].unique()
DC_combinations = [",".join(map(str, comb)) for comb in combinations(L, 2)]
print(len(DC_combinations))

DC_combinations = pd.DataFrame(pd.DataFrame(DC_combinations, columns=['DC_combinations'])['DC_combinations'].str.split(pat=',', n=-1, expand=False).to_list(), columns=['Site_Code_1','Site_Code_2'])
DC_combinations = DC_combinations.merge(dc_mapping[['Depot Code']].rename(columns={'Depot Code':'Site_Code_1'}),on='Site_Code_1',how='left')
DC_combinations = DC_combinations.merge(dc_mapping[['Depot Code']].rename(columns={'Depot Code':'Site_Code_2'}),on='Site_Code_2',how='left')
print(DC_combinations.shape)
DC_combinations.head()
trip_cost_df_all = DC_combinations.rename(columns={'Site_Code_1':'Site_Code_Source','Site_Code_2':'Site_Code_Destination'})
trip_cost_df_all = trip_cost_df_all.append(trip_cost_df_all.rename(columns={'Site_Code_Source':'Site_Code_Destination','Site_Code_Destination':'Site_Code_Source'})[['Site_Code_Source','Site_Code_Destination']]).reset_index(drop=True)
trip_cost_df_all['plant_site']=trip_cost_df_all['Site_Code_Source'].astype(str)+'_'+trip_cost_df_all['Site_Code_Destination'].astype(str)
trip_cost_df_all=trip_cost_df_all.drop_duplicates(keep='first')

trip_cost_df.rename(columns={'Trip Cost':'trip cost'},inplace=True)
trip_cost_df = trip_cost_df[~trip_cost_df['Site_Code_Source'].isna()]
trip_cost_df = trip_cost_df[~trip_cost_df['Site_Code_Destination'].isna()]
trip_cost_df['plant_site']=trip_cost_df['Site_Code_Source'].astype(str)+'_'+trip_cost_df['Site_Code_Destination'].astype(str)
trip_cost_df=trip_cost_df.drop_duplicates(keep='first')
print(trip_cost_df_all.shape, trip_cost_df.shape)
trip_cost_df = trip_cost_df_all.merge(trip_cost_df,how='left',on=['plant_site','Site_Code_Source','Site_Code_Destination'])
print(trip_cost_df.shape)

trip_cost_df['trip cost'] = trip_cost_df['trip cost'].fillna(trip_cost_df['trip cost'].mean())
cost_sheet_dict=dict(list(zip(trip_cost_df['plant_site'],trip_cost_df['trip cost'])))
cvf_dict_new=dict(list(zip(conversion_file['SKU'],conversion_file['1HL_TO_Pal'])))
supply=pd.DataFrame()
Final_Dataframe_Display1.rename(columns={'ATD_7Days':'Stock_On_floor_7Days'},inplace=True)
# Final_Dataframe_Display=Final_Dataframe_Display[Final_Dataframe_Display['Reorder_Point_Hl']!=0].reset_index(drop=True)
###CHECK###
Final_Dataframe_Display['Reorder_Point_Hl'] = Final_Dataframe_Display['Reorder_Point_Hl'].replace([0,0.0], np.NaN)
Final_Dataframe_Display['Reorder_Point_Hl'] = Final_Dataframe_Display['Reorder_Point_Hl'].fillna(Final_Dataframe_Display['Reorder_Point_Hl'].median())

Final_Dataframe_Display1['Reorder_Point_Hl'] = Final_Dataframe_Display1['Reorder_Point_Hl'].replace([0,0.0], np.NaN)
Final_Dataframe_Display1['Reorder_Point_Hl'] = Final_Dataframe_Display1['Reorder_Point_Hl'].fillna(Final_Dataframe_Display1['Reorder_Point_Hl'].median())


data_set=Final_Dataframe_Display1.copy(deep=True)
final_df=pd.DataFrame()
error_skus=[]

##### demand vs supply mapping for sku and depots##############
for sku in data_set['SKU'].unique():
    try:
        print(sku)
        data_set_sku=data_set[data_set['SKU']==sku].reset_index(drop=True)

        supply_points=data_set_sku[data_set_sku['HL Above/Below']<0].reset_index(drop=True)
        supply_points=supply_points[~(supply_points['Reorder_Point_Hl']>=supply_points['Stock_On_floor_7Days'])]
        supply_points['HL Above/Below']=supply_points['HL Above/Below'].abs()
        supply_points['New stock at depot']=np.where((supply_points['Stock_On_floor_7Days']-supply_points['Reorder_Point_Hl'])>=supply_points['HL Above/Below'],
                                                     supply_points['HL Above/Below'],supply_points['Stock_On_floor_7Days']-supply_points['Reorder_Point_Hl'])
        
        supply_points=supply_points.drop_duplicates(subset=['SKU', 'Site_Code', 'Stock_on_Floor'],keep='first')
        supply=pd.concat([supply,supply_points])
        demand_points=data_set_sku[data_set_sku['HL Above/Below']>0].reset_index(drop=True)
        demand_points=demand_points.drop_duplicates(subset=['SKU', 'Site_Code', 'Stock_on_Floor'],keep='first')

#         print(data_set_sku[data_set_sku['HL Above/Below']==0].reset_index(drop=True))

        final_demand_supply=demand_points.merge(supply_points[['SKU','Site_Code','HL Above/Below','New stock at depot']],right_on='SKU',left_on='SKU',how='left')
        final_demand_supply.rename(columns={'Site_Code_x':'Plant','Site_Code_y':'Site_Code','HL Above/Below_x':'ROP Hl at Plant','New stock at depot':'ATD'},inplace=True)
        final_demand_supply['ATD']=final_demand_supply['ATD'].abs()
        final_demand_supply['MACOwoVLC']=final_demand_supply['SKU'].map(maco_dict)
#         final_demand_supply['Site_Code']=final_demand_supply['Site_Code'].astype(str)
#         final_demand_supply['Plant']=final_demand_supply['Plant'].astype(str)
        final_demand_supply['SKU']=final_demand_supply['SKU'].astype(str)
        final_demand_supply['1HL_TO_Pal']=final_demand_supply['SKU'].map(cvf_dict_new)
        final_demand_supply['plant_site']=final_demand_supply['Site_Code'].astype(str)+'_'+final_demand_supply['Plant'].astype(str)
        final_demand_supply['Cost']=final_demand_supply['plant_site'].map(cost_sheet_dict)
        final_demand_supply['TripCost_perPallet24']=final_demand_supply['Cost']/24
        final_demand_supply=final_demand_supply[~final_demand_supply['Cost'].isna()]
        final_df=pd.concat([final_demand_supply,final_df]).reset_index(drop=True)
    except:
        print('Error sku :',sku)
        error_skus.append(sku)

        
        
        
#Supply check---> Estimating Value w/o transportation Cost
supply_check= final_df.groupby(['Site_Code','SKU','ATD'], as_index=False).agg({'ROP Hl at Plant':'sum'})
print(supply_check.shape)
supply_check= supply_check.drop_duplicates(['Site_Code','SKU'])
print(supply_check.shape)

supply_check['VOL_REDPLY']= np.where(supply_check['ATD']<supply_check['ROP Hl at Plant'],supply_check['ATD'],
                                     supply_check['ROP Hl at Plant'])

supply_check= supply_check.merge(maco_wo_vlc_df.rename(columns={'Material_Number':'SKU'}),
                                on=['SKU'], how='left')
print(supply_check.shape)
print(supply_check.VOL_REDPLY.sum())
supply_check['EST_VALUE']= supply_check['VOL_REDPLY']*supply_check['MACOwoVLC/HL']
print(supply_check.EST_VALUE.sum())
print(supply_check.EST_VALUE.sum()*52)
supply_check.head()


### CHECK ###
###### to filter out for certain depots####################
##### filter from source points with full stock capacity#####
### remove from destination points because they have full stock capacity#######
######### to filter for bulk skus( give skus as string)############################################

model_input=final_df.copy(deep=True)

######################### Adding MACO and converting HL to pallets to be fed in solver#################

main_file=model_input.copy(deep=True)
# obsolescence=obsolete_tag_final.copy(deep=True)

conversion_file=uom_df[(uom_df['Base Unit']=='KG')&(uom_df['Convert Unit']=='CS')].rename(columns={'Conversion Factor':'1KG to CS'})
conversion_file_2=uom_df[(uom_df['Base Unit']=='CS')&(uom_df['Convert Unit']=='PAL')].rename(columns={'Conversion Factor':'1CS to PAL'})
conversion_file_2['1PAL to CS'] = 1/conversion_file_2['1CS to PAL']
conversion_file = conversion_file.merge(conversion_file_2[['Material_Number','1PAL to CS']],on='Material_Number',how='left')
conversion_file['1Pal to KG'] = conversion_file['1PAL to CS']/conversion_file['1KG to CS']
# conversion_file.loc[conversion_file['1Pal to KG']==np.inf,'1Pal to KG']= conversion_file['1Pal to KG'].median()




weights_dict=dict(list(zip(conversion_file['Material_Number'],conversion_file['1Pal to KG'])))
print(weights_dict)
conversion_file_CS=uom_df[(uom_df['Base Unit']=='HL')&(uom_df['Convert Unit']=='CS')].rename(columns={'Conversion Factor':'HL to CS'})
cs_dict=dict(list(zip(conversion_file_CS['Material_Number'],conversion_file_CS['HL to CS'])))
print(cs_dict)
main_file['MACOwoVLC']=main_file['MACOwoVLC'].fillna(median_maco)
# obsolete_stock=obs_final_v2.copy(deep=True)
# obsolete_stock=obsolete_stock[obsolete_stock['Site_Code'].notnull()]
# obsolete_stock=obsolete_stock[obsolete_stock['Material_Number'].notnull()]
# obsolete_stock['UID']=obsolete_stock['Site_Code'].astype(str)+"_"+obsolete_stock['Material_Number'].astype(str)

## Incorporating OOS###########
oos_grouped=oos_df.groupby(['Site_Code', 'Material_Number']).agg({'OOS' : 'sum'}).reset_index()
oos_grouped['OOS']=oos_grouped['OOS'].fillna(0)
# oos_grouped['OOS']=oos_grouped['OOS']/oos_grouped['Date']
oos_grouped['Dest ID']=oos_grouped['Site_Code']
oos_grouped['Material_Number']=oos_grouped['Material_Number'].astype(str)
oos_grouped['SKU']=oos_grouped['Material_Number']
oos_grouped['UID']=oos_grouped['Dest ID'].astype(str)+"_"+oos_grouped['SKU'].astype(str)
oos_dict=dict(list(zip(oos_grouped['UID'],oos_grouped['OOS'])))

##################################### expiry date mapping based on FIFO #######################
# obsolescence['Depot Name'] = obsolescence['Site_Code'].map(depot_code_dict)
# obsolescence['UID']=obsolescence['Depot Name'].astype(str)+"_"+obsolescence['Material_Number'].astype(str)
# obsolescence=obsolescence[obsolescence['In Risk']=='Sí']




#Checking potential OOS value based on OOS
print(oos_grouped.shape)
print(oos_grouped['OOS'].sum())
print(oos_grouped.shape)
oos_value= oos_grouped.merge(maco_wo_vlc_df, on=['Material_Number'], how='left')
print(oos_value.shape)
oos_value['OOS_VALUE']= oos_value['OOS']*oos_value['MACOwoVLC/HL']
print(oos_value.OOS_VALUE.sum())
print(oos_value.OOS_VALUE.sum()*52)
oos_value.head()



################ Adding weight to the MACO based on OOS and Obsolescence ####################################################################

main_file['UID obs']=main_file['Site_Code'].astype(str)+"_"+main_file['SKU'].astype(str)
main_file['UID oos']=main_file['Plant'].astype(str)+"_"+main_file['SKU'].astype(str)
# obs_dict=dict(list(zip(obsolete_stock['UID'],obsolete_stock['Calculated Obs'])))
# obs_dict_USD=dict(list(zip(obsolete_stock['UID'],obsolete_stock['Calculated Obs (USD)'])))
# obs_dict



# main_file['obs vol']=main_file['UID obs'].map(obs_dict)
# main_file['obs vol']=main_file['obs vol'].fillna(0)

# main_file['Obs (USD)']=main_file['UID obs'].map(obs_dict_USD)
# main_file['Obs (USD)']=main_file['Obs (USD)'].fillna(0)

main_file['oos vol']=main_file['UID oos'].map(oos_dict)
main_file['oos vol']=main_file['oos vol'].fillna(0)

# main_file['final obs vol']=np.where((main_file['ATD']>=main_file['obs vol'])&(main_file['obs vol']!=0),main_file['obs vol'],np.where(main_file['obs vol']==0,0,main_file['ATD']))
# main_file['final obs vol']=main_file['final obs vol'].fillna(0)
main_file['final oos vol']=np.where((main_file['ROP Hl at Plant']>=main_file['oos vol'])&(main_file['oos vol']!=0),main_file['oos vol'],
                                    np.where(main_file['oos vol']==0,0,main_file['ROP Hl at Plant']))
main_file['final oos vol']=main_file['final oos vol'].fillna(0)

main_file['actual MACO']=main_file['MACOwoVLC']
# main_file['obs %']=main_file['final obs vol']/main_file['ATD']
main_file['OOS %']=main_file['final oos vol']/main_file['ROP Hl at Plant']
main_file



## CHECK ##
# main_file['plant_site_destruction'] = main_file['plant_site'].str[:4] + '_' +main_file['plant_site'].str[:2] + '00' 
# main_file['trip_cost_destruction'] = trip_cost_df['trip cost'].mean() #main_file['plant_site_destruction'].map(cost_sheet_dict)
# main_file['trip_cost_destruction'] = main_file['trip_cost_destruction'].fillna(0)
# main_file['trip_cost_destruction per hl'] = ((main_file['obs %']*main_file['trip_cost_destruction'])/main_file['final obs vol']).replace([-np.inf,np.inf,np.NaN],0)

# main_file['cost_destruction'] = 5*main_file['final obs vol']*main_file['SKU'].astype(str).map(cs_dict)
# main_file['cost_destruction per hl'] = ((main_file['obs %']*main_file['cost_destruction'])/main_file['final obs vol']).replace([-np.inf,np.inf,np.NaN],0)
# main_file['cost of goods per hl'] = ((main_file['obs %']*main_file['Obs (USD)'])/main_file['final obs vol']).replace([-np.inf,np.inf,np.NaN],0)

# main_file['Obs MACO'] = (main_file['obs %']*main_file['actual MACO'])
main_file['OOS MACO'] = (main_file['OOS %']*main_file['actual MACO'])


# main_file['MACOwoVLC']= main_file['trip_cost_destruction per hl'] + main_file['cost_destruction per hl'] + main_file['cost of goods per hl'] + main_file['Obs MACO'] + main_file['OOS MACO']#+main_file['actual MACO']
# main_file['obs weights MACOwoVLC']=main_file['trip_cost_destruction per hl'] + main_file['cost_destruction per hl'] + main_file['cost of goods per hl'] + main_file['Obs MACO']#+main_file['actual MACO']


main_file['MACOwoVLC']= main_file['OOS MACO']#+main_file['actual MACO']
# main_file['obs weights MACOwoVLC']=main_file['trip_cost_destruction per hl'] + main_file['cost_destruction per hl'] + main_file['cost of goods per hl'] + main_file['Obs MACO']#+main_file['actual MACO']





###################### only keeping obsolescence sources-sku in input file###############


#main_file=main_file[main_file['final obs vol']!=0].reset_index(drop=True)


###################### only keeping OOS destination-skus in input file###############


#main_file=main_file[main_file['final oos vol']!=0].reset_index(drop=True)


###################### only keeping obsolescence sources-skus and oos destinations-skus in input file###############

#main_file=main_file[(main_file['final obs vol']!=0)|(main_file['final oos vol']!=0)].reset_index(drop=True)


#####################################################################################



main_file=main_file[main_file['1HL_TO_Pal'].notna()]
main_file.to_csv(output_path/'input to solver.csv',index=False)
one_truck_input=main_file.copy(deep=True)

# final_obs_vol_dict=dict(list(zip(main_file['UID obs'],main_file['obs vol'])))
Final_Dataframe_Display1['UID']=Final_Dataframe_Display1['Site_Code'].astype(str)+"_"+Final_Dataframe_Display1['SKU'].astype(str)

depot_mapping=dc_mapping.copy()
depot_dict=dict(list(zip(depot_mapping['Depot Description'],depot_mapping['Depot Code'])))
depot_code_dict=dict(list(zip(depot_mapping['Depot Code'],depot_mapping['Depot Description']))) 

################################ input sheet formatted for the output file ############################
solver_input=main_file.copy(deep=True)
solver_input=solver_input[['SKU', 'Site_Code','ATD','Plant','ROP Hl at Plant','1HL_TO_Pal','Cost','MACOwoVLC','actual MACO','oos vol','final oos vol','OOS %','OOS MACO']]
solver_input.rename(columns={'Site_Code':'Source','ATD':'Vol available for redeployment','Plant':'Destination','ROP Hl at Plant':'Vol required to reach National Average','Cost':'Total Trip Cost(USD)','MACOwoVLC':'Weighed MACOwoVLC','actual MACO':'MACO wo VLC','oos vol':'OOS(HL)','final oos vol':'OOS capped(HL)'},inplace=True)
solver_input['SKU']=solver_input['SKU'].astype(str)
solver_input['SKU Name']=solver_input['SKU'].map(sku_name_dict)
solver_input['Source Name']=solver_input['Source'].map(depot_code_dict)
solver_input['Destination Name']=solver_input['Destination'].map(depot_code_dict)
final_solver_input=solver_input[['SKU','SKU Name','Source','Source Name','Vol available for redeployment','Destination','Destination Name','Vol required to reach National Average','Total Trip Cost(USD)','MACO wo VLC','Weighed MACOwoVLC','OOS(HL)','OOS capped(HL)','OOS %','OOS MACO']]

input_cols=['Vol available for redeployment','Vol required to reach National Average','Total Trip Cost(USD)','MACO wo VLC','Weighed MACOwoVLC','OOS(HL)','OOS capped(HL)','OOS MACO']
final_solver_input[input_cols]=final_solver_input[input_cols]
# final_solver_input[input_cols]=final_solver_input[input_cols].round(2)

# final_solver_input['obs %']=final_solver_input['obs %'].astype(float).map("{:.2%}".format)
final_solver_input['OOS %']=final_solver_input['OOS %'].astype(float).map("{:.4%}".format)

######################### defining global variables to be used in both one truck and solver codes#####################
    
global stock_on_floor,rop,variance,plan_load,total_demand,seven_days,var_vs_rop

stock_on_floor=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['Stock_on_Floor'])))
rop=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['Reorder_Point_Hl'])))
variance=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['%Variance_vs_ROP'])))
plan_load=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['Planned_Loads_7days'])))
total_demand=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['Demand_7days'])))
seven_days=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['Stock_On_floor_7Days'])))
var_vs_rop=dict(list(zip(Final_Dataframe_Display1['UID'],Final_Dataframe_Display1['%Predicted_Var_vs ROP 7days'])))



# final_solver_input = final_solver_input[final_solver_input['Total Trip Cost(USD)']!=0]
#CHECK
final_solver_input['Total Trip Cost(USD)'] = final_solver_input['Total Trip Cost(USD)'].replace([0,0.0], np.NaN)
final_solver_input['Total Trip Cost(USD)'] = final_solver_input['Total Trip Cost(USD)'].fillna(final_solver_input['Total Trip Cost(USD)'].mean())



print(main_file.shape)
main_file = main_file.drop_duplicates(subset=['SKU','plant_site'])
print(main_file.shape)


####################################################################################
############################### code for solver output##############################
####################################################################################

# def solver(main_file,output_path,obsolete_tag_final,depot_code_dict,sku_name_dict,final_solver_input,oos_dict,obs_data2,final_stock_cal_file,obs_final_v1,cvf_dict_new,uom_df,usd_to_pesos,trip_cost_df_outside_region):

main_file['Site_name']=main_file['Site_Code'].map(depot_code_dict)
main_file['Plant_name']=main_file['Plant'].map(depot_code_dict)
main_file['SID']=main_file['Site_name'].astype(str)+"_"+main_file['SKU'].astype(str)
main_file['SRID']=main_file['Site_name'].astype(str)+"_"+main_file['SKU'].astype(str)+"_"+main_file['Plant_name'].astype(str)
# weighed_maco_dict=dict(list(zip(main_file['SID'],main_file['obs weights MACOwoVLC'])))
sum_weights_dict=dict(list(zip(main_file['SRID'],main_file['MACOwoVLC'])))

##CHECK##
# trip_cost_destruction_per_hl_dict=dict(list(zip(main_file['UID obs'],main_file['trip_cost_destruction per hl'])))
# cost_destruction_per_hl_dict=dict(list(zip(main_file['UID obs'],main_file['cost_destruction per hl'])))
# cost_of_goods_per_hl_dict=dict(list(zip(main_file['UID obs'],main_file['cost of goods per hl'])))
# obs_maco_dict=dict(list(zip(main_file['UID obs'],main_file['Obs MACO'])))
oos_maco_dict=dict(list(zip(main_file['UID oos'],main_file['OOS MACO'])))

conversion_file=uom_df[(uom_df['Base Unit']=='KG')&(uom_df['Convert Unit']=='CS')].rename(columns={'Conversion Factor':'1KG to CS'})
conversion_file_2=uom_df[(uom_df['Base Unit']=='CS')&(uom_df['Convert Unit']=='PAL')].rename(columns={'Conversion Factor':'1CS to PAL'})
conversion_file_2['1PAL to CS'] = 1/conversion_file_2['1CS to PAL']
conversion_file = conversion_file.merge(conversion_file_2[['Material_Number','1PAL to CS']],on='Material_Number',how='left')
conversion_file['1Pal to KG'] = conversion_file['1PAL to CS']/conversion_file['1KG to CS']
weights_dict=dict(list(zip(conversion_file['Material_Number'],conversion_file['1Pal to KG'])))

org_maco_dict=dict(list(zip(main_file['SID'],main_file['actual MACO'])))

maco_file=main_file[['Site_Code','Plant','SKU','MACOwoVLC','1HL_TO_Pal']]
maco_file.rename(columns={'Site_Code':'Source','Plant':'Destination','MACOwoVLC':'MACO','1HL_TO_Pal':'Conversion factor'},inplace=True)
maco_file.drop(maco_file[maco_file['MACO'].isnull()].index,inplace=True)




####################### Creation of Supply file ####################

supply_file=main_file[['Site_Code','SKU','ATD','1HL_TO_Pal']]
supply_file=supply_file.drop_duplicates(subset=None,keep='first')
supply_file.rename(columns={'Site_Code':'Source','ATD':'Excess supply','1HL_TO_Pal':'Conversion factor'},inplace=True)
supply_file['Supply check']=(supply_file['Conversion factor']*supply_file['Excess supply']).apply(np.floor)
supply_file['Excess supply']=supply_file['Supply check']


################## Creation of Demand file ###################

demand_file=main_file[['Plant','SKU','ROP Hl at Plant','1HL_TO_Pal']]
demand_file=demand_file.drop_duplicates(subset=None,keep='first')
demand_file.rename(columns={'Plant':'Destination','ROP Hl at Plant':'Demand','1HL_TO_Pal':'Conversion factor'},inplace=True)
demand_file['Demand check']=(demand_file['Conversion factor']*demand_file['Demand']).apply(np.floor)
demand_file['Demand']=demand_file['Demand check']


########### Column type conversions###########

maco_file['Source']=maco_file['Source'].astype(str)
maco_file['Destination']=maco_file['Destination'].astype(str)
maco_file['SKU']=maco_file['SKU'].astype(str)
supply_file['Source']=supply_file['Source'].astype(str)
supply_file['SKU']=supply_file['SKU'].astype(str)
demand_file['Destination']=demand_file['Destination'].astype(str)
demand_file['SKU']=demand_file['SKU'].astype(str)


#### Problem Formulation ##
problem = LpProblem('Benefit vs Cost', LpMaximize)


########### Creating source destination data points and SKU list#######################

maco_file['main_points']=maco_file['Source']+"_"+maco_file['Destination']+"__"+maco_file['SKU'].astype(str)
SKU_list=list(set(maco_file['SKU']))

#### defining quantity decision variables###########

sr_dest_dict=dict(list(zip(maco_file['SKU'],maco_file['main_points'])))

for sku in SKU_list:
    points=list(set(maco_file[maco_file['SKU']==sku]['main_points']))
    sr_dest_dict[sku]=points

benefits1=dict()
for sku in SKU_list:
    benefits=LpVariable.dicts("qty",([sku],sr_dest_dict[sku]),0,cat='Integer')
    benefits1.update(benefits)
    
    
############### setting demand constraints ####################################################

sku_dict_demand=dict()
for sku in SKU_list:
    sku_dict_demand[sku]={}
    destination=list(set(maco_file[maco_file['SKU']==sku]['Destination']))
    for d in destination:
        sku_dict_demand[sku][d]=list(set((maco_file[(maco_file['SKU']==sku)&(maco_file['Destination']==d)]['main_points'])))

demand_dict=dict()
for demand in demand_file['SKU'].unique():
    dd=demand_file[demand_file['SKU']==demand]
    dd1=dd.groupby(['Destination']).sum().reset_index()
    demand_dict[demand]={}
    for dest in dd1['Destination'].unique():
        final_demand=dd1[dd1['Destination']==dest]['Demand'].unique()[0]
        demand_dict[demand][dest]=final_demand

for sku in SKU_list:
    for plant in sku_dict_demand[sku].keys():
        problem+=lpSum(benefits1[sku][j] for j in sku_dict_demand[sku][plant])<=demand_dict[sku][plant]

        
        
########### Setting Supply constraint#######################################################
sku_dict_supply=dict()        
for sku in SKU_list:
    sku_dict_supply[sku]={}
    sources=list(set(maco_file[maco_file['SKU']==sku]['Source']))
    for s in sources:
        sku_dict_supply[sku][s]=list(set((maco_file[(maco_file['SKU']==sku)&(maco_file['Source']==s)]['main_points'])))


supply_dict=dict()
for supply in supply_file['SKU'].unique():
    dds=supply_file[supply_file['SKU']==supply]
    dds1=dds.groupby(['Source']).sum().reset_index()
    supply_dict[supply]={}
    for sr in dds1['Source'].unique():
        final_supply=dds1[dds1['Source']==sr]['Excess supply'].unique()[0]
        supply_dict[supply][sr]=final_supply

existing_list=[]
for sku in SKU_list:
    for source in sku_dict_supply[sku].keys():
        list2=sku_dict_supply[sku][source]
        if list2 in existing_list:
            pass
        else:
            for elements in list2:

                existing_list.append(list2)
                problem+=lpSum(benefits1[sku][i] for i in sku_dict_supply[sku][source])<=supply_dict[sku][source]
                
                
########### Cost file creation and column conversions#######

cost_file=main_file[['Site_Code','Plant','TripCost_perPallet24']]
cost_file.rename(columns={'Site_Code':'Source','Plant':'Destination'},inplace=True)
cost_file=cost_file.drop_duplicates(subset=['Source','Destination'],keep='first').reset_index(drop=True)
cost_file['Cost']=cost_file['TripCost_perPallet24']*24
cost_file['Trucks']='Y'+cost_file.index.astype(str)
cost_file['Source']=cost_file['Source'].astype(str)
cost_file['Destination']=cost_file['Destination'].astype(str)
cost_file['main_points']=cost_file['Source'] +'_'+cost_file['Destination']

#     cost_file = cost_file[cost_file['Cost']!=0]
###CHECK
cost_file['Cost'] = cost_file['Cost'].replace([0,0.0], np.NaN)
cost_file['Cost'] = cost_file['Cost'].fillna(cost_file['Cost'].mean())

maco_file['depot_combos']=maco_file['Source']+'_'+maco_file['Destination']
depot_combos=list(set(maco_file['depot_combos']))

trucks=list(set(cost_file['Trucks']))
truck_dict=dict()
for points in cost_file['main_points'].unique():
    filtered_truck=[cost_file[cost_file['main_points']==points]['Trucks'].unique()[0]]
    truck_dict[points]=filtered_truck



################# defining truck decision variable and max 10 truck constraint#########################
final_truck_dict=dict()
for combos in depot_combos:
#         try:
    comb=[combos]
    cost_benefits=LpVariable.dicts("truck",(comb,truck_dict[combos]), lowBound=0, upBound=2,cat='Integer')           
    final_truck_dict.update(cost_benefits)
#         except:
#             print('Error')
new_truck_dict=dict()
for combos in depot_combos:
    truck_number=final_truck_dict[combos]
    for keys in truck_number.keys():
        variable=[truck_number[keys]]
        new_truck_dict[combos]=variable

print('creating truck constraints..........')


################## max 10 truck constraint###############

problem+=lpSum(new_truck_dict[combos][0] for combos in depot_combos )<=10



new_dict1=dict()
mm=dict()
aa=dict()
for combos in depot_combos:
    df4=maco_file[maco_file['depot_combos']==combos]
    sku_list=list(set(df4['SKU']))
    for s in sku_list:
        new_dict=dict()
        new_dict=benefits1[s]
        for k in new_dict.keys():
            mm=k.split("__")[0]
            if mm in new_dict1.keys():               
                 aa[mm].append(new_dict[k])
            else:
                aa[mm]= [(new_dict[k])]
                new_dict1.update(aa)

final_qty_dict=dict()
for m in new_dict1.keys():
    abc=set(new_dict1[m])
    final_qty_dict[m]=list(abc)


new_sku_dict1=dict()
sku_combo_dict=dict()
for combos in depot_combos:
    df5=maco_file[maco_file['depot_combos']==combos]
    sku_list=list(set(df5['SKU']))
    for sk in sku_list:

        new_dict=dict()
        new_dict=benefits1[sk]
        for k in new_dict.keys():
            mm=k.split("__")[0]

            if mm in  new_sku_dict1.keys():

                if sk not in  new_sku_dict1[mm].keys():

                    sku_combo_dict[mm][sk]=[(new_dict[k])]

            else:

                sku_combo_dict[mm]={}
                sku_combo_dict[mm][sk]= [(new_dict[k])]
                new_sku_dict1.update(sku_combo_dict)    


#################################### Conversion factors dictionary from pallets to HL##############

cvf_dict=dict()
cv_file=main_file[['SKU','1HL_TO_Pal']]
cv_file=cv_file.drop_duplicates(subset=None,keep='first').reset_index(drop=True)
cv_file['SKU']=cv_file['SKU'].astype(str)
for skus in benefits1.keys():
    cvf=cv_file[cv_file['SKU']==skus]['1HL_TO_Pal'].unique()[0]
    routes_dict=benefits1[skus]
    for routes in routes_dict.keys():
        rt=routes_dict[routes]
        cvf_dict[rt]=cvf

################################### Creating Truck Constraint of max and min pallets per truck ###############

for combos in depot_combos:
#         try:
    truck_variable=truck_dict[combos][0]
    problem+=lpSum(final_qty_dict[combos])<=final_truck_dict[combos][truck_variable]*24
#         except:
#             print('CHECK ERROR')

for combos in depot_combos:
#         try:
    truck_variable=truck_dict[combos][0]
    problem+=lpSum(final_qty_dict[combos])>=final_truck_dict[combos][truck_variable]*6
#         except:
#             print('CHECK ERROR')
for combos in depot_combos:
#         try:
    sku_key_list=list(new_sku_dict1[combos].keys())
    truck_variable=truck_dict[combos][0]
    problem+=lpSum(new_sku_dict1[combos][sku][0]*weights_dict[sku] for sku in sku_key_list)<=final_truck_dict[combos][truck_variable]*27215
#         except:
#             print('CHECK ERROR: Weights are missing in UOM file')
#             continue

per_truck_cost=dict()
for dc in cost_file['main_points'].unique():
    df_source_t=cost_file[cost_file['main_points']==dc]
    t_dict=dict(list(zip(df_source_t['Trucks'],df_source_t['Cost'])))
    per_truck_cost[dc]=t_dict    

####################################################################################################################                
##### Objective function ################


cvf=dict()
for skus in cvf_dict_new.keys():
    value=1/cvf_dict_new[skus]
    cvf[skus]=value    


MACO_dict=dict()
for sku in maco_file['SKU'].unique():
    df_source=maco_file[maco_file['SKU']==sku]
    dest_dict=dict(list(zip(df_source['main_points'],df_source['MACO'])))
    MACO_dict[sku]=dest_dict    


depots=[(i,j) for i in SKU_list for j in benefits1[i].keys()]
#     try:
maco_func=lpSum(benefits1[i][j]*MACO_dict[i][j]*cvf[i] for (i,j) in depots)
#     except:
#         print('CHECK ERROR')

#     try:
dc_across=[(x,y) for x in depot_combos for y in final_truck_dict[x].keys()]
#     except:
#         print('CHECK ERROR')

#     try:    
truck_func=lpSum(((final_truck_dict[x][y]*per_truck_cost[x][y]) for (x,y) in dc_across))
#     except:
#         print('CHECK ERROR')

problem+=maco_func-truck_func
print('solving.........')

time.sleep( 200 )
problem.solve(PULP_CBC_CMD(maxSeconds = 1500, threads = None))
print(LpStatus[problem.status])    




for skus in benefits1.keys():
    cvf=cv_file[cv_file['SKU']==skus]['1HL_TO_Pal'].unique()[0]
    routes_dict=benefits1[skus]
    for routes in routes_dict.keys():
        rt=routes_dict[routes]
        cvf_dict[rt]=cvf
        
        

sr_dest_dict=dict(list(zip(maco_file['SKU'],maco_file['main_points'])))

for sku in SKU_list:
    points=list(set(maco_file[maco_file['SKU']==sku]['main_points']))
    sr_dest_dict[sku]=points

benefits1=dict()
for sku in SKU_list:
    benefits=LpVariable.dicts("qty",([sku],sr_dest_dict[sku]),0,cat='Integer')
    benefits1.update(benefits)

    
benefits1




####################################### Output and output format ########################

final_output=pd.DataFrame()
final_trucks=pd.DataFrame()
for v in problem.variables():
    
    print(v, v.varValue)
    if v.varValue>0:
        print(v.varValue)
        aa=str(v)

        if 'qty' in aa:
#             print(aa)
            sp=aa.split("_")[2]
            dp=aa.split("_")[3]
            print(aa.split("_")[1])
            skus=str(aa.split("_")[1])
            data = [[sp,dp,skus,v.varValue]] 

            ot1 = pd.DataFrame(data, columns = ['Source', 'Destination','SKU','QTY'])
            final_output=pd.concat([ot1,final_output]).reset_index(drop=True)
        else:
            spt=aa.split("_")[1]
            dpt=aa.split("_")[2]
            datat = [[spt,dpt,v.varValue]]
            ot2 = pd.DataFrame(datat, columns = ['Source', 'Destination','No of trucks'])
            final_trucks=pd.concat([ot2,final_trucks]).reset_index(drop=True)

            
final_output['SKU']=final_output['SKU'].astype(str)
final_output['1HL_1Pl']=final_output['SKU'].map(cvf_dict_new)
final_output['1Pl_1HL']=1/final_output['1HL_1Pl']
final_output['Qty in HL']=final_output['QTY']*final_output['1Pl_1HL']
main_file['SKU']=main_file['SKU'].astype(str)
maco_dict=dict(list(zip(main_file['SKU'],main_file['actual MACO'])))   
final_output['MACO per HL']=final_output['SKU'].map(maco_dict)
final_output['Benefit(in USD)']=final_output['Qty in HL']*final_output['MACO per HL']

final_output['Benefit($)']=final_output['Benefit(in USD)']#*(1/usd_to_pesos)
#     final_output['Source']=final_output['Source'].astype(int)
final_output['Source Name']=final_output['Source'].map(depot_code_dict)
final_output['Source']=final_output['Source'].astype(str)

#     final_output['Destination']=final_output['Destination'].astype(int)
final_output['Destination Name']=final_output['Destination'].map(depot_code_dict)
final_output['Destination']=final_output['Destination'].astype(str)

#     SKU_file=pd.read_excel(static_path+'SKU_Mapping.xlsx')
#     SKU_file['SKU Code']=SKU_file['SKU Code'].astype(int)
#     SKU_file['SKU Code']=SKU_file['SKU Code'].astype(str)
#     sku_name_dict=dict(list(zip(SKU_file['SKU Code'],SKU_file['SKU Description'])))

final_output['SKU Name']=final_output['SKU'].map(sku_name_dict)

del final_output['1HL_1Pl'],final_output['1Pl_1HL']
final_output=final_output[['Source','Source Name','Destination','Destination Name','SKU','SKU Name','QTY', 'Qty in HL', 'MACO per HL',
       'Benefit(in USD)']]

print(17)
final_trucks['UID']=final_trucks['Source']+"_"+final_trucks['Destination']
cd=dict(list(zip(cost_file['main_points'],cost_file['Cost'])))
final_trucks['Cost per Truck']=final_trucks['UID'].map(cd)
final_trucks['Total TripCost( in USD)']=final_trucks['Cost per Truck']*final_trucks['No of trucks']
final_trucks['Total TripCost($)']=final_trucks['Total TripCost( in USD)']#*(1/usd_to_pesos)
del final_trucks['UID']
#     final_trucks['Source']=final_trucks['Source'].astype(int)
final_trucks['Source Name']=final_trucks['Source'].map(depot_code_dict)
final_trucks['Source']=final_trucks['Source'].astype(str)

#     final_trucks['Destination']=final_trucks['Destination'].astype(int)
final_trucks['Destination Name']=final_trucks['Destination'].map(depot_code_dict)
final_trucks['Destination']=final_trucks['Destination'].astype(str)

final_trucks=final_trucks[['Source','Source Name','Destination','Destination Name','No of trucks', 'Cost per Truck',
       'Total TripCost( in USD)']]


# obs_check=main_file[['Site_Code','SKU','final obs vol']]
# obs_check=obs_check.drop_duplicates(subset=None,keep='first')
# obs_df=obs_check.groupby(['Site_Code','SKU'])['final obs vol'].sum().reset_index()

output_df=final_output[['Source','SKU','Qty in HL']]
output_df_grouped=output_df.groupby(['Source','SKU'])['Qty in HL'].sum().reset_index()

#     cols=['Source','SKU']
cols=['SKU']
# output_df_grouped[cols]=output_df_grouped[cols].astype(int)
output_df_grouped[cols]=output_df_grouped[cols].astype(str)

#     cols1=['Site_Code','SKU']
cols1=['SKU']
# obs_df[cols1]=obs_df[cols1].astype(int)
# obs_df[cols1]=obs_df[cols1].astype(str)


print(18)
# obs_df2=obs_df.merge(output_df_grouped,right_on=['Source','SKU'],left_on=['Site_Code','SKU'],how='left').reset_index(drop=True)
# obs_df2['Qty in HL']=obs_df2['Qty in HL'].fillna(0)
# obs_df2['final obs vol']=obs_df2['final obs vol'].fillna(0)
# obs_df2['Transferred Obsolescense(HL)']=np.where(obs_df2['Qty in HL']>obs_df2['final obs vol'],obs_df2['final obs vol'],obs_df2['Qty in HL'])


# main_file_obs_sorted=main_file_obs.sort_values(by='Days to expiry',ascending=True)
# main_file_obs_sorted1=main_file_obs_sorted[['Site_Code','Material_Number','Days to expiry']]

# main_file_obs_sorted1=main_file_obs_sorted1.drop_duplicates(subset=['Site_Code','Material_Number'],keep='first')
# main_file_obs_sorted1['Source']=main_file_obs_sorted1['Site_Code']#.map(depot_dict_final)
# main_file_obs_sorted1['Material']=main_file_obs_sorted1['Material_Number'].astype(str)
# main_file_obs_sorted1['Material']=main_file_obs_sorted1['Material'].astype(str)
# obs_df2_final=obs_df2.merge(main_file_obs_sorted1[['Source','Material','Days to expiry']],left_on=['Site_Code','SKU'],right_on=['Source','Material'],how='left')
# obs_df2_final1=obs_df2_final[['Site_Code','Material','Transferred Obsolescense(HL)','Days to expiry']]
# obs_df2_final1=obs_df2_final1[obs_df2_final1['Transferred Obsolescense(HL)']!=0]
# obs_df2_final1.rename(columns={'Site_Code':'Source','Material':'SKU'},inplace=True)
# final_output1=final_output.merge(obs_df2_final1[['Source','SKU','Days to expiry']],left_on=['Source','SKU'],right_on=['Source','SKU'],how='left')



final_output1= final_output.copy()

print(19)
################## output format########################
final_output1['Send UID']=final_output1['Source'].astype(str)+"_"+final_output1['SKU'].astype(str)
final_output1['Receive UID']=final_output1['Destination'].astype(str)+"_"+final_output1['SKU'].astype(str)

final_output1['Shipment No.']=final_output1['Source'].astype(str)+"_"+final_output1['Destination'].astype(str)

final_output1['Stock on Floor at sending site (HL volume)']=final_output1['Send UID'].map(stock_on_floor)

final_output1['Reorder Point (HL) at sending site']=final_output1['Send UID'].map(rop)
final_output1['Percentage variance from re-order point']=final_output1['Send UID'].map(variance)
final_output1['Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site']=final_output1['Send UID'].map(plan_load)
final_output1['Demand for the next 7 days (HL Volume) at sending site']=final_output1['Send UID'].map(total_demand)
final_output1['Future stock on hand at sending site in 7 days']=final_output1['Send UID'].map(seven_days)
final_output1['Variance to re-order in 7 days Sending']=final_output1['Send UID'].map(var_vs_rop)


final_output1['Stock on Floor at receiving site (HL volume)']=final_output1['Receive UID'].map(stock_on_floor)

final_output1['Reorder Point (HL) at receiving site']=final_output1['Receive UID'].map(rop)
final_output1['Percentage variance from re-order point at receiving site']=final_output1['Receive UID'].map(variance)
final_output1['Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site']=final_output1['Receive UID'].map(plan_load)
final_output1['Demand for the next 7 days (HL Volume) at receiving site']=final_output1['Receive UID'].map(total_demand)
final_output1['Future stock on hand at receiving site in 7 days']=final_output1['Receive UID'].map(seven_days)
final_output1['Variance to re-order in 7 days receiving']=final_output1['Receive UID'].map(var_vs_rop)

# final_output1.rename(columns={'Source Name':'Sending Site','Destination Name':'Receiving Site','Days to expiry':'Obsolecence Risk days to expiry at sending site (HL)','Qty in HL':'Volume to redeploy(HL)'},inplace=True)
final_output1.rename(columns={'Source Name':'Sending Site','Destination Name':'Receiving Site','Qty in HL':'Volume to redeploy(HL)'},inplace=True)

# final_output1['Obsolecence Risk at sending site (HL)']=final_output1['Send UID'].map(final_obs_vol_dict)
final_trucks['UID']=final_trucks['Source'].astype(str)+"_"+final_trucks['Destination'].astype(str)

cost_USD=dict(list(zip(final_trucks['UID'],final_trucks['Total TripCost( in USD)'])))
#     print('cost_USD',cost_USD)
final_output1['Cost of shipment(USD)']=final_output1['Shipment No.'].map(cost_USD)    
final_output1['Volume to redeploy(Pal)']=final_output1['QTY']


print(6)
final_output1=final_output1.reset_index(drop=True)
final_output1['OOS']=final_output1['Receive UID'].map(oos_dict)
final_output1['OOS']=final_output1['OOS'].fillna(0)

# final_output2=final_output1[['Shipment No.','Sending Site','Receiving Site','SKU','SKU Name','Stock on Floor at sending site (HL volume)',
#                          'Reorder Point (HL) at sending site','Percentage variance from re-order point',
#                          'Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site','Obsolecence Risk at sending site (HL)',
#                          'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Percentage variance from re-order point at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
#                          'Volume to redeploy(HL)','Volume to redeploy(Pal)','Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days','Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving','OOS','Benefit(in USD)','Cost of shipment(USD)']]
final_output2=final_output1[['Shipment No.','Sending Site','Receiving Site','SKU','SKU Name','Stock on Floor at sending site (HL volume)',
                         'Reorder Point (HL) at sending site','Percentage variance from re-order point',
                         'Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site',
                         'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Percentage variance from re-order point at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
                         'Volume to redeploy(HL)','Volume to redeploy(Pal)','Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days','Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving','OOS','Benefit(in USD)','Cost of shipment(USD)']]


final_output2=final_output2.sort_values(by='Shipment No.')
final_output2.rename(columns={'Shipment No.':'Re-deployment ID'},inplace=True)


cols=['Percentage variance from re-order point','Percentage variance from re-order point at receiving site',
     'Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving']

final_output2[cols]=final_output2[cols]/100

final_output2['Percentage variance from re-order point']=final_output2['Percentage variance from re-order point'].astype(float).map("{:.2%}".format)
final_output2['Percentage variance from re-order point at receiving site']=final_output2['Percentage variance from re-order point at receiving site'].astype(float).map("{:.2%}".format)
final_output2['Variance to re-order in 7 days Sending']=final_output2['Variance to re-order in 7 days Sending'].astype(float).map("{:.2%}".format)
final_output2['Variance to re-order in 7 days receiving']=final_output2['Variance to re-order in 7 days receiving'].astype(float).map("{:.2%}".format)
no_sku=final_output2.groupby(['Re-deployment ID'])['SKU'].nunique().reset_index()
no_sku_dict=dict(list(zip(no_sku['Re-deployment ID'],no_sku['SKU'])))
no_pal=final_output2.groupby(['Re-deployment ID'])['Volume to redeploy(Pal)'].sum().reset_index()
no_pal_dict=dict(list(zip(no_pal['Re-deployment ID'],no_pal['Volume to redeploy(Pal)'])))

benefit=final_output2.groupby(['Re-deployment ID'])['Benefit(in USD)'].sum().reset_index()
benefit_dict=dict(list(zip(benefit['Re-deployment ID'],benefit['Benefit(in USD)'])))
final_trucks['No of SKUs']=final_trucks['UID'].map(no_sku_dict)
final_trucks['No of Pallets']=final_trucks['UID'].map(no_pal_dict)
final_trucks['Benefit(in USD)']=final_trucks['UID'].map(benefit_dict)
final_trucks['Benefit vs Cost(USD)']=final_trucks['Benefit(in USD)']-final_trucks['Total TripCost( in USD)']
final_trucks1=final_trucks[['Source', 'Source Name', 'Destination', 'Destination Name','No of SKUs', 'No of Pallets','No of trucks', 'Cost per Truck', 'Total TripCost( in USD)',
       'Benefit(in USD)','Benefit vs Cost(USD)']]





############################################## Load Split among trucks in the output #################################

final_trucks1['UID']=final_trucks1['Source'].astype(str)+"_"+final_trucks1['Destination'].astype(str)
final_output2['Pal left']=final_output2['Volume to redeploy(Pal)']
new_df_final=pd.DataFrame()
id_list=[]

print('final_output2',final_output2)
for id in final_trucks['UID'].unique():
    new_df=pd.DataFrame()
    one_id_original=final_output2[final_output2['Re-deployment ID']==id]
    one_id=final_output2[final_output2['Re-deployment ID']==id]
    one_id['UID']=one_id['Re-deployment ID']+"_"+one_id['SKU'].astype(str)

    one_id_trucks=final_trucks1[final_trucks1['UID']==id]
    no_trucks=one_id_trucks['No of trucks'].unique()[0]
    no_trucks=int(no_trucks)

    print('no_trucks',no_trucks)
    if no_trucks==1:
        print('run check 1')
        one_id['New Pal']=one_id['Volume to redeploy(Pal)']
        one_id['truck']=1
        new_df=pd.concat([one_id,new_df])
    else:

        for i in range(1,no_trucks+1):
            print('run check 2')
            print('range is', i)
            pal=0
            final_pal=0
            flag=1
#                 print(one_id)
            one_id=one_id[one_id['Pal left']!=0]
            if not one_id.empty:                   
                while (final_pal<6)&( not one_id.empty):

                    for sku in one_id['SKU'].unique():
                        print('flag value is',sku)
                        print('sku',sku)
                        sku_id=one_id[one_id['SKU']==sku]
                        max_count=sku_id['Pal left'].unique()[0]
                        pallet_count=sku_id['Pal left'].unique()[0]
                        original_count=sku_id['Pal left'].unique()[0]
                        if pallet_count>6:
                            pallet_count=6
                            sku_id['Pal left']=max_count-6


                        else:
                            pass


                        if pal+pallet_count>6:
                            pallet_count=6-pal
                        pal+=pallet_count

                        final_pal+=pallet_count

                        if final_pal==6:

                            sku_id['New Pal']=pallet_count
                            sku_id['truck']=i
                            new_df=pd.concat([sku_id,new_df])
                            if original_count==pallet_count:
                                print('yes original count true')
                                one_id.loc[one_id['SKU']==sku,'Pal left']=0

                                print(one_id[['SKU','Pal left']])
                            else:
                                one_id.loc[one_id['SKU']==sku,'Pal left']=original_count-pallet_count 

                            break
                        else:

                            sku_id['New Pal']=pallet_count
                            sku_id['truck']=i
                            new_df=pd.concat([sku_id,new_df])

                            if original_count==pallet_count:

                                one_id.loc[one_id['SKU']==sku,'Pal left']=0

                            else:

                                one_id.loc[one_id['SKU']==sku,'Pal left']=original_count-pallet_count 

                            continue




        print('new_df',new_df)
        new_df_grouped=new_df.groupby(['SKU'])['New Pal'].sum().reset_index()
        dict_pal=dict(list(zip(new_df_grouped['SKU'],new_df_grouped['New Pal'])))
        one_id_original['Alloted Pal']=one_id_original['SKU'].map(dict_pal)
        one_id_original['Alloted Pal']= one_id_original['Alloted Pal'].fillna(0)
        one_id_original['Pal left']=one_id_original['Volume to redeploy(Pal)']-one_id_original['Alloted Pal']
        one_left=one_id_original[one_id_original['Pal left']!=0]

        conversion_file=uom_df[(uom_df['Base Unit']=='KG')&(uom_df['Convert Unit']=='CS')].rename(columns={'Conversion Factor':'1KG to CS'})
        conversion_file_2=uom_df[(uom_df['Base Unit']=='CS')&(uom_df['Convert Unit']=='PAL')].rename(columns={'Conversion Factor':'1CS to PAL'})
        conversion_file_2['1PAL to CS'] = 1/conversion_file_2['1CS to PAL']
        conversion_file = conversion_file.merge(conversion_file_2[['Material_Number','1PAL to CS']],on='Material_Number',how='left')
        conversion_file['1 Pal to KG'] = conversion_file['1PAL to CS']/conversion_file['1KG to CS']
        weights_dict=dict(list(zip(conversion_file['Material_Number'],conversion_file['1 Pal to KG'])))

        new_df['1 Pal to KG']=new_df['SKU'].map(weights_dict)
        new_df['Weights']=new_df['1 Pal to KG']*new_df['New Pal']
        trucks_grouped=new_df.groupby(['truck'])['Weights'].sum().reset_index()
        truck_pal_grouped=new_df.groupby(['truck'])['New Pal'].sum().reset_index()
        truck_weight_dict=dict(list(zip(trucks_grouped['truck'],trucks_grouped['Weights'])))
        truck_pallets=dict(list(zip(truck_pal_grouped['truck'],truck_pal_grouped['New Pal'])))
        max_pal_dict=dict()
        for trucks in new_df['truck'].unique():
            max_pal_dict[trucks]=4


        while not one_left.empty:    
            for sku in one_left['SKU'].unique():
                total_weight=0
                one_sku_left=one_left[one_left['SKU']==sku]
                pallet_left=one_sku_left['Pal left'].unique()[0]
                original_pallets_left=one_sku_left['Pal left'].unique()[0]

                ### CHECK ###
                if pallet_left>4:
                    pallet_left=4
                weights=pallet_left*weights_dict[sku]

                while True :

                    sorted_keys_wt={r: truck_weight_dict[r] for r in sorted(truck_weight_dict, key=truck_weight_dict.get, reverse=False)}
                    keys_pal= [key for key, value2 in truck_pallets.items() if value2 < 24 ]
                    keys_list=list(set.intersection(set(sorted_keys_wt), set(keys_pal)))
                    wt_key_list=list(sorted_keys_wt.keys())
                    keys=wt_key_list[0]

                    while True:

                        if keys not in keys_list:

                            wt_key_list.remove(keys)
                            keys=wt_key_list[0]
                            continue
                        else:
                            break


                    total_weight=weights+truck_weight_dict[keys]+total_weight
                    truck_tot_pal=truck_pallets[keys]

                    ### CHECK ###
                    if (total_weight>327215) & (truck_tot_pal<24):
                        extra_loads=327215-truck_weight_dict[keys]
                        pallets_allot=math.floor(extra_loads/weights_dict[sku])
                        if pallets_allot>(24-truck_tot_pal):
                            pallets_allot=24-truck_tot_pal
                        if pallets_allot==0:

                            del truck_weight_dict[keys]
                            continue
                        else:
                            key_value=truck_weight_dict[keys]
                            count=truck_pallets[keys]
                            truck_weight_dict[keys]=key_value+extra_loads
                            truck_pallets[keys]=count+pallets_allot
                            break
                    elif truck_tot_pal + pallet_left >24:
                        pallets_allot= 24 - truck_tot_pal
                        if pallets_allot==0:
                            del truck_pallets[keys]
                            continue
                        else:
                            count=truck_pallets[keys]
                            key_value=truck_weight_dict[keys]
                            truck_weight_dict[keys]=total_weight
                            truck_pallets[keys]=count+pallets_allot

                            break
                    else:
                        pallets_allot=pallet_left
                        count=truck_pallets[keys]
                        key_value=truck_weight_dict[keys]
                        truck_weight_dict[keys]=total_weight
                        truck_pallets[keys]=count+pallets_allot
                        break

                final_sku=one_sku_left.copy(deep=True)
                del final_sku['Alloted Pal']
                del final_sku['Pal left']
                final_sku['New Pal']=pallets_allot

                final_sku['truck']=keys
                new_df=pd.concat([new_df,final_sku])
                one_left.loc[one_left['SKU']==sku,'Pal left']=original_pallets_left-pallets_allot
                one_left=one_left[one_left['Pal left']!=0].reset_index(drop=True)


    new_df_final=pd.concat([new_df_final,new_df])

    id_list.append(id) 

    
    


new_df_final_copy=new_df_final.copy(deep=True)


new_df_final=new_df_final_copy.copy(deep=True)
new_df_final=new_df_final_copy.copy(deep=True)
new_df_final['Volume to redeploy(Pal)']=new_df_final['New Pal']
new_df_final['Re-deployment ID']=new_df_final['Re-deployment ID']+"_"+new_df_final['truck'].astype(str)

conversion_file=uom_df[(uom_df['Base Unit']=='KG')&(uom_df['Convert Unit']=='CS')].rename(columns={'Conversion Factor':'1KG to CS'})
conversion_file_2=uom_df[(uom_df['Base Unit']=='CS')&(uom_df['Convert Unit']=='PAL')].rename(columns={'Conversion Factor':'1CS to PAL'})
conversion_file_2['1PAL to CS'] = 1/conversion_file_2['1CS to PAL']
conversion_file = conversion_file.merge(conversion_file_2[['Material_Number','1PAL to CS']],on='Material_Number',how='left')
conversion_file_CS=uom_df[(uom_df['Base Unit']=='HL')&(uom_df['Convert Unit']=='CS')].rename(columns={'Conversion Factor':'HL to CS'})
conversion_file = conversion_file.merge(conversion_file_CS[['Material_Number','HL to CS']],on='Material_Number',how='left')    
conversion_file['1 Pal to KG'] = conversion_file['1PAL to CS']/conversion_file['1KG to CS']
conversion_file['1Pal to HL'] = conversion_file['1PAL to CS']/conversion_file['HL to CS']
weights_dict=dict(list(zip(conversion_file['Material_Number'],conversion_file['1 Pal to KG'])))
weights_dict_hl=dict(list(zip(conversion_file['Material_Number'],conversion_file['1Pal to HL'])))
cs_dict=dict(list(zip(conversion_file_CS['Material_Number'],conversion_file_CS['HL to CS'])))

new_df_final['1 Pal to KG']=new_df_final['SKU'].map(weights_dict)
new_df_final['total weights']=new_df_final['1 Pal to KG']*new_df_final['Volume to redeploy(Pal)']
new_df_final=new_df_final.fillna('Not available')
print('new_df_final before grouping ', new_df_final.shape)
# new_df_final=new_df_final.groupby(['Re-deployment ID','Sending Site','Receiving Site','SKU','SKU Name','Stock on Floor at sending site (HL volume)',
#                          'Reorder Point (HL) at sending site','Percentage variance from re-order point',
#                          'Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site','Obsolecence Risk at sending site (HL)',
#                          'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Percentage variance from re-order point at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site', 
#                          'Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days','Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving',
#                          'OOS','Benefit(in USD)', 'Cost of shipment(USD)'])[['Volume to redeploy(Pal)','total weights']].sum().reset_index()
new_df_final=new_df_final.groupby(['Re-deployment ID','Sending Site','Receiving Site','SKU','SKU Name','Stock on Floor at sending site (HL volume)',
                         'Reorder Point (HL) at sending site','Percentage variance from re-order point',
                         'Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site',
                         'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Percentage variance from re-order point at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site', 
                         'Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days','Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving',
                         'OOS','Benefit(in USD)', 'Cost of shipment(USD)'])[['Volume to redeploy(Pal)','total weights']].sum().reset_index()

new_df_final['SKU']=new_df_final['SKU'].astype(str)
new_df_final['1Pal to HL']=new_df_final['SKU'].map(weights_dict_hl)
new_df_final['Volume to redeploy(HL)']=new_df_final['Volume to redeploy(Pal)']*new_df_final['1Pal to HL']
new_df_final['SID']=new_df_final['Sending Site']+"_"+new_df_final['SKU'].astype(str)
new_df_final['DID']=new_df_final['Receiving Site']+"_"+new_df_final['SKU'].astype(str)
new_df_final['SID_new']=new_df_final['Re-deployment ID'].str[:4]+"_"+new_df_final['SKU'].astype(str)
new_df_final['DID_new']=new_df_final['Re-deployment ID'].str[5:9]+"_"+new_df_final['SKU'].astype(str)
new_df_final['SRID']=new_df_final['Sending Site']+"_"+new_df_final['SKU'].astype(str)+"_"+new_df_final['Receiving Site'].astype(str)
# new_df_final['Weighed MACO']=new_df_final['SID'].map(weighed_maco_dict)
new_df_final['Total MACO']=new_df_final['SRID'].map(sum_weights_dict)
new_df_final['Benefits (in USD)']=new_df_final['Total MACO']*new_df_final['Volume to redeploy(HL)']
# new_df_final['Benefits obs new']=new_df_final['Weighed MACO']*new_df_final['Volume to redeploy(HL)']
new_df_final['Actual MACO']=new_df_final['SID'].map(org_maco_dict)
new_df_final['Benefits without obs']=new_df_final['Actual MACO']*new_df_final['Volume to redeploy(HL)']
# new_df_final['Obs value']=new_df_final['Benefits obs new']#-new_df_final['Benefits without obs']

##CHECK##
# new_df_final['trip cost saved from destruction process']=new_df_final['Volume to redeploy(HL)']*new_df_final['SID_new'].map(trip_cost_destruction_per_hl_dict)
# new_df_final['destruction process cost']=new_df_final['Volume to redeploy(HL)']*new_df_final['SID_new'].map(cost_destruction_per_hl_dict)
# new_df_final['cost of goods saved from obsolecence']=new_df_final['Volume to redeploy(HL)']*new_df_final['SID_new'].map(cost_of_goods_per_hl_dict)
# new_df_final['Obs MACO benefit']=new_df_final['Volume to redeploy(HL)']*new_df_final['SID_new'].map(obs_maco_dict)
new_df_final['OOS MACO benefit']=new_df_final['Volume to redeploy(HL)']*new_df_final['DID_new'].map(oos_maco_dict)  

new_df_final['OOS value']=new_df_final['Benefits (in USD)']#-new_df_final['Obs value']#-new_df_final['Benefits without obs']
new_df_final['HL to CS']=new_df_final['SKU'].map(cs_dict)
new_df_final['Volume to redeploy(CS)']=new_df_final['Volume to redeploy(HL)']*new_df_final['HL to CS']
new_df_final['Volume to redeploy(CS)']=new_df_final['Volume to redeploy(CS)'].apply(np.floor)
new_df_final['UID']=new_df_final['Sending Site']+"_"+new_df_final['SKU'].astype(str)
# new_df_final['Days to expiry']=new_df_final['UID'].map(expiry_dict)

# new_df_final1=new_df_final[['Re-deployment ID','Sending Site','Receiving Site','SKU','SKU Name','Stock on Floor at sending site (HL volume)',
#                          'Reorder Point (HL) at sending site','Percentage variance from re-order point','SID','DID',
#                          'Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site','Obsolecence Risk at sending site (HL)','Days to expiry',
#                          'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Percentage variance from re-order point at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
#                          'Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days','Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving',
#    'OOS','Benefits (in USD)', 'Cost of shipment(USD)','Volume to redeploy(HL)','Volume to redeploy(CS)','Volume to redeploy(Pal)','trip cost saved from destruction process','destruction process cost','cost of goods saved from obsolecence','Obs MACO benefit','OOS MACO benefit']]

new_df_final1=new_df_final[['Re-deployment ID','Sending Site','Receiving Site','SKU','SKU Name','Stock on Floor at sending site (HL volume)',
                         'Reorder Point (HL) at sending site','Percentage variance from re-order point','SID','DID',
                         'Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site',
                         'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Percentage variance from re-order point at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
                         'Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days','Variance to re-order in 7 days Sending','Variance to re-order in 7 days receiving',
   'OOS','Benefits (in USD)', 'Cost of shipment(USD)','Volume to redeploy(HL)','Volume to redeploy(CS)','Volume to redeploy(Pal)','OOS MACO benefit']]




print('new_df_final shape is post ', new_df_final.shape)
# new_df_final['New id']=new_df_final['Re-deployment ID'].astype(str).str[:-2]
# benefit_USD=new_df_final.groupby(['New id'])['Obs value'].sum().reset_index()
# benefit_USD_dict=dict(list(zip(benefit_USD['New id'],benefit_USD['Obs value'])))

new_df_final['New id']=new_df_final['Re-deployment ID'].astype(str).str[:-2]
benefit_USD_oos=new_df_final.groupby(['New id'])['OOS value'].sum().reset_index()
benefit_USD_oos_dict=dict(list(zip(benefit_USD_oos['New id'],benefit_USD_oos['OOS value'])))



old_benefit=new_df_final.groupby(['New id'])['Benefits without obs'].sum().reset_index()
old_benefit_dict=dict(list(zip(old_benefit['New id'],old_benefit['Benefits without obs'])))

##CHECK##
# trip_cost_destruction=new_df_final.groupby(['New id'])['trip cost saved from destruction process'].sum().reset_index()
# trip_cost_destruction_dict=dict(list(zip(trip_cost_destruction['New id'],trip_cost_destruction['trip cost saved from destruction process'])))

# cost_destruction=new_df_final.groupby(['New id'])['destruction process cost'].sum().reset_index()
# cost_destruction_dict=dict(list(zip(cost_destruction['New id'],cost_destruction['destruction process cost'])))

# cost_of_goods=new_df_final.groupby(['New id'])['cost of goods saved from obsolecence'].sum().reset_index()
# cost_of_goods_dict=dict(list(zip(cost_of_goods['New id'],cost_of_goods['cost of goods saved from obsolecence'])))

# obs_maco_benefit=new_df_final.groupby(['New id'])['Obs MACO benefit'].sum().reset_index()
# obs_maco_benefit_dict=dict(list(zip(obs_maco_benefit['New id'],obs_maco_benefit['Obs MACO benefit'])))

oos_maco_benefit=new_df_final.groupby(['New id'])['OOS MACO benefit'].sum().reset_index()
oos_maco_benefit_dict=dict(list(zip(oos_maco_benefit['New id'],oos_maco_benefit['OOS MACO benefit'])))


# final_trucks1['Obsolescence Benefit(USD)']=final_trucks1['UID'].map(benefit_USD_dict)
final_trucks1['OOS Benefit(USD)']=final_trucks1['UID'].map(benefit_USD_oos_dict)

#CHECK
# final_trucks1['trip cost saved from destruction process']=final_trucks1['UID'].map(trip_cost_destruction_dict)
# final_trucks1['destruction process cost']=final_trucks1['UID'].map(cost_destruction_dict)
# final_trucks1['cost of goods saved from obsolecence']=final_trucks1['UID'].map(cost_of_goods_dict)
# final_trucks1['Obs MACO benefit']=final_trucks1['UID'].map(obs_maco_benefit_dict)
final_trucks1['OOS MACO benefit']=final_trucks1['UID'].map(oos_maco_benefit_dict)
#     final_trucks1['Benefit (in USD)']=final_trucks1['UID'].map(old_benefit_dict)
# final_trucks1['Obsolescence Benefit(USD)']=final_trucks1['Obsolescence Benefit(USD)'].fillna(0)
final_trucks1['OOS Benefit(USD)']=final_trucks1['OOS Benefit(USD)'].fillna(0)

#CHECK
# final_trucks1['trip cost saved from destruction process']=final_trucks1['trip cost saved from destruction process'].fillna(0)
# final_trucks1['destruction process cost']=final_trucks1['destruction process cost'].fillna(0)
# final_trucks1['cost of goods saved from obsolecence']=final_trucks1['cost of goods saved from obsolecence'].fillna(0)
# final_trucks1['Obs MACO benefit']=final_trucks1['Obs MACO benefit'].fillna(0)
final_trucks1['OOS MACO benefit']=final_trucks1['OOS MACO benefit'].fillna(0)


# final_trucks1['Benefit vs Cost(USD)']=final_trucks1['Obsolescence Benefit(USD)']+final_trucks1['OOS Benefit(USD)']-final_trucks1['Total TripCost( in USD)']
final_trucks1['Benefit vs Cost(USD)']=final_trucks1['OOS Benefit(USD)']-final_trucks1['Total TripCost( in USD)']

# final_trucks2=final_trucks1[['Source', 'Source Name', 'Destination', 'Destination Name','No of SKUs', 'No of Pallets','No of trucks', 'Cost per Truck', 'Total TripCost( in USD)','Obsolescence Benefit(USD)','trip cost saved from destruction process','destruction process cost','cost of goods saved from obsolecence','Obs MACO benefit','OOS MACO benefit','OOS Benefit(USD)','Benefit vs Cost(USD)']]
final_trucks2=final_trucks1[['Source', 'Source Name', 'Destination', 'Destination Name','No of SKUs', 'No of Pallets','No of trucks', 'Cost per Truck', 'Total TripCost( in USD)','OOS MACO benefit','OOS Benefit(USD)','Benefit vs Cost(USD)']]

final_trucks2=final_trucks2.sort_values(by=['Benefit vs Cost(USD)'],ascending=False)
final_trucks2=final_trucks2.reset_index(drop=True)
final_trucks2['Ranking (as per Benefit vs Cost(USD))']=final_trucks2.index+1

# cols2=['Stock on Floor at sending site (HL volume)','Reorder Point (HL) at sending site','Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site','Obsolecence Risk at sending site (HL)',
#                          'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
#                          'Volume to redeploy(HL)','Volume to redeploy(CS)','Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days',
#    'OOS','Benefits (in USD)', 'Cost of shipment(USD)','trip cost saved from destruction process','destruction process cost','cost of goods saved from obsolecence','Obs MACO benefit','OOS MACO benefit']

# cols3=['Cost per Truck', 'Total TripCost( in USD)','Obsolescence Benefit(USD)','trip cost saved from destruction process','destruction process cost','cost of goods saved from obsolecence','Obs MACO benefit','OOS MACO benefit','OOS Benefit(USD)','Benefit vs Cost(USD)']

# cols4=['Stock on Floor at sending site (HL volume)','Reorder Point (HL) at sending site','Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site','Obsolecence Risk at sending site (HL)',
#                          'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
#                          'Volume to redeploy(HL)','Volume to redeploy(CS)','Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days',
#    'OOS', 'Cost of shipment(USD)','Cost vs Benefit','trip cost saved from destruction process','destruction process cost','cost of goods saved from obsolecence','Obs MACO benefit','OOS MACO benefit']

cols2=['Stock on Floor at sending site (HL volume)','Reorder Point (HL) at sending site','Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site',
                         'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
                         'Volume to redeploy(HL)','Volume to redeploy(CS)','Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days',
   'OOS','Benefits (in USD)', 'Cost of shipment(USD)','OOS MACO benefit']

cols3=['Cost per Truck', 'Total TripCost( in USD)','OOS MACO benefit','OOS Benefit(USD)','Benefit vs Cost(USD)']

cols4=['Stock on Floor at sending site (HL volume)','Reorder Point (HL) at sending site','Planned Loads for next 7 (Planned in - Planned out + intransit) days (HL) at sending site','Demand for the next 7 days (HL Volume) at sending site',
                         'Stock on Floor at receiving site (HL volume)','Reorder Point (HL) at receiving site','Planned  Loads for next 7 days (Planned in - Planned out + intransit)  (HL) at receiving site','Demand for the next 7 days (HL Volume) at receiving site',
                         'Volume to redeploy(HL)','Volume to redeploy(CS)','Future stock on hand at sending site in 7 days','Future stock on hand at receiving site in 7 days',
   'OOS', 'Cost of shipment(USD)','Cost vs Benefit','OOS MACO benefit']



new_df_final1[cols2]=new_df_final1[cols2]
final_trucks2[cols3]=final_trucks2[cols3]

# new_df_final1[cols2]=new_df_final1[cols2].round(2)
# final_trucks2[cols3]=final_trucks2[cols3].round(2)



# final_output2_truck[cols4]=final_output2_truck[cols4].round(2)

# obs_cols=['Vol', 'Customer_Orders','Daily Demand Day1',
#                           'Daily Demand Day2','Daily Demand Day3','Daily Demand Day4',
#                           'Daily Demand Day5','Daily Demand Day6','Daily Demand Day7',
#                           'Demand_7days','Calculated Obs']

# obs_final_v1[obs_cols]=obs_final_v1[obs_cols].round(2)

# obs_final_v1.rename(columns={'Site_Code':'Site Code','Material_Number':'Material Number','Vol':'Vol( as per Stock age)',
#                          'Customer_Orders':'Customer Orders',
#                          'Demand_7days':'Total Demand 7 Days'},inplace=True)


new_df_final1['Agreement to Recommendation(Yes/No)']=''
new_df_final1['Recommendation Executed(Yes/No)']=''
new_df_final1['Reason for non-agreement/non-execution']=''


########### creating an additional risk column for expiry day< 45 days ############
#     del obs_data2['In Risk']
#     try:
#         obs_data2['Risk( Expiry day< 45 days)']=np.where((obs_data2['Days to expiry']<45)&(obs_data2['Keg tag']=='Not KEG'),'High Risk','')
#     except:
#         pass

#     writer = pd.ExcelWriter(input_path/'SDO Validations.xlsx', engine='xlsxwriter')
#     final_stock_cal_file.to_excel(writer,sheet_name = 'Volume for Redeployment', index = False)
#     final_solver_input.to_excel(writer,sheet_name = 'Input to Model', index = False)
#     new_df_final1.to_excel(writer,sheet_name = 'Output1', index = False)
#     final_trucks2.to_excel(writer,sheet_name = 'Output2', index = False)
#     final_output2_truck.to_excel(writer,sheet_name='One truck load Output',index=False)
#     obs_data2.to_excel(writer,sheet_name='Obsolete Tag as per rules',index=False)
#     obs_final_v1.to_excel(writer,sheet_name='Obs calculation',index=False)
#obs_df2_final1.to_excel(writer,sheet_name='Obsolescense Information',index=False)
#     writer.save()

#     writer = pd.ExcelWriter(input_path/'SDO Validations.xlsx', engine='xlsxwriter')
#     final_stock_cal_file.to_csv(output_path/'Volume for Redeployment.csv', index = False)
#     final_solver_input.to_csv(output_path/'Input to Model.csv', index = False)
#     new_df_final1.to_csv(output_path/'Output1.csv', index = False)
#     final_trucks2.to_csv(output_path/'Output2.csv', index = False)
#     final_output2_truck.to_csv(output_path/'One truck load Output.csv',index=False)
#     stock_age_df2.to_csv(output_path/'Obsolete Tag as per rules.csv',index=False)
#     obs_final_v1.to_csv(output_path/'Obs calculation.csv',index=False)
#     return print('Solver output generated')


# if __name__ == '__main__':
#     t1=Thread(target = one_truck_load,args=(output_path,oos_dict,obsolete_tag_final,depot_code_dict,sku_name_dict,uom_df,usd_to_pesos)).start()
#     t2=Thread(target = solver,args=(main_file,output_path,obsolete_tag_final,depot_code_dict,sku_name_dict,final_solver_input,oos_dict,stock_age_df2,final_stock_cal_file,obs_final_v1,cvf_dict_new,uom_df,usd_to_pesos,trip_cost_df_outside_region)).start()
 
    
    
final_stock_cal_file.to_csv(output_path/'Volume for Redeployment_new.csv', index = False)
final_solver_input.to_csv(output_path/'Input to Model_new.csv', index = False)
new_df_final1.to_csv(output_path/'Output1_new.csv', index = False)
final_trucks2.to_csv(output_path/'Output2_new.csv', index = False)
# final_output2_truck.to_csv(output_path/'One truck load Output_new.csv',index=False)
# stock_age_df2.to_csv(output_path/'Obsolete Tag as per rules_new.csv',index=False)
# obs_final_v1.to_csv(output_path/'Obs calculation_new.csv',index=False)


TimeEnd = time.time()


print("The Start End time:",TimeEnd - TimeStart)