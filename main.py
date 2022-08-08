import streamlit as st
import pickle
import numpy as np

# import the model
pipe= pickle.load(open("pipe.pkl","rb"))
df=pickle.load(open("df.pkl","rb"))


st.title("Laptop Predicter")

# brand
company=st.selectbox("Brand",df["Company"].unique())

# type of laptop
type=st.selectbox("Type",df["TypeName"].unique())

# Ram
ram=st.selectbox("Ram(In GB)",[2,4,6,8,12,16,32,64])

# Weight
weight=st.number_input("weight of the Laptop")

# Touchscreen
Touchscreen=st.selectbox("Touchscreen",["Yes","No"])

# IPS
ips=st.selectbox("IPS",["Yes","No"])

# Ppi
screen_size= st.number_input("Screen Size")

# resolution
resolution=st.selectbox("Screen Resolution",["1920x1080","1366x768","1600x900","3840x2160","3200x1800",
                                             "2880x1800","2560x1600","2560x1440","2340x1440"])

# Cpu
cpu=st.selectbox("CPU",df["Cpu brand"].unique())

# HDD
hdd=st.selectbox("HDD( in GB)",[0,128,256,512,1024,2048])

# SSD
ssd=st.selectbox("SSD( in GB)",[0,8,128,256,512,1024])

# Gpu
gpu=st.selectbox("GPU",df["Gpu brand"].unique())

# OS
os=st.selectbox("OS",df["os"].unique())

if st.button("Predict Price"):
    ppi = None
    if Touchscreen == "Yes":
        Touchscreen = 1
    else:
        Touchscreen = 0
    if ips == "Yes":
        ips = 1
    else:
        ips = 0


    X_res=int(resolution.split("x")[0])
    Y_res=int(resolution.split("x")[1])
    ppi= ((X_res**2) +(Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, Touchscreen, ips, ppi,
                      cpu, hdd, ssd, gpu, os])
    query=query.reshape(1,12)

    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))



# For running in terminal streamlit run filename.py






