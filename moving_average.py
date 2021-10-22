import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv('houston_temp_2019.csv')
df.set_index('Day', inplace=True)

st.title('Exponential Moving Averages')


@st.cache
def ewma(x, y, beta=0.9, bias_correction=True):
    avg_arr = np.zeros((x.shape[0]+1))
    avg_arr_corr = np.zeros((x.shape[0] + 1))
    avg_arr[0] = 0
    avg_arr_corr[0] = 0
    for t in x:
        vt = beta * avg_arr[t-1] + (1-beta) * y[t]
        vt_corr = vt / (1 - (beta ** t))
        avg_arr[t] += vt
        avg_arr_corr[t] += vt_corr
    if bias_correction:
        return avg_arr_corr      
    return avg_arr

with st.form('EMA'):
    beta = st.slider('β Value', 0.0, 1.0, 0.9)
    bias = st.checkbox('Bias Correction', True)
    st.form_submit_button()


em_avg = ewma(df.index, df['AvgTemperature'], beta=beta, bias_correction=bias)
fig, ax = plt.subplots()
plt.figure(figsize = (8,6))
ax.scatter(x=df.index, y=df['AvgTemperature'])
ax.plot(df.index, em_avg[1:], 'r-')
st.pyplot(fig)


