import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

@st.cache_data
def load_data():
    print("Loading data...")  # only executed once

    frekvencija = np.linspace(20, 20000, 500)

    # Example measured uncertainty
    err = 2 + 0.5 * np.log10(frekvencija)

    return frekvencija, err

#
@st.cache_data
def load_measurement_data():
    df=pd.read_pickle('data_power_dodeka.pkl')
    pozicije=df.POZ.drop_duplicates()
    loudspeakers=df.LS.drop_duplicates()
    versions=df.VER.drop_duplicates()
    DTs=df.DT.drop_duplicates()
    dolzine=df.dolzina.drop_duplicates()
    
    data=np.load('data_power_dodeka.npz')
    frekvencija=data['frekvencija']
    SPL_STACnc=data['SPL_STAC']
    SPLnc=data['SPL']

    data=np.load('calibration.npz')
    CALIB=data['CALIB']
    CALIB=CALIB/np.max(CALIB)

    SPLc=1*SPLnc
    SPL_STACc=1*SPL_STACnc

    for kk in range(36):
        SPLc[:,kk,:]=SPLnc[:,kk,:]*CALIB[kk]
        SPL_STACc[:,kk,:]=SPL_STACnc[:,kk,:]*CALIB[kk]
    

    return df, pozicije,loudspeakers, versions, DTs,dolzine, frekvencija,SPLnc,SPL_STACnc,SPLc,SPL_STACc

[df, pozicije,loudspeakers, versions, DTs,dolzine, frekvencija,SPLnc,SPL_STACnc,SPLc,SPL_STACc] = load_measurement_data()


[c1,c2,c3,c4]=st.columns([3,3,3,3])


dolzina = c1.select_slider(
    "Analysis window length [s]",
    options=dolzine)

DT = c2.select_slider(
    "Reverberant decay delay [s]",
    options=DTs)

# Nmic = c3.select_slider(
#     "N mic average",
#     options=range(1,37))

Nmic = c3.select_slider(
    "N mic average",
    options=[2,4,8,16,32,36])

rng = random.Random(4.4)
R=rng.sample(range(36), k=Nmic)

print(R)
calibrate = c4.checkbox("Mic calibration")
normalize = c4.checkbox("Normalize")
logY = c4.checkbox("logY scale")

if calibrate:
    SPL=1*SPLc[:,R,:]
    SPL_STAC=1*SPL_STACc[:,R,:]
else:
    SPL=1*SPLnc[:,R,:]
    SPL_STAC=1*SPL_STACnc[:,R,:]
print(SPL.shape,SPL_STAC.shape)
IND=df.index[(df.DT==DT) & (df.dolzina==dolzina)].to_numpy().astype(int)



povp1=np.mean(SPL[IND,:,:],axis=1)/dolzina
povp1_stac=np.mean(SPL_STAC[IND,:,:],axis=1)/dolzina


popv2=np.mean(povp1,axis=0)/dolzina
err=np.std(povp1,axis=0)/dolzina
popv2_stac=np.mean(povp1_stac,axis=0)/dolzina
err_stac=np.std(povp1_stac,axis=0)/dolzina


if normalize:
    n1=np.argmin(np.abs(frekvencija-2000))
    fak=np.mean(popv2_stac[:n1])/np.mean(popv2[:n1])
    popv2=popv2*fak
    err=err*fak

# Plot
fig, ax = plt.subplots(figsize=(8,4))

ax.plot(frekvencija, popv2, color="#7570B3")
ax.fill_between(
    frekvencija,
    popv2 - err / 2,
    popv2 + err / 2,
    color="#7570B3",
    alpha=0.4,
    label='reverberant'
)

ax.plot(frekvencija, popv2_stac, color="#1B9E77")
ax.fill_between(
    frekvencija,
    popv2_stac - err_stac / 2,
    popv2_stac + err_stac / 2,
    color="#1B9E77",
    alpha=0.4,
    label='stationary'
)

ax.set_xscale("log")
ax.grid(True)
plt.legend(frameon=True)
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude [a.u.]')

if logY:
    plt.yscale('log')
    plt.ylim(np.max(popv2_stac*4/100),np.max(popv2_stac*2))
else:
    plt.yscale('linear')
    plt.ylim(0,np.max(popv2_stac*1.8))

plt.xlim(100,10000)
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(8,4))

n1=np.argmin(np.abs(frekvencija-100))
stdev=err/popv2*100
stdev_stac=err_stac/popv2_stac*100


E=np.mean(stdev[n1:])
E_stac=np.mean(stdev_stac[n1:])


ax.plot(frekvencija, stdev, color="#7570B3",label='reverberant, mean: '+ str(np.round(E,1))+'%')
ax.plot(frekvencija, stdev_stac, color="#1B9E77",label='stationary, mean: '+ str(np.round(E_stac,1))+'%')

#plt.text(frekvencija,20,str(np.round(E,1)))

ax.set_xscale("log")
ax.grid(True)
plt.legend(frameon=True, loc=1)
plt.xlabel('frequency [Hz]')
plt.ylabel('standard deviation [%]')

plt.ylim(0,100)

plt.xlim(100,10000)
st.pyplot(fig)