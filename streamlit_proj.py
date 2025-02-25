import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

# Data files and constant values 
URL_1 = 'https://raw.githubusercontent.com/klesment/PopProj/main/ESTasfrRR.txt'
URL_2 = 'https://raw.githubusercontent.com/klesment/PopProj/main/LT.txt'
URL_3 = 'https://raw.githubusercontent.com/klesment/PopProj/main/Population.txt'

@st.cache_data
def load_data(url):
    s = requests.get(url).content
    ddata = pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\s+', header = 1)
    return ddata

asfr = load_data(URL_1)
lt = load_data(URL_2)
pop = load_data(URL_3)

asfr = asfr.apply(pd.to_numeric, errors = 'coerce')

lt.loc[lt['Age'] == '110+', 'Age'] = 110 # open upper age interval to 110
lt = lt.apply(pd.to_numeric)
lt.loc[lt['qx'] == 0, 'qx'] = np.nan # zeros to NaN

pop.loc[pop['Age'] == '110+', 'Age'] = 110 # open upper age interval to 110
pop = pop.apply(pd.to_numeric)


# Constants
# mean age at birth in 2019
mab = 30.62
# standard deviation of mean age at birth in 2019
sd_mab = 5.62

# Constructing and populating the Leslie matrix
base_year = 2019

# life table
lt19 = lt[lt['Year']==base_year]

# population structure
pop19 = pop[pop['Year']==base_year]
N0 = pop19.Total[0:110].tolist()

# ASFR
asfr19 = asfr[asfr['Year']==base_year]

# Matrix elements
# Probability of survival at each age from life table, goes to matrix subdiagonal
SUBD = lt19.Lx.iloc[1:].values/lt19.Lx.iloc[0:110].values
SUBD[-1:] = 0
Tx = lt19['Tx']

# Age-specific fertility rates, values for ages 12-54
Fx = [0]*12 + asfr19.ASFR.values.tolist() + [0]*55
k = (1/(1+1.05))*(lt19.Lx.iloc[0]/200000)
R1 = (k*(np.array(Fx[0:110]) + np.array(Fx[1:111]) * np.array(SUBD[0:110])))

# matrix
L = np.zeros((109, 110))

np.fill_diagonal(L, SUBD[0:109])

L = np.vstack((R1,L))

L[109,109] = Tx.values[110]/Tx.values[109]


# Projection function
def proj(m, t, n0):
    Nproj = []

    for i in range(0, t, 1):
        if i == 0:
            project = m
            Nproj = np.dot(project, n0)

        if i > 0:
            project = np.dot(project, m)
            Nproj = np.dot(project, n0)

    return Nproj

# Gamma model function
def asfr_gamma(aa):
    f_x_out = []

  # parameters
    a1 = aa['tfr']
    a2 = aa['mab']/(aa['sd_mab']**2)
    a3 = (aa['mab']/aa['sd_mab'])**2

  # age range
    ages = list(range(12,50,1))

    for age in ages:
        v = (1/gamma(a3))*a1*(a2**a3)*(age**(a3-1))*np.exp(-a2*age)
        f_x_out.append(v)

  # 1-D array
    return np.array(f_x_out).ravel()

# Dynamic projection function
def project_dyn(lmat, pop, per):
    '''
    lmat - yearly L matrix; pop - initial population structure; per - period to be projected
    '''

    N_0 = pop

    for i in range(1, per, 1):
        N_x = proj(lmat[(i-1)], 1, N_0)
        N_0 = N_x

    return N_0

# Leslie matrix function
def leslie(fert):
    '''
    fert - ASFR vector;
    mort - life table dataframe
    'lt19' data frame must be present
    '''
    mort = lt19

    SUBD = mort.Lx.iloc[1:].values/mort.Lx.iloc[0:110].values
    SUBD[-1:] = 0

    Tx = mort['Tx']

    Fx = [0]*12 + fert.values.tolist() + [0]*61
    k = (1/(1+1.05))*(mort.Lx.iloc[0]/200000)
    R1 = (k*(np.array(Fx[0:110]) + np.array(Fx[1:111]) * np.array(SUBD[0:110])))

    L = np.zeros((109, 110))
    np.fill_diagonal(L, SUBD[0:109])
    L = np.vstack((R1,L))
    L[109,109] = Tx.values[110]/Tx.values[109]

    return L

# TFR change ramp function
def ramp_fun(TFR_chng,speed,pr_per):
    a,b,c = TFR_chng,5,speed

    x = np.linspace(0,1,pr_per)
    y = a*np.exp(-b*np.exp(-c*x))
    y2 = [y[-1]]*0
    return np.add([*y,*y2],1)



# Figure
plt.rcParams['figure.figsize'] = [8, 4.5]

st.sidebar.markdown('''Vali prognoosi eeldused: sündimustaseme (TFR) muutus, muutuse kiirus, keskmine sünnitusvanus ja prognoosi pikkus.''')

option_map = {5: "Aeglasem", 6: "Keskmine", 8: "Kiirem"}
def user_input_features():
    TFR_Change  = st.sidebar.number_input("TFR muutus % võrreldes 2019.a", 
                                          min_value=-50, max_value=50, step=10, value=0)/100
    Ramp = st.sidebar.segmented_control("TFR muutuse kiirus", 
                                        options=option_map.keys(), 
                                        format_func=lambda option: option_map[option], 
                                        selection_mode='single', default=6)
    MAB_end = st.sidebar.number_input("Keskmine sünnitusvanus", min_value=27.0, max_value=33.0, step=0.5, value=mab)
    Years = st.sidebar.slider("Prognoosi pikkus (aastat)", min_value=5, max_value=100, step=5, value=5)
    return TFR_Change, Ramp, MAB_end, Years
    



TFR_Change, Ramp, MAB_end, Years = user_input_features()


mab_stop,sd_mab_stop,period = MAB_end,sd_mab,Years

# fixed starting values from source data frames
tfr_start,mab_start,sd_mab_start = round(sum(asfr['ASFR'][asfr['Year']==base_year]),2),mab,sd_mab

if period==0:
    period += 1

# user input values
d = pd.DataFrame(data={
    'tfr': np.linspace(tfr_start, tfr_start, period),
    'mab': np.linspace(mab_start, mab_stop, period),
    'sd_mab': np.linspace(sd_mab_start, sd_mab_stop, period)
})

d['tfr'] = np.multiply(d['tfr'], ramp_fun(TFR_Change,Ramp,period))

if period==0:
    tfr_last = round(d['tfr'][period],2)
if period>0:
    tfr_last = round(d['tfr'][period-1],2)

# age-specific fertility rates from the gamma function
F = []
F = np.array(d.apply(asfr_gamma, axis=1).tolist())

D = pd.DataFrame(data=F[0:,0:])

# Leslie matrices as lists
LL = np.array(D.apply(leslie, axis=1))

# projection function
out = project_dyn(LL, N0, period)
# Population size
p_size_start,p_size_end = sum(pop['Total'][pop['Year']==base_year]),sum(out)

# Plotting
col1, col2 = st.columns([1, 1])

with col1:
    # TFR plot
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=2)
    tfr_plot = sns.lineplot(data=d, x=list(range(base_year, base_year + len(LL), 1)), y='tfr', linewidth=2.5, ax=plt.gca())
    tfr_plot.set(ylabel='Summaarkordaja', xlabel='Aasta')
    st.caption(f"Summmaarkordaja prognoositud muutus {base_year} - {base_year + len(LL)}")
    st.pyplot(tfr_plot.get_figure())

with col2:
    a, b = st.columns(2)
    c, d = st.columns(2)

    a.metric(f"TFR {base_year + period}", tfr_last, round(tfr_last-tfr_start, 1), border=False)
    b.metric(f"Sünnitusvanus {base_year + period}", round(mab_stop,1), round(mab_stop-mab_start,1), border=False)
    c.metric(f"Rahvaarv (milj.)  {base_year + period}", round(p_size_end/1000000,3), round(p_size_end-p_size_start), border=False)
    d.metric(f"Sündide arv {base_year + period}", round(out[0]), "", border=False)


plt.clf()

st.divider()
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1)
p = sns.barplot(x=range(0,110,1), y=pd.Series(out))
plt.plot([period-1,period-1], [0,20000], color='red', ls='dotted',linewidth=4)
p.set(xlabel='1-aastane vanusrühm', ylabel='Inimesi vanusrühmas')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xticks(np.arange(0, 100, 5))
st.caption(f"Prognoositud vanuskoostis {base_year + period} aastal")

st.pyplot(p.get_figure())

