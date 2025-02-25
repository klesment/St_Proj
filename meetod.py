import streamlit as st

st.header('Rahvastikuprognoos kohort-komponent meetodil')
st.markdown('''
    ### Üldine
Kohort-komponent meetod on 
deterministlik viis rahvastiku vanuselise koosseisu ja suuruse prognoosimiseks. 
Kuna tuleviku sündimus, rändevood ja suremusnäitajad pole teada, tehakse nende kohta eeldused. 
Näiteks suhteliselt madala sündimusega riigi prognoosis võib eeldada, et sündimus jääb samaks, 
väheneb või kasvab. Erinevate eeldustega prognoose nimetatakse stsenaariumiteks. 

Käesolev mudel on tehtud selliselt, et kasutajal on võimalus :

- kas perioodsündimuse tase (TFR e. summaarkordaja) jääb samaks, väheneb või kasvab võrreldes baasaastaga (2019). 
            Kasvu või kahanemist arvutatakse protsendina 2019. aasta tasemega võrreldes. 
- kui kiiresti perioodsündimuse tase kasvab või väheneb.
- kas keskmine ema vanus sünnil tõuseb, langeb või jääb samaks.


            
Lisaks võimaldame kasutajal muuta seda, kui kiiresti sündimus prognoositud perioodi jooksul muutub. 
            

Lihtsuse huvides rakendatakse stsenaariume ainult sündimusele, suremusnäitajad hoitakse konstantsena 
ja rahvusvahelist rännet ei võeta arvesse.


Arvutamisel lähtume 2019. aastast järgmisest teabest:

rahvastiku suurus 1-aastaste vanuserühmade kaupa
vanusepõhised suremusnäitajad (elu tabelist)
vanusespetsiifilised sündimusnäitajad 1-aastaste vanuserühmade kaupa
keskmine vanus sünnihetkel
keskmise sünnivanuse standardhälve

## Prognoosimine

            

    ''')   