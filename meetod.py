import streamlit as st

st.header('Rahvastikuprognoos Leslie maatriksi abil')
st.markdown('''
Prognoosimiseks kasutatav kohort-komponent meetod on deterministlik viis rahvastiku vanuselise koosseisu ja suuruse prognoosimiseks teadaolevate ja 
eeldatud andmete põhjal. Selle aluseks on tõedemus, et inimesi tekib juurde ainult sündides või sisse rännates, 
ning neid jääb vähemaks surres või välja rännates. Teades rahvastiku suurust ja vanuselist koosseisu mingil aastal, saame 
prognoosida rahvastiku suurust ja vanuselist koosseisu järgnevatel aastatel, vastavalt sellele, kuidas eeldatakse sündimuse, 
suremus ja rände muutust. 

Arusaadavalt tuleviku sündimus, rändevood ja suremus ei ole teada, mistõttu on prognoosimisel vaja nende kohta teha eeldused. 

Käesolev mudel võimaldab teatud piirides sündimust puudutavate eeldustega manipuleerida. Näiteks:

- perioodsündimuse muutuse kohta prognoositaval perioodil  
- määrata sündimustaseme muutumise suhtelist kiirust
- keskmise ema vanuse muutuse kohta. 

Lihtsuse huvides on muud rahvastiku koostist mõjutavad komponendid välja jäetud. S.t, ei eeldata suremuse muutust ja oodatava eluea tõusu, 
kuigi see oleks loogiline eeldus. Teiseks, sisse- ja väljaränne, mis võiks omakorda vanuskoostist mõjutada, ei ole arvesse võetud. 

### Arvutuslik pool
Üks võimalus prgnoosis rahvastikumuutuste aastast-aastasse edasikandmiseks on nn Leslie maatriks. 
''')

### Leslie maatriks            
st.markdown('''
            Leslie maatriks (https://en.wikipedia.org/wiki/Leslie_matrix) on maatriks, mille ridade arv vastab 
            kasutatavatele vanuserühmade arvule. Maatriksi esimene rida sisaldab vanuspõhiseid sündimuskordajaid (sündide arv vanusrühmas 
            oleva naisrahvastiku kohta) ja diagonaalil on iga vanusrühma ellujäämise tõenäosus (elutabeli $L_x$). 
            Ülejäänud elemendid on nullid. 
            ''')
         


st.write('Nt kolme vanusrühmaga Leslie maatriks:')

st.latex(r'''
\mathbf{L}=%
\begin{pmatrix}
f_{1} & f_{2} & f_{3} \\
P_{1} & 0  & 0 \\
0 & P_{2} & 0%
\end{pmatrix}
''')


st.markdown('''
            Kui prognoosi aluseks oleva aasta rahvastiku vanusejaotust (vektorit $x$) korrutatakse maatriksiga 
            $\mathbf{L}$ (millel on sama külje pikkus kui $x$-il), siis saadakse uus 
            vanusejaotus $x_t$, mis on nihutatud $t_{0+1}$ võrra. 
            Selle vektori esimene element on vastsündinute arv:
            ''')


st.latex(r'''
x_t = 
\begin{pmatrix}
f_{1} & f_{2} & \cdots & f_{x} \\
P_{1} & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & P_{x-1} & P_x
\end{pmatrix}
\begin{pmatrix}
           x_{1} \\
           x_{2} \\
           \vdots \\
           x_{n}
         \end{pmatrix}
''')


st.latex(r'''
x_t = \mathbf{L}^t x_0
''')


st.write('''
### Alusandmed    
Prognoosimisel võetakse aluseks 2019. aasta seis. Selle aasta kohta on meil vaja teada järgmisi andmeid:

- rahvastiku suurus 1-aastaste vanuserühmade kaupa
- vanusepõhised suremuskordajad (elutabelist)
- vanusespetsiifilised sündimuskordajad 1-aastaste vanuserühmade kaupa
- keskmine ema vanus sünnihetkel ja selle standardhälve

Prognoosis määratakse iga prognoositava aasta vastsündinute arv vanusepõhiste sündimuskordajate (ingl. ASFR) abil. 
ASFR on suhtarv, mille lugeja on teatud vanuses või vanuserühmas olevatele naistele aasta jooksul sündinute arv ja 
nimetaja on samasse rühma kuuluvate naiste arv. ASFR-i summa üle kõigi vanusvahemike annab summaarse sündimuskordaja (TFR), 
mis on iga-aastase sündimustaseme levinum näitaja.
''')    

