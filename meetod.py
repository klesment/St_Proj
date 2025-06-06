import streamlit as st

st.header('Sündimuse mõju tuleviku rahvastiku suurusele ja vanuskoostisele')
st.markdown('''
Rahvastikuprognoosides kasutatav kohort-komponent meetod on deterministlik viis rahvastiku vanuselise koosseisu ja suuruse prognoosimiseks teadaolevate 
andmete ja tehtavate eelduste põhjal. Selle aluseks on tõedemus, et inimesi tekib juurde ainult sündides või sisse rännates, 
neid jääb vähemaks surres või välja rännates. Teades rahvastiku suurust ja vanuselist koosseisu mingil aastal, saame 
prognoosida rahvastiku suurust ja vanuselist koosseisu järgnevatel aastatel, vastavalt sellele, kuidas eeldatakse sündimuse, 
suremus ja rände muutust järgnevatel aastatel. 

Arusaadavalt tuleviku sündimus,  suremus ja rändevood ei ole teada. Prognoosimiseks on vaja nende kohta teha eeldusi 
            ehk oletada, mis suunas vastavad arengud võiksid toimuda. 

Käesolev prognoosimudel keskendub ainult sündimuse komponendile. 
Mudel võimaldab teatud piirides sündimuse muutust puudutavate eeldustega manipuleerida. Näiteks anda prognoosimiseks ette:

- perioodsündimuse langus/tõus prognoositava perioodi jooksul
- sündimustaseme muutumise kiirus, st kas muutus toimub kiiresti või aeglaselt
- ema keskmise sünnivanuse muutus, nt et see jätkab tõusu. 

Lihtsuse huvides on muud rahvastiku koostist mõjutavad komponendid prognoosist välja jäetud. St, ei eeldata suremuse muutust ja 
oodatava eluea tõusu, kuigi see oleks loogiline eeldus. Teiseks, sisse- ja väljaränne, mis võiks omakorda vanuskoostist mõjutada, 
ei ole arvesse võetud. 

### Prognoosimismeetod 
Üks võimalus prgnoosimise tarvis rahvastikumuutuste aastast-aastasse edasikandmiseks on nn Leslie maatriks. 
''')

### Leslie maatriks            
st.markdown('''
            Leslie maatriks (https://en.wikipedia.org/wiki/Leslie_matrix) on maatriks, mille ridade arv vastab 
            kasutatavatele vanuserühmade arvule. Maatriksi esimesele reale kantakse vanuspõhiseid sündimuskordajaid (sündide arv vanusrühmas 
            oleva naisrahvastiku kohta) ja nn alamdiagonaalile iga vanusrühma ellujäämise tõenäosus (elutabeli $L_x$). 
            Ülejäänud elemendid on nullid. 
            ''')
         


st.write('Nt kolme vanusrühmaga Leslie maatriks:')

st.latex(r'''
         \Large{     
\mathbf{L}=%
\begin{pmatrix}
f_{1} & f_{2} & f_{3} \\
P_{1} & 0  & 0 \\
0 & P_{2} & 0%
\end{pmatrix}
         }
''')


st.markdown('''
            Kui prognoosi aluseks oleva aasta rahvastiku vanusejaotust (vektorit $x$) korrutatakse maatriksiga 
            $\mathbf{L}$ (millel on sama külje pikkus kui $x$-il), siis saadakse uus 
            vanusejaotus $x_t$, mis on nihutatud $t_{0+1}$ võrra. 
            Selle vektori esimene element on vastsündinute arv:
            ''')


st.latex(r'''
         \Large{     
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
         }
''')


st.markdown('''Sellist prognoosi saab teha ka pikema perioodi peale kui üks aasta. Sellisel juhul saab prognoositava perioodi 
            pikkusest maatriksi $\mathbf{L}$ astendaja, mis siis korrutatakse baasaasta vanusstruktuuriga: 
            ''')

st.latex(r'''
\Large{          
x_t = \mathbf{L}^t x_0
         }
''')


st.markdown('''
            seejuures $x_0$ on prognoosi aluseks oleva aasta rahvastiku vanusejaotus. 
            See aga tähendab, et prognoositava perioodi jooksul ei muutu sündimuskordajad, mis ei ole tõenäoliselt realistlik.
                  Antud juhul kasutame järjestikuseid ühe aasta prognoose nii, et sündimuskordajad muutuvad igal aastal. 
                  See tähendab, et igal aastal arvutatakse uus Leslie maatriks, mis vastab selle aasta eeldatavatele sündimuskordajatele ja 
                  ellujäämise tõenäosustele. See võimaldab prognoosis sündimust dünaamilisemalt käsitleda.
                  ''')

st.markdown('''Seejuures tekib probleem, et summmaarse sündimuskordaja (TFR) asemel on Leslie maatriksis vaja kasutada
            vanusepõhiseid sündimuskordajaid (ASFR), samas kui TFR on ASFR-i summa üle kõigi vanusvahemike. Järelikult peame eeldatud tuleviku TFR-i
            kõigepealt teisendama vanuspõhisteks sündimuskordajateks.  
            ''')



### Gamma-mudel vanusepõhise sündimuskordaja (ASFR) arvutamiseks
st.write('''
Vanusepõhiste sündimuskordajate modelleerimiseks kasutame nn Gamma-mudelit. Sellel on kolm parameetrit:
- summaarne sündimuskordaja (TFR)
- Keskmine sünnitusvanus (MAB)
- Keskmise sünnitusvanuse standardhälve (sdMAB)
         
Gamma-mudel ja selle parameetrid on defineeritud järgmiselt:)
''')

st.latex(r"""
         \Large{     
F_x = \frac{1}{\Gamma(\alpha_3)} \alpha_1 \alpha_2^{\alpha_3} (x - \alpha_4)^{\alpha_3 - 1} \exp(-\alpha_2(x - \alpha_4)); \quad x \geq \alpha_4
    }
         """)


st.markdown(r""" 
         Praktikas määratakse $\alpha_4 = 0$, ja ülejäänud parameetrid on defineeritud järgmiselt:
         """)


st.latex(r"\alpha_1 = TFR")
st.latex(r"\alpha_2 = \frac{MAB}{sdMAB^2}")
st.latex(r"\alpha_3 = \left(\frac{MAB}{sdMAB}\right)^2")

st.write("Nendest kahte parameetrit (TFR, MAB) saab kasutaja prognoosi tegemisel muuta.")


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

