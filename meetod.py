import streamlit as st

st.header('Sündimuse mõju tuleviku rahvastiku suurusele ja vanuskoostisele')

st.markdown('''
Rahvastikuprognoosides kasutatav kohort-komponent meetod on deterministlik viis rahvastiku vanuselise
koosseisu ja suuruse prognoosimiseks teadaolevate andmete ja tehtavate eelduste põhjal. Selle aluseks
on tõdemus, et inimesi tekib juurde ainult sündides või sisse rännates, neid jääb vähemaks surres või
välja rännates. Teades rahvastiku suurust ja vanuselist koosseisu mingil aastal, saame prognoosida
rahvastiku suurust ja vanuselist koosseisu järgnevatel aastatel vastavalt sellele, kuidas eeldatakse
sündimuse, suremuse ja rände muutust.

Arusaadavalt ei ole tuleviku sündimus, suremus ega rändevood teada. Prognoosimiseks on vaja nende
kohta teha eeldusi ehk oletada, mis suunas vastavad arengud võiksid toimuda.

Käesolev prognoosimudel keskendub ainult sündimuse komponendile. Mudel võimaldab teatud piirides
sündimuse muutust puudutavate eeldustega manipuleerida. Näiteks anda prognoosimiseks ette:

- perioodsündimuse langus/tõus prognoositava perioodi jooksul
- sündimustaseme muutumise kiirus, st kas muutus toimub kiiresti või aeglaselt
- ema keskmise sünnivanuse muutus, nt et see jätkab tõusu

Lihtsuse huvides on muud rahvastiku koostist mõjutavad komponendid prognoosist välja jäetud:
suremus on fikseeritud alusaasta tasemel ning sisse- ja väljarännet ei ole arvesse võetud.
''')

st.markdown('''
### Prognoosimismeetod

Üks võimalus prognoosimise tarvis rahvastikumuutuste aastast-aastasse edasikandmiseks on nn Leslie maatriks.
''')

st.markdown('''
### Leslie maatriks

[Leslie maatriks](https://en.wikipedia.org/wiki/Leslie_matrix) on ruutmaatriks, mille suurus vastab
kasutatavate vanuserühmade arvule. Maatriksi esimesele reale kantakse vanuspõhised sündimuskordajad
(sündide arv vanusrühmas oleva naisrahvastiku kohta) ja alamdiagonaalile iga vanusrühma ellujäämise
tõenäosus (elutabeli $L_x$ põhjal). Ülejäänud elemendid on nullid.

Mudel on **naissoost domineeriv** (*female-dominant*): projektsiooni käigus jälgitakse naisrahvastikku
eraldi elutabeli abil. Meesrahvastiku prognoos tuleneb meeste elutabelist ja sündide soosuhtarvust
($1{,}05$ poissi iga tüdruku kohta).
''')

st.write('Nt kolme vanusrühmaga Leslie maatriks:')

st.latex(r'''
\Large{
\mathbf{L}=
\begin{pmatrix}
f_{1} & f_{2} & f_{3} \\
P_{1} & 0  & 0 \\
0 & P_{2} & 0
\end{pmatrix}
}
''')

st.markdown(r'''
Kui prognoosi aluseks oleva aasta naisrahvastiku vanusejaotust (vektorit $x_0$) korrutatakse
maatriksiga $\mathbf{L}$, saadakse uus vanusejaotus $x_{t+1}$. Selle vektori esimene element
on vastsündinud tüdrukute arv:
''')

st.latex(r'''
\Large{
x_{t+1} =
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

st.markdown(r'''
Pikema perioodi prognoos saadaks maatriksi $\mathbf{L}$ astendamisega:
''')

st.latex(r'''
\Large{
x_t = \mathbf{L}^t x_0
}
''')

st.markdown(r'''
See aga tähendab, et sündimuskordajad jäävad kogu perioodi vältel muutumatuks, mis ei ole realistlik.
Antud juhul kasutame järjestikuseid ühe aasta prognoose nii, et sündimuskordajad muutuvad igal aastal —
igal aastal arvutatakse uus Leslie maatriks, mis vastab selle aasta eeldatavatele sündimuskordajatele.
Suremus on fikseeritud 2023. aasta tasemel.
''')

st.markdown(r'''
Kuna Leslie maatriksis kasutatakse vanusepõhiseid sündimuskordajaid (ASFR), aga kasutaja määrab
summaarse sündimuskordaja (TFR), tuleb TFR esmalt teisendada vanuspõhisteks kordajateks. Selleks
kasutame Gamma-mudelit (vt allpool). TFR muutus toimub Gompertzi S-kõvera kujuliselt — muutus
kiireneb esmalt ja aeglustub perioodi lõpuks. Muutuse kiiruse saab kasutaja ise valida.

Aastate 2024 ja 2025 tegelikud TFR väärtused (mõlemal 1,18) on mudelisse fikseeritud.
Kasutaja valitud stsenaarium rakendub alates 2026. aastast.
''')

st.markdown('''
### Gamma-mudel vanusepõhise sündimuskordaja (ASFR) arvutamiseks

Vanusepõhiste sündimuskordajate modelleerimiseks kasutame Gamma-mudelit. Sellel on kolm parameetrit:

- summaarne sündimuskordaja (TFR)
- keskmine sünnitusvanus (MAB)
- keskmise sünnitusvanuse standardhälve (sdMAB)

Gamma-mudel on defineeritud järgmiselt:
''')

st.latex(r"""
\Large{
F_x = \frac{1}{\Gamma(\alpha_3)} \alpha_1 \alpha_2^{\alpha_3} x^{\alpha_3 - 1} \exp(-\alpha_2 x)
}
""")

st.markdown(r"""
kus parameetrid on defineeritud järgmiselt:
""")

st.latex(r"\alpha_1 = TFR")
st.latex(r"\alpha_2 = \frac{MAB}{sdMAB^2}")
st.latex(r"\alpha_3 = \left(\frac{MAB}{sdMAB}\right)^2")

st.write("Nendest kahte parameetrit (TFR, MAB) saab kasutaja prognoosi tegemisel muuta.")

st.markdown('''
### Alusandmed

Prognoosimisel võetakse aluseks 2023. aasta seis. Selle aasta kohta on mudelis kasutatud järgmisi andmeid:

- rahvastiku suurus 1-aastaste vanuserühmade kaupa (eraldi mehed ja naised)
- naiste vanusepõhised suremuskordajad (elutabelist)
- meeste vanusepõhised suremuskordajad (elutabelist)
- vanusespetsiifilised sündimuskordajad 1-aastaste vanuserühmade kaupa
- keskmine ema vanus sünnihetkel ja selle standardhälve

Andmeallikad on toodud lehel *Andmeallikad*.
''')
