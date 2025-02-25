import streamlit as st

st.header('Rahvastikuprognoos kohort-komponent meetodil')
st.markdown('''
    ### Üldine
Kohort-komponent meetod on deterministlik viis rahvastiku vanuselise koosseisu ja suuruse prognoosimiseks. 
Kuna tuleviku sündimus, rändevood ja suremusnäitajad pole teada, tehakse nende kohta eeldused. 
Näiteks suhteliselt madala sündimusega riigi prognoosis võib eeldada, et sündimus jääb samaks, 
väheneb või kasvab. Erinevate eeldustega prognoose nimetatakse stsenaariumiteks. 

Käesolev mudel võimaldab teatud piirides tuleviku sündimuse eeldustega manipuleerida. Näiteks:

- võtta prognoosi aluseks eeldus, et perioodsündimuse tase jääb prognoositaval perioodil samaks, väheneb või kasvab. 
Kasvu või kahanemise eeldus antakse protsentuaalse muutusena (sündimustase perioodi lõpus võrreldes perioodi algusega). 
- eeldada perioodsündimuse taseme kasvu või vähenemise kiirust.
- pakkuda, kas prognoositava perioodi jooksul keskmine ema vanus sünnil tõuseb, langeb või jääb samaks.

Lihtsuse huvides on muud prognoosimudeli komponendid nagu suremusnäitajad  
ja rahvusvaheline ränne muutumatud.


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


### Leslie maatriks
    ''')   
            
st.markdown('''
            Leslie maatriks (https://en.wikipedia.org/wiki/Leslie_matrix) on maatriks, 
            mida kasutatakse elanikkonna struktuuri projitseerimiseks järgmisele ajaperioodile. Maatriksi
            suurus vastab elanikkonna vanuserühmade arvule. See sisaldab nii ellujäämise tõenäosusi kui ka sündimuskordajaid. 
            ''')
         


st.write("""
- Olgu \( x \) veerg, mis esindab elanikkonna vanuselist struktuuri baas-aastal (tähistame seda \( t_0 \) järgi). Veeru elemendid esindavad erinevates vanuserühmades olevaid inimesi (näiteks 0-4 aastat, 5-9 aastat jne).

- Maatriks \( \mathbf{L} \) on Leslie maatriks, mida kasutatakse elanikkonna struktuuri projitseerimiseks järgmisele ajaperioodile. Maatriksi \( \mathbf{L} \) suurus vastab elanikkonna vanuserühmade arvule. See sisaldab nii ellujäämismäärasid kui ka viljakusmäära.

  - Leslie maatriksi \( \mathbf{L} \) elemendid on paigutatud nii, et:
    - \( \mathbf{L} \) esimene rida sisaldab viljakusmäära, see tähendab, kui palju järglasi toodetakse iga vanuserühma isikute poolt.
    - Alamdiagonaalis (peadiagonaali all olevad elemendid) on ellujäämismäärad, mis näitavad, kui tõenäoline on, et iga vanuserühma isikud elavad järgmisesse vanuserühma.
    - Kõik teised elemendid on nullid (ei toimu otsest üleminekut ühelt vanuserühma teise kaugemasse).
""")

st.subheader("Protsess")
st.write("""
1. **Baas-aasta elanikkonna struktuur \( x \)**: See on sinu algne elanikkonna struktuur aastal \( t_0 \), esindatud veeruga \( x \).
   
2. **Leslie maatriksi \( \mathbf{L} \) rakendamine**: Kui vanuseline struktuur \( x \) korrutatakse Leslie maatriksiga \( \mathbf{L} \), on tulemuseks elanikkonna struktuur järgmise ajaperioodi, \( t_0 + 1 \), järgi, mida tähistame kui \( x_t \). Spetsiifiliselt:
   \[
   x_t = \mathbf{L} \cdot x
   \]

3. **Uued vastsündinud**: \( x_t \) esimene element, mis vastab uutele vastsündinutele, arvutatakse fertiilsusmäära (Leslie maatriksi esimene rida) ja vastavate vanuserühmade \( x \) järgi. See on iga vanuserühma poolt toodetud laste kogusumma.

4. **Rahvastiku vananemine**: \( x_t \) teised elemendid vastavad isikutele, kes on vananenud ühe perioodi võrra (näiteks 0-4 aastast 5-9 aastaseks, 5-9 aastast 10-14 aastaseks jne), tuginedes ellujäämismääradele.
""")

st.subheader("Näide")
st.write("""
Eeldame, et meil on lihtne mudel 3 vanuserühmaga (0-4, 5-9, 10-14):

- Baas-aasta elanikkonna vektor \( x \) võib välja näha selline:  
  \[
  x = \begin{pmatrix} 1000 \\ 500 \\ 300 \end{pmatrix}
  \]
  Siin on 1000 isikut 0-4 vanuserühmas, 500 5-9 vanuserühmas ja 300 10-14 vanuserühmas.

- Leslie maatriks \( \mathbf{L} \) võib välja näha selline:
  \[
  \mathbf{L} = \begin{pmatrix} 
  0.2 & 0.3 & 0 \\
  0.5 & 0 & 0 \\
  0 & 0.7 & 0.8
  \end{pmatrix}
  \]
  See maatriks näitab järgmist:
  - Fertiliteedi määrad on 0.2 ja 0.3 esimese ja teise vanuserühma puhul, kolmandal vanuserühmal ei ole viljakuse panust.
  - Ellujäämismäärad on 0.5 esimese ja teise vanuserühma vahel, samuti 0.7 ja 0.8 teise ja kolmanda vanuserühma vahel.

- Kui korrutada maatriks \( \mathbf{L} \) vektoriga \( x \), saame:
  \[
  x_t = \mathbf{L} \cdot x
  \]
""")

st.header("Kood")
st.write("""
Järgmine kood näitab, kuidas seda arvestust Pythonis rakendada, kasutades NumPy raamatukogu:
""")


st.markdown('''
        Kasutame järgmisi tähistusi:
        - $x_0, x_1 ... x_n$ - 1-aastased vanusrühmad, mille vahemik on 0-110.
        - $t_0, t_m$ - prgnoosi kestus $m$ aastates, kui $t_0$ on algusaasta.
        - $f_1, f_2 ... f_x$ - vanuspõhised sündimuskordajad.
        - $P_1, P_2 ... P_x$ - vanuspõhised ellujäämise tõenäosused.
        ''')



st.write('An example of a Leslie matrix with only three age groups:')

st.latex(r'''
\mathbf{L}=%
\begin{pmatrix}
f_{1} & f_{2} & f_{3} \\
P_{1} & 0  & 0 \\
0 & P_{2} & 0%
\end{pmatrix}
''')

#If the age structure of the base year (vector $x$) is multiplied with the matrix $\mathbf{L}$ (which has the same side length as $x$), it results in a new  $t_{0+1}$ shifted age structure $x_t$, in which the first element of the vector is the number of newborns:

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
