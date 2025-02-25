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
            Leslie maatriks (https://en.wikipedia.org/wiki/Leslie_matrix) on 
            rahvastikuprgonoosi meetod, mille sisendiks on vanuspõhised 
            sündimuskordajad ja suremustõenäosused.
            ''')
         

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
