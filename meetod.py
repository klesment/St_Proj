# v2 — baseline inflow by nationality
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st

from projection import URL_MT_STOCK, URL_EMIG_RATES, URL_IMMIG_BASELINE


@st.cache_data
def _load_migration_rate_data():
    def fetch(url):
        return pd.read_csv(io.StringIO(requests.get(url).content.decode()))

    emig     = fetch(URL_EMIG_RATES)
    baseline = fetch(URL_IMMIG_BASELINE)
    pop      = fetch(URL_MT_STOCK)
    return emig, baseline, pop

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

Käesolev prognoosimudel võimaldab teatud piirides manipuleerida sündimuse, suremuse ja rände
eeldustega. Kasutaja saab anda prognoosimiseks ette:

- perioodsündimuse langus/tõus prognoositava perioodi jooksul
- sündimustaseme muutumise kiirus, st kas muutus toimub kiiresti või aeglaselt
- ema keskmise sünnivanuse muutus, nt et see jätkab tõusu
- suremuse aastane langus (%)
- aastane lisasisseränne (muu emakeel)
- baassisserände ja baasväljarände maht (0/50/100% baastasemest)

Sisse- ja väljaränne on mudelis arvesse võetud (vt allpool).
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
igal aastal arvutatakse uus Leslie maatriks, mis vastab selle aasta eeldatavatele sündimuskordajatele
ja suremustasemele.
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

st.markdown(r'''
### Suremuse muutus

Suremuse stsenaarium on määratud ühe parameetriga: **suremuse aastane langus** (%). See tähendab, et
iga vanusrühma suremuskordaja $m_x$ langeb igal prognoosiaastal sama protsendi võrra:

$$m_x(t) = m_x(0) \cdot (1 - r)^t$$

kus $r$ on kasutaja valitud aastane langumäär ja $t$ on prognoosiaasta number. Langus on seega
**geomeetriline**: absoluutne muutus väheneb iga aastaga, kuna protsent rakendub juba langenud
tasemele. See on kooskõlas tegeliku suremuse arenguga — mida madalam on juba saavutatud suremustase,
seda raskem on seda edasi vähendada.

Igal prognoosiaastal arvutatakse muudetud $m_x$ põhjal uus elutabel ($q_x$, $l_x$, $L_x$, $T_x$)
ning selle põhjal uus Leslie maatriks. Nii naiste kui meeste suremus langeb sama määraga.

Kui langumäär on 0%, jääb suremus kogu perioodi vältel 2023. aasta tasemele.
''')

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

st.markdown('''
### Rändekomponent

Mudel eristab kaht rahvastikurühma **emakeele** alusel: eesti emakeelega ja muu emakeelega rahvastik.

**Algjaotus.** Rühmade algsuurus tuleneb 2021. aasta rahvaloendusest (tabel RL21434). Loenduse
emakeele proportsioonid kantakse üle 2023. aasta rahvastikuandmetele, nii et mõlema rühma suurus
on kooskõlas 2023. aasta kogurahvastikuga.

**Laste emakeel** järgib ema emakeelt: eesti emakeelega emade lapsed lisatakse eesti emakeelega
rühma ja muu emakeelega emade lapsed muu emakeelega rühma.

**Väljaränne** on modelleeritud vanuse- ja soospeciifiliste rändemääradena, mis on arvutatud
aastate 2017–2019 keskmisena. Eesti emakeelega rahvastiku rändemäärad tulenevad tabelitest RVR03
(koguränne) ja RVR10 (eestlaste ränne); muu emakeelega rahvastiku rändemäärad on nende vahe.
Nimetajana kasutatakse 2021. aasta rahvaloenduse andmeid. Väljarändemäärad skaleeruvad
automaatselt rahvastiku suurusega — mida suurem rahvastik, seda rohkem lahkujaid.

**Sisseränne** koosneb kahest komponendist:

- **Baassisseränne** (kasutaja valib 0/50/100%): 2017–2019 keskmised aastased saabujad rahvuse
  kaupa (RVR03 ja RVR10 põhjal). Eesti kodanikud (tagasirändajad) lisatakse eesti emakeelega
  rühma, muu kodakondsusega isikud muu emakeelega rühma. Vaikimisi 100%.
- **Lisaränne** (kasutaja valib): täiendav aastane muu emakeelega sisserände maht baastaseme peal.
  Vanusejaotus tuleneb 2017–2019 välismaalt saabunute keskmisest vanusjaotusest (tabel RVR09,
  v.a eestlaste tagasiränne). Soosuhe on fikseeritud vaadeldud perioodi keskmise põhjal
  (39% naised, 61% mehed).

**Baasväljaränne** on samuti kasutaja poolt skaleeritav (0/50/100%). Vaikimisi rakenduvad täies
mahus 2017–2019 keskmised vanuse- ja soospeciifilised väljarändemäärad. 0% tähendab, et kedagi
ei lahku; 50% poolitab kõik väljarändemäärad.

Iga prognoosiaasta järjekord on järgmine:

```
AASTA t TSÜKKEL
────────────────────────────────────────────────────────────────

  AASTA t ALGUS
  ┌─────────────────┐   ┌─────────────────┐
  │  N_nat_f / m    │   │  N_imm_f / m    │
  └────────┬────────┘   └────────┬────────┘
           │                     │
           ▼  1. ELLUJÄÄMINE + SÜNDID
  ┌─────────────────┐   ┌─────────────────┐
  │ Leslie (naised) │   │ Leslie (naised) │  ← muu emakeelega emade
  │ subd_male (m)   │   │ subd_male (m)   │    lapsed → muu rühma
  └────────┬────────┘   └────────┬────────┘
           │                     │
           ▼  2. VÄLJARÄNNE  (määrad × ellujäänud rahvastik)
  ┌─────────────────┐   ┌─────────────────┐
  │ × (1−emig_nat)  │   │ × (1−emig_imm)  │
  └────────┬────────┘   └────────┬────────┘
           │                     │
           ▼  3. SISSERÄNNE
  ┌─────────────────┐   ┌────────────────────────────────────┐
  │ + baas_nat_f/m  │   │ + baas_imm_f/m                     │
  │  (fikseeritud)  │   │ + lisaränne_f/m  ← kasutaja valib  │
  └────────┬────────┘   └────────┬───────────────────────────┘
           │                     │
           ▼                     ▼
  AASTA t LÕPP  →  läheb aasta t+1 alguseks


NETORÄNNE (kuvatav näitaja)
────────────────────────────────────────────────────────────────

  sisseränne  = baas_nat + baas_imm + lisaränne
  väljaränne  = rändemäärad × AASTA LÕPU rahvastik   ← sisaldab
  netoränne   = sisseränne − väljaränne                  saabujaid
```

NB: tsüklis rakendatakse väljaränne enne sisserännet (samm 2 enne samm 3),
kuid kuvatav netoränne arvutab väljarände perioodi lõpu rahvastiku põhjal
(mis juba sisaldab saabujaid). Seega on kuvatav väljaränne väikeses osas
ülehinnatud — 1000 lisarändajaga näiteks umbes 3 inimese võrra.

Kuvatavad näitajad:

- *Aastane netoränne* — kogusisseränne (baas + lisaränne) miinus väljarändemäärad, mis on
  rakendatud prognoosiperioodi lõpu rahvastikule. Peegeldab, kuidas rahvastiku suuruse ja
  vanuselise koosseisu muutumine mõjutab väljarände absoluutmahtu.
- *Muu emakeel sisserändest* — muu emakeelega sisserändajate osakaal kogusisserändest
  (baas + lisaränne). Näitab alati väärtust vahemikus 0–100%.
- *Vanus 0–17 / 18–64 / 65+* — vastava vanuserühma osakaal kogurahvastikust prognoosiperioodi
  lõpus. Muutus on näidatud alusaasta suhtes.
- *Muu emakeel 0–17 / 18–64 / 65+* — muu emakeelega elanike osakaal igas vanuserühmas
  prognoosiperioodi lõpus. Muutus on näidatud alusaasta suhtes.
''')

try:
    emig, baseline, pop = _load_migration_rate_data()

    age_labels = emig['age_group'].tolist()
    ages = np.arange(len(emig))

    # Convert baseline inflow counts to rates using 2021 census population
    inflow_rates = {}
    for group, sex in [('estonian', 'male'), ('estonian', 'female'),
                       ('other',    'male'), ('other',    'female')]:
        denom = pop[f'{group}_{sex}'].replace(0, np.nan)
        inflow_rates[f'{group}_{sex}'] = baseline[f'{group}_{sex}'] / denom

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    fig.suptitle('Aastased rändemäärad vanuse järgi  (2017–2019 keskmine)', fontsize=12)

    panels = [
        ('estonian', 'male',   'Eesti emakeel — mehed',   axes[0, 0]),
        ('estonian', 'female', 'Eesti emakeel — naised',  axes[0, 1]),
        ('other',    'male',   'Muu emakeel — mehed',     axes[1, 0]),
        ('other',    'female', 'Muu emakeel — naised',    axes[1, 1]),
    ]

    for group, sex, title, ax in panels:
        col = f'{group}_{sex}'
        ax.plot(ages, emig[f'{col}_rate'].values * 100,
                color='#c0392b', marker='o', markersize=3, label='Väljaränne')
        ax.plot(ages, inflow_rates[col].values * 100,
                color='#2980b9', marker='s', markersize=3, label='Sisseränne')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ages)
        ax.set_xticklabels(age_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Määr (%)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
except Exception:
    pass
