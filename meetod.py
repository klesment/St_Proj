# v2 — baseline inflow by nationality
# © 2026 Martin Klesment. Licensed under CC BY-NC 4.0.
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

lang = st.radio("Language", ["ET", "EN"], horizontal=True, label_visibility="hidden")

# --- Header ---
if lang == 'ET':
    st.header('Interaktiivne rahvastikuprognoos')
else:
    st.header('Interactive Population Projection')

# --- Intro ---
if lang == 'ET':
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
else:
    st.markdown('''
The cohort-component method used in population projections is a deterministic approach to projecting
the size and age structure of a population based on known data and assumptions. It rests on the
observation that the population grows only through births or immigration, and decreases through
deaths or emigration. Knowing the size and age structure of a population in a given year, we can
project its size and age structure in subsequent years according to assumed changes in fertility,
mortality and migration.

Future fertility, mortality and migration flows are of course unknown. Projecting them requires
assumptions — educated guesses about the direction these developments might take.

This projection model allows the user to vary fertility, mortality and migration assumptions within
defined limits. The user can specify:

- increase or decrease in period fertility over the projection period
- speed of the fertility change — whether it happens quickly or gradually
- change in mean age at birth, e.g. a continued rise
- annual rate of mortality decline (%)
- annual additional immigration (non-Estonian mother tongue)
- baseline immigration and emigration levels (0/50/100% of baseline)

Immigration and emigration are accounted for in the model (see below).
''')

if lang == 'ET':
    st.markdown('Üks võimalus prognoosimise tarvis rahvastikumuutuste aastast-aastasse edasikandmiseks on nn Leslie maatriks.')
else:
    st.markdown('One way to carry population changes forward year by year for projection purposes is the Leslie matrix.')

# --- Leslie matrix ---
if lang == 'ET':
    st.markdown(r'''
### Leslie maatriks

[Leslie maatriks](https://en.wikipedia.org/wiki/Leslie_matrix) on ruutmaatriks, mille suurus vastab
kasutatavate vanuserühmade arvule. Maatriksi esimesele reale kantakse vanuspõhised sündimuskordajad
(sündide arv vanusrühmas oleva naisrahvastiku kohta) ja alamdiagonaalile iga vanusrühma ellujäämise
tõenäosus (elutabeli $L_x$ põhjal). Ülejäänud elemendid on nullid.

Mudel on **naissoost domineeriv** (*female-dominant*): projektsiooni käigus jälgitakse naisrahvastikku
eraldi elutabeli abil. Meesrahvastiku prognoos tuleneb meeste elutabelist ja sündide soosuhtarvust
($1{,}05$ poissi iga tüdruku kohta).
''')
else:
    st.markdown(r'''
### Leslie matrix

The [Leslie matrix](https://en.wikipedia.org/wiki/Leslie_matrix) is a square matrix whose size equals
the number of age groups used. The first row contains age-specific fertility rates (births per woman
in each age group) and the sub-diagonal contains the survival probability for each age group (derived
from the life table $L_x$). All remaining elements are zero.

The model is **female-dominant**: the female population is tracked separately using its own life table.
The male population projection follows from the male life table and the sex ratio at birth
($1{.}05$ boys per girl).
''')

if lang == 'ET':
    st.write('Nt kolme vanusrühmaga Leslie maatriks:')
else:
    st.write('Example Leslie matrix with three age groups:')

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

if lang == 'ET':
    st.markdown(r'''
Kui prognoosi aluseks oleva aasta naisrahvastiku vanusejaotust (vektorit $x_0$) korrutatakse
maatriksiga $\mathbf{L}$, saadakse uus vanusejaotus $x_{t+1}$. Selle vektori esimene element
on vastsündinud tüdrukute arv:
''')
else:
    st.markdown(r'''
Multiplying the age distribution vector $x_0$ of the base-year female population by the matrix
$\mathbf{L}$ gives the new age distribution $x_{t+1}$. The first element of this vector is the
number of newborn girls:
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

if lang == 'ET':
    st.markdown(r'Pikema perioodi prognoos saadaks maatriksi $\mathbf{L}$ astendamisega:')
else:
    st.markdown(r'A longer-period projection would be obtained by raising the matrix $\mathbf{L}$ to a power:')

st.latex(r'''
\Large{
x_t = \mathbf{L}^t x_0
}
''')

if lang == 'ET':
    st.markdown(r'''
See aga tähendab, et sündimuskordajad jäävad kogu perioodi vältel muutumatuks, mis ei ole realistlik.
Antud juhul kasutame järjestikuseid ühe aasta prognoose nii, et sündimuskordajad muutuvad igal aastal —
igal aastal arvutatakse uus Leslie maatriks, mis vastab selle aasta eeldatavatele sündimuskordajatele
ja suremustasemele.
''')
else:
    st.markdown(r'''
This however implies that fertility rates remain unchanged throughout the period, which is unrealistic.
In this model we use successive one-year projections so that fertility rates change each year — a new
Leslie matrix is computed each year to reflect the assumed fertility rates and mortality level for that year.
''')

if lang == 'ET':
    st.markdown(r'''
Kuna Leslie maatriksis kasutatakse vanusepõhiseid sündimuskordajaid (ASFR), aga kasutaja määrab
summaarse sündimuskordaja (TFR), tuleb TFR esmalt teisendada vanuspõhisteks kordajateks. Selleks
kasutame Gamma-mudelit (vt allpool). TFR muutus toimub Gompertzi S-kõvera kujuliselt — muutus
kiireneb esmalt ja aeglustub perioodi lõpuks. Muutuse kiiruse saab kasutaja ise valida.

Aastate 2024 ja 2025 tegelikud TFR väärtused (mõlemal 1,18) on mudelisse fikseeritud.
Kasutaja valitud stsenaarium rakendub alates 2026. aastast.

Lisaks saab kasutaja aktiveerida **TFR kõikumise** platoo ümber: sinusoidaalne komponent
ilmub pärast platoo saavutamist ja on moduleeritud Gompertzi kõveraga nii, et kõikumine
on rampimise ajal summutatud ning avaldub täies mahus alles siis, kui TFR on sihtnivoole
jõudnud. Kasutaja saab valida kõikumise amplituudi (TFR ühikutes) ja perioodi (aastates).
''')
else:
    st.markdown(r'''
Since the Leslie matrix uses age-specific fertility rates (ASFR) but the user sets the total fertility
rate (TFR), TFR must first be converted to age-specific rates. This is done using the Gamma model
(see below). The TFR change follows a Gompertz S-curve — the change initially accelerates and then
slows towards the end of the period. The speed of change can be selected by the user.

The observed TFR values for 2024 and 2025 (both 1.18) are fixed in the model.
The user-selected scenario applies from 2026 onwards.

The user can also activate **TFR oscillation** around the plateau: a sinusoidal component appears
after the plateau is reached, modulated by a Gompertz curve so that oscillation is dampened during
the ramp phase and reaches full amplitude only once TFR has converged to the target level. The user
can set the oscillation amplitude (in TFR units) and period (in years).
''')

# --- Gamma model ---
if lang == 'ET':
    st.markdown('''
### Gamma-mudel vanusepõhise sündimuskordaja (ASFR) arvutamiseks

Vanusepõhiste sündimuskordajate modelleerimiseks kasutame Gamma-mudelit. Sellel on kolm parameetrit:

- summaarne sündimuskordaja (TFR)
- keskmine sünnitusvanus (MAB)
- keskmise sünnitusvanuse standardhälve (sdMAB)

Gamma-mudel on defineeritud järgmiselt:
''')
else:
    st.markdown('''
### Gamma model for age-specific fertility rates (ASFR)

To model age-specific fertility rates we use the Gamma model. It has three parameters:

- total fertility rate (TFR)
- mean age at birth (MAB)
- standard deviation of age at birth (sdMAB)

The Gamma model is defined as follows:
''')

st.latex(r"""
\Large{
F_x = \frac{1}{\Gamma(\alpha_3)} \alpha_1 \alpha_2^{\alpha_3} x^{\alpha_3 - 1} \exp(-\alpha_2 x)
}
""")

if lang == 'ET':
    st.markdown(r"kus parameetrid on defineeritud järgmiselt:")
else:
    st.markdown(r"where the parameters are defined as follows:")

st.latex(r"\alpha_1 = TFR")
st.latex(r"\alpha_2 = \frac{MAB}{sdMAB^2}")
st.latex(r"\alpha_3 = \left(\frac{MAB}{sdMAB}\right)^2")

if lang == 'ET':
    st.write("Nendest kahte parameetrit (TFR, MAB) saab kasutaja prognoosi tegemisel muuta.")
else:
    st.write("Two of these parameters (TFR, MAB) can be adjusted by the user when setting up the projection.")

# --- Mortality ---
if lang == 'ET':
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
else:
    st.markdown(r'''
### Mortality change

The mortality scenario is governed by a single parameter: the **annual rate of mortality decline** (%).
This means that the mortality rate $m_x$ for each age group falls by the same percentage each
projection year:

$$m_x(t) = m_x(0) \cdot (1 - r)^t$$

where $r$ is the user-selected annual decline rate and $t$ is the projection year number. The decline
is therefore **geometric**: the absolute change diminishes each year as the percentage is applied to
an already-reduced level. This is consistent with observed mortality trends — the lower the mortality
level already achieved, the harder it is to reduce it further.

Each projection year, a new life table ($q_x$, $l_x$, $L_x$, $T_x$) is computed from the adjusted
$m_x$, and from it a new Leslie matrix. Both female and male mortality decline at the same rate.

If the decline rate is 0%, mortality remains at the 2023 level throughout the period.
''')

# --- Base data ---
if lang == 'ET':
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
else:
    st.markdown('''
### Base data

The projection uses 2023 as the base year. The following data are used for that year:

- population size by single-year age groups (men and women separately)
- female age-specific mortality rates (from the life table)
- male age-specific mortality rates (from the life table)
- age-specific fertility rates by single-year age groups
- mean age of mother at birth and its standard deviation

Data sources are listed on the *Data sources* page.
''')

# --- Migration component ---
if lang == 'ET':
    st.markdown('''
### Rändekomponent

Mudel eristab kaht rahvastikurühma **emakeele** alusel: eesti emakeelega ja muu emakeelega rahvastik.

**Algjaotus.** Rühmade algsuurus tuleneb 2021. aasta rahvaloendusest (tabel RL21434). Loenduse
emakeele proportsioonid kantakse üle 2023. aasta rahvastikuandmetele, nii et mõlema rühma suurus
on kooskõlas 2023. aasta kogurahvastikuga.

**Laste emakeel** järgib ema emakeelt: eesti emakeelega emade lapsed lisatakse eesti emakeelega
rühma ja muu emakeelega emade lapsed muu emakeelega rühma.

**Väljaränne** on modelleeritud vanuse- ja soospetsiifiliste rändemääradena, mis on arvutatud
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
mahus 2017–2019 keskmised vanuse- ja soospetsiifilised väljarändemäärad. 0% tähendab, et kedagi
ei lahku; 50% poolitab kõik väljarändemäärad.

Iga prognoosiaasta järjekord on järgmine:
''')
else:
    st.markdown('''
### Migration component

The model distinguishes two population groups by **mother tongue**: Estonian mother tongue and other
mother tongue.

**Initial distribution.** The initial size of each group is derived from the 2021 census (table
RL21434). The census mother-tongue proportions are applied to the 2023 population data so that both
groups are consistent with the 2023 total population.

**Children's mother tongue** follows the mother's: children of Estonian mother-tongue mothers are
added to the Estonian group, and children of other mother-tongue mothers to the other group.

**Emigration** is modelled as age- and sex-specific emigration rates computed as the 2017–2019
average. Estonian mother-tongue rates are derived from tables RVR03 (total migration) and RVR10
(Estonian migration); other mother-tongue rates are the difference. The 2021 census population is
used as the denominator. Emigration rates scale automatically with population size — the larger the
population, the more people leave.

**Immigration** consists of two components:

- **Baseline immigration** (user selects 0/50/100%): average annual arrivals by nationality for
  2017–2019 (based on RVR03 and RVR10). Estonian citizens (returnees) are added to the Estonian
  group, other nationals to the other group. Default is 100%.
- **Additional immigration** (user selects): extra annual non-Estonian mother-tongue immigration
  on top of the baseline. The age distribution is based on the average age distribution of arrivals
  from abroad in 2017–2019 (table RVR09, excluding Estonian returnees). The sex ratio is fixed at
  the observed period average (39% female, 61% male).

**Baseline emigration** is also scalable by the user (0/50/100%). By default, the full 2017–2019
average age- and sex-specific emigration rates apply. 0% means nobody leaves; 50% halves all
emigration rates.

The order of operations within each projection year is as follows:
''')

if lang == 'ET':
    st.markdown('''
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
''')
else:
    st.markdown('''
```
YEAR t CYCLE
────────────────────────────────────────────────────────────────

  START OF YEAR t
  ┌─────────────────┐   ┌─────────────────┐
  │  N_nat_f / m    │   │  N_imm_f / m    │
  └────────┬────────┘   └────────┬────────┘
           │                     │
           ▼  1. SURVIVAL + BIRTHS
  ┌─────────────────┐   ┌─────────────────┐
  │ Leslie (female) │   │ Leslie (female) │  ← children of other
  │ subd_male (m)   │   │ subd_male (m)   │    MT mothers → other group
  └────────┬────────┘   └────────┬────────┘
           │                     │
           ▼  2. EMIGRATION  (rates × surviving population)
  ┌─────────────────┐   ┌─────────────────┐
  │ × (1−emig_nat)  │   │ × (1−emig_imm)  │
  └────────┬────────┘   └────────┬────────┘
           │                     │
           ▼  3. IMMIGRATION
  ┌─────────────────┐   ┌────────────────────────────────────┐
  │ + base_nat_f/m  │   │ + base_imm_f/m                     │
  │    (fixed)      │   │ + extra_immig_f/m  ← user selects  │
  └────────┬────────┘   └────────┬───────────────────────────┘
           │                     │
           ▼                     ▼
  END OF YEAR t  →  becomes start of year t+1


NET MIGRATION (displayed indicator)
────────────────────────────────────────────────────────────────

  immigration   = base_nat + base_imm + extra_immig
  emigration    = rates × END-OF-YEAR population   ← includes
  net migration = immigration − emigration             arrivals
```

Note: in the cycle, emigration is applied before immigration (step 2 before step 3), but the
displayed net migration computes emigration based on the end-of-period population (which already
includes arrivals). The displayed emigration is therefore slightly overstated — by around 3 persons
for every 1,000 additional immigrants.
''')

if lang == 'ET':
    st.markdown('''
Kuvatavad näitajad:

- *Rahvaarv kokku* — kogurahvastik prognoosiperioodi lõpus.
- *Muu emakeel kokku* — muu emakeelega elanike arv prognoosiperioodi lõpus.
- *Sündide arv* — vastsündinute arv prognoosiperioodi lõpuaastal.
- *Surmade arv* — hinnanguline surmade arv prognoosiperioodi lõpuaastal. Arvutatakse
  lõpuaasta rahvastiku ja vanusespetsiifiliste suremuskordajate ($q_x$) korrutiste summana.
  Kui suremuse langus on seadistatud, kasutatakse vastavalt kohandatud elutabelit.
- *Aastane netoränne* — kogusisseränne (baas + lisaränne) miinus väljarändemäärad, mis on
  rakendatud prognoosiperioodi lõpu rahvastikule. Peegeldab, kuidas rahvastiku suuruse ja
  vanuselise koosseisu muutumine mõjutab väljarände absoluutmahtu.
- *Eestlaste osa sisserändest* — eesti emakeelega sisserändajate osakaal kogusisserändest
  (baas + lisaränne). Näitab alati väärtust vahemikus 0–100%.
- *Prognoositud vanusrühmade osakaal kogurahvastikus* — vanuserühmade 0–17, 18–64 ja 65+
  osakaal kogurahvastikust prognoosiperioodi lõpus. Muutus on näidatud alusaasta suhtes.
- *sh eesti emakeel* — eesti emakeelega elanike osakaal igas vanuserühmas prognoosiperioodi
  lõpus. Muutus on näidatud alusaasta suhtes.
''')
else:
    st.markdown('''
Displayed indicators:

- *Total population* — total population at the end of the projection period.
- *Other mother tongue total* — number of residents with a non-Estonian mother tongue at the end of the projection period.
- *Births* — number of births in the final projection year.
- *Deaths* — estimated number of deaths in the final projection year. Computed as the sum of
  end-year population × age-specific mortality rates ($q_x$). If mortality decline is set, the
  adjusted life table is used.
- *Annual net migration* — total immigration (baseline + additional) minus emigration rates applied
  to the end-of-period population. Reflects how changes in population size and age structure affect
  the absolute volume of emigration.
- *Estonian share of immigration* — share of Estonian mother-tongue immigrants in total immigration
  (baseline + additional). Always between 0–100%.
- *Projected age group shares in total population* — shares of age groups 0–17, 18–64 and 65+ in
  the total population at the end of the projection period. Change shown relative to the base year.
- *of which Estonian mother tongue* — share of Estonian mother-tongue residents in each age group
  at the end of the projection period. Change shown relative to the base year.
''')

# --- Migration rate charts ---
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

    if lang == 'ET':
        chart_title  = 'Aastased rändemäärad vanuse järgi  (2017–2019 keskmine)'
        panels_labels = [
            ('estonian', 'male',   'Eesti emakeel — mehed'),
            ('estonian', 'female', 'Eesti emakeel — naised'),
            ('other',    'male',   'Muu emakeel — mehed'),
            ('other',    'female', 'Muu emakeel — naised'),
        ]
        label_emig = 'Väljaränne'
        label_immig = 'Sisseränne'
        ylabel = 'Määr (%)'
    else:
        chart_title  = 'Annual migration rates by age  (2017–2019 average)'
        panels_labels = [
            ('estonian', 'male',   'Estonian mother tongue — men'),
            ('estonian', 'female', 'Estonian mother tongue — women'),
            ('other',    'male',   'Other mother tongue — men'),
            ('other',    'female', 'Other mother tongue — women'),
        ]
        label_emig = 'Emigration'
        label_immig = 'Immigration'
        ylabel = 'Rate (%)'

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    fig.suptitle(chart_title, fontsize=12)

    for (group, sex, title), ax in zip(panels_labels, axes.flatten()):
        col = f'{group}_{sex}'
        ax.plot(ages, emig[f'{col}_rate'].values * 100,
                color='#c0392b', marker='o', markersize=3, label=label_emig)
        ax.plot(ages, inflow_rates[col].values * 100,
                color='#2980b9', marker='s', markersize=3, label=label_immig)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ages)
        ax.set_xticklabels(age_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
except Exception:
    pass

# --- Data sources ---
if lang == 'ET':
    st.markdown('''
### Andmeallikad

#### Sündimus ja suremus

- **Human Fertility Database** — vanusepõhised sündimuskordajad (ASFR) ja summaarne sündimuskordaja (TFR): https://www.humanfertility.org/
- **Human Mortality Database** — naiste ja meeste elutabelid: https://www.mortality.org/

#### Rahvastik ja ränne (Statistikaamet)

- **RL21434** — rahvastik emakeele, soo ja vanuserühma järgi, 2021. aasta rahvaloendus
- **RVR03** — välistränne soo, vanuserühma ja näitaja järgi (koguränne)
- **RVR09** — sisserändajad sünniriigi, vanuserühma ja soo järgi
- **RVR10** — eestlaste välistränne vanuserühma ja soo järgi

Statistikaameti andmed: https://andmed.stat.ee/
''')
else:
    st.markdown('''
### Data sources

#### Fertility and mortality

- **Human Fertility Database** — age-specific fertility rates (ASFR) and total fertility rate (TFR): https://www.humanfertility.org/
- **Human Mortality Database** — female and male life tables: https://www.mortality.org/

#### Population and migration (Statistics Estonia)

- **RL21434** — population by mother tongue, sex and age group, 2021 census
- **RVR03** — international migration by sex, age group and indicator (total migration)
- **RVR09** — immigrants by country of birth, age group and sex
- **RVR10** — international migration of Estonians by age group and sex

Statistics Estonia data: https://andmed.stat.ee/
''')
