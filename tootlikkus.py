# © 2026 Martin Klesment. Licensed under CC BY-NC 4.0.
import streamlit as st

st.set_page_config(page_title="Tööjõu tootlikkus | Eesti rahvastikuprognoos")
import matplotlib.pyplot as plt

from projection import (
    BASE_YEAR, MAB_BASE, EMP_PRIME, EMP_OLDER,
    last_known_tfr, load_emp_rates, load_productivity_growth, load_eu_productivity, mean_growth,
    employment_path, required_productivity, catch_up_productivity,
)
from model import load_base, run_scenario

FIGURE_SIZE = (8, 4.5)
CHART_STYLE = {
    "axes.facecolor": "white", "axes.grid": True, "grid.color": "#cccccc",
    "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20,
}
BENCHMARK_START = (2000, 2010)   # benchmark periods: start year → last year in the data

TEXT = {
    'ET': dict(
        header='Tootlikkuse kasv, mis kompenseerib tööjõu vähenemist',
        intro=('Kui hõivatute arv kahaneb, peab iga töötaja tootma rohkem, et toodang (SKP) ei väheneks. '
               'See leht arvutab, kui kiiresti peaks tööjõu tootlikkus kasvama, et toodang kokku või '
               'toodang inimese kohta püsiks 2025. aasta tasemel või kasvaks valitud määraga, või et '
               'Eesti jõuaks tootlikkuselt järele ELi keskmisele.'),
        loading='Laadin andmeid...',
        load_error='Andmete laadimine ebaõnnestus. Palun kontrollige internetiühendust ja proovige lehte uuesti laadida.',
        sidebar='Vali eeldused',
        target_year='Sihtaasta',
        target='Eesmärk', opt_total='Toodang kokku', opt_pp='Toodang inimese kohta', opt_eu='ELi keskmine',
        catch_up_year='Järelejõudmise aasta',
        eu_growth='ELi tootlikkuse kasv (%/a)',
        eu_growth_help='Eurostat: ELi keskmine {a}–{last} oli {g:.2f}%/a.',
        m_output='Toodang kokku vs {year}', m_required_eu='Vajalik kasv kuni {cy}',
        eu_note=('Eesti tootlikkus hõivatu kohta oli {year}. aastal {lvl:.1f}% ELi keskmisest (ostujõu '
                 'standardis, Eurostat). Eesmärk: 100% aastaks {cy}; seejärel kasvab tootlikkus ELiga sama '
                 'kiiresti. Toodang kokku kujuneb tootlikkuse ja hõivatute arvu koosmõjul.'),
        cap_eu='Tootlikkus hõivatu kohta, % ELi keskmisest', lab_actual='Tegelik',
        growth='Toodangu kasv (%/a)',
        growth_help='0 = toodang (kokku või inimese kohta) püsib baasaasta tasemel.',
        prime=f'Hõivemäära muutus, vanus {EMP_PRIME[0]}–{EMP_PRIME[1] - 1} (pp)',
        older=f'Hõivemäära muutus, vanus {EMP_OLDER[0]}–{EMP_OLDER[1] - 1} (pp)',
        shift_help='Muutus sihtaastaks; rakendub lineaarselt alates baasaastast.',
        scen_from=('Demograafilised eeldused lehelt *Mudel*: TFR perioodi lõpus {tfr:.2f}, '
                   'suremuse langus {mort:.1f}%/a, baassisseränne {bi:.0f}%, baasväljaränne {em:.0f}%, '
                   'lisaränne {extra:,} inimest/a.'),
        scen_default=('Demograafilised eeldused: lehe *Mudel* vaikeväärtused (TFR {tfr:.2f}, rändeta). '
                      'Eelduste muutmiseks ava leht *Mudel*.'),
        m_employed='Hõivatute arv', m_ratio='Hõivatute osa rahvastikust',
        m_required='Vajalik tootlikkuse kasv', m_cumul='Tootlikkus vs {year}',
        per_year='%/a', thousand='tuh',
        benchmark=('Võrdluseks: tegelik tootlikkuse kasv hõivatu kohta oli {a0}–{last} keskmiselt {g0:.1f}%/a '
                   'ja {a1}–{last} {g1:.1f}%/a (Statistikaamet, RAL0013).'),
        cap_ratio='Hõivatute osa rahvastikust {y0}–{y1}',
        cap_employed='Hõivatute arv {y0}–{y1}',
        cap_req='Vajalik tootlikkuse kasv aastas {y0}–{y1}',
        ylab_ratio='%', ylab_employed='Tuhat inimest', ylab_req='%/a', xlab='Aasta',
        lab_req='Vajalik', lab_bench='Tegelik {a}–{last}',
        method_title='Kuidas arvutatakse',
        method=r'''
Toodang = tootlikkus × hõivatud. Kui toodang peab kasvama määraga $g$, on vajalik tootlikkus
aastal $t$ (baasaasta = 1):

- toodang kokku: $$A_t = (1+g)^t \cdot \frac{E_0}{E_t}$$
- toodang inimese kohta: $$A_t = (1+g)^t \cdot \frac{E_0 / P_0}{E_t / P_t}$$
- ELi keskmine: $$A_t = (1+g_{EL})^t \cdot (1/r_0)^{\min(t,T)/T}$$, kus $r_0$ on Eesti tase ELi
  suhtes baasaastal ja $T$ aastate arv järelejõudmiseni

kus $E_t$ on hõivatute arv ja $P_t$ rahvaarv. Hõivatute arv = prognoositud rahvastik soo ja
vanuse järgi × hõivemäär. Hõivemäärad on 2025. aasta tööjõu-uuringu tulemused soo ja
viieaastaste vanuserühmade kaupa (Statistikaamet, TT330); kasutaja valitud muutus rakendub
lineaarselt ja hõivemäär ei ületa 95%. Tootlikkus on mõõdetud hõivatu, mitte töötunni kohta.
''',
    ),
    'EN': dict(
        header='Productivity growth needed to offset a shrinking workforce',
        intro=('When the number of employed people shrinks, each worker has to produce more for output (GDP) '
               'not to fall. This page calculates how fast labour productivity would have to grow to keep total '
               'output or output per person at the 2025 level, or growing at a chosen rate, or for Estonia to '
               'catch up with the EU average productivity level.'),
        loading='Loading data...',
        load_error='Failed to load data. Please check your internet connection and reload the page.',
        sidebar='Select assumptions',
        target_year='Target year',
        target='Target', opt_total='Total output', opt_pp='Output per person', opt_eu='EU average',
        catch_up_year='Catch-up year',
        eu_growth='EU productivity growth (%/yr)',
        eu_growth_help='Eurostat: the EU average in {a}–{last} was {g:.2f}%/yr.',
        m_output='Total output vs {year}', m_required_eu='Required growth until {cy}',
        eu_note=('In {year}, Estonia\'s productivity per employed person was {lvl:.1f}% of the EU average '
                 '(purchasing power standards, Eurostat). Target: 100% by {cy}; after that productivity grows '
                 'at the EU rate. Total output results from productivity and the number of employed.'),
        cap_eu='Productivity per employed person, % of EU average', lab_actual='Actual',
        growth='Output growth (%/yr)',
        growth_help='0 = output (total or per person) stays at the base-year level.',
        prime=f'Employment rate change, ages {EMP_PRIME[0]}–{EMP_PRIME[1] - 1} (pp)',
        older=f'Employment rate change, ages {EMP_OLDER[0]}–{EMP_OLDER[1] - 1} (pp)',
        shift_help='Change by the target year, phased in linearly from the base year.',
        scen_from=('Demographic assumptions from the *Model* page: TFR at end of period {tfr:.2f}, '
                   'mortality decline {mort:.1f}%/yr, baseline immigration {bi:.0f}%, baseline emigration {em:.0f}%, '
                   'additional immigration {extra:,} persons/yr.'),
        scen_default=('Demographic assumptions: *Model* page defaults (TFR {tfr:.2f}, no migration). '
                      'Open the *Model* page to change them.'),
        m_employed='Employed persons', m_ratio='Employed share of population',
        m_required='Required productivity growth', m_cumul='Productivity vs {year}',
        per_year='%/yr', thousand='k',
        benchmark=('For comparison: actual productivity growth per employed person averaged {g0:.1f}%/yr in '
                   '{a0}–{last} and {g1:.1f}%/yr in {a1}–{last} (Statistics Estonia, RAL0013).'),
        cap_ratio='Employed share of population {y0}–{y1}',
        cap_employed='Employed persons {y0}–{y1}',
        cap_req='Required annual productivity growth {y0}–{y1}',
        ylab_ratio='%', ylab_employed='Thousand people', ylab_req='%/yr', xlab='Year',
        lab_req='Required', lab_bench='Actual {a}–{last}',
        method_title='How it is calculated',
        method=r'''
Output = productivity × employed. For output to grow at rate $g$, the required productivity
in year $t$ (base year = 1) is

- total output: $$A_t = (1+g)^t \cdot \frac{E_0}{E_t}$$
- output per person: $$A_t = (1+g)^t \cdot \frac{E_0 / P_0}{E_t / P_t}$$
- EU average: $$A_t = (1+g_{EU})^t \cdot (1/r_0)^{\min(t,T)/T}$$, where $r_0$ is Estonia's level
  relative to the EU in the base year and $T$ the number of years to catch up

where $E_t$ is the number of employed persons and $P_t$ the population. Employed persons =
projected population by sex and age × employment rate. Employment rates are the 2025 labour
force survey results by sex and five-year age group (Statistics Estonia, TT330); the
user-selected change is phased in linearly and rates are capped at 95%. Productivity is measured
per employed person, not per hour worked.
''',
    ),
}


@st.cache_data
def load_labour_data():
    rate_f, rate_m = load_emp_rates()
    return rate_f, rate_m, load_productivity_growth(), load_eu_productivity()


lang = st.radio("Language", ["ET", "EN"], horizontal=True, label_visibility="hidden")
T = TEXT[lang]

st.header(T['header'])
st.markdown(T['intro'])

# --- Data ---
try:
    with st.spinner(T['loading']):
        b = load_base()
        rate_f, rate_m, prod_growth, eu_prod = load_labour_data()
except Exception as e:
    st.error(T['load_error'])
    st.exception(e)
    st.stop()

# --- Sidebar ---
st.sidebar.markdown(T['sidebar'])
TARGETS = {'total': 'opt_total', 'per_person': 'opt_pp', 'eu': 'opt_eu'}
_target = st.sidebar.segmented_control(T['target'], options=list(TARGETS),
                                       format_func=lambda o: T[TARGETS[o]],
                                       selection_mode='single', default='total', key='prod_target')
target     = _target or 'total'   # None (deselected) falls back to total output
per_person = target == 'per_person'
target_year = st.sidebar.slider(T['target_year'], min_value=BASE_YEAR + 5, max_value=BASE_YEAR + 75,
                                step=1, value=2050, key='prod_target_year')
eu_growth_hist = eu_prod['eu_growth_pct'].dropna()
eu_last        = int(eu_growth_hist.index.max())
eu_mean        = mean_growth(eu_growth_hist, BENCHMARK_START[1], eu_last)
if target == 'eu':
    catch_up_year = st.sidebar.slider(T['catch_up_year'], min_value=BASE_YEAR + 1, max_value=BASE_YEAR + 75,
                                      step=1, value=2050, key='prod_catch_up_year')
    eu_growth     = st.sidebar.slider(T['eu_growth'], min_value=0.0, max_value=2.0, step=0.1,
                                      value=float(round(eu_mean, 1)), key='prod_eu_growth',
                                      help=T['eu_growth_help'].format(a=BENCHMARK_START[1], last=eu_last,
                                                                      g=eu_mean)) / 100
else:
    growth = st.sidebar.slider(T['growth'], min_value=0.0, max_value=4.0, step=0.25, value=0.0,
                               help=T['growth_help'], key='prod_growth') / 100
shift_prime = st.sidebar.slider(T['prime'], min_value=-10, max_value=10, step=1, value=0,
                                help=T['shift_help'], key='prod_shift_prime') / 100
shift_older = st.sidebar.slider(T['older'], min_value=-10, max_value=25, step=1, value=0,
                                help=T['shift_help'], key='prod_shift_older') / 100
period = target_year - BASE_YEAR

# --- Demographic scenario: from the Model page if visited, else its defaults ---
scenario = st.session_state.get('scenario')
if scenario is not None:
    st.caption(T['scen_from'].format(
        tfr=b.tfr_start * (1 + scenario['tfr_change']), mort=scenario['mort_improvement'] * 100,
        bi=scenario['baseline_inflow'] * 100, em=scenario['emigration_scale'] * 100,
        extra=scenario['extra_immig']))
else:
    _, tfr_last = last_known_tfr(b.tfr_start)
    tfr_default = round(tfr_last * 20) / 20
    scenario = dict(tfr_change=tfr_default / b.tfr_start - 1, ramp=6, mab_stop=MAB_BASE,
                    extra_immig=0, mort_improvement=0.0, baseline_inflow=0.0,
                    emigration_scale=0.0, osc_amp=0.0, osc_cycle=10)
    st.caption(T['scen_default'].format(tfr=tfr_default))

*_, yearly, by_age = run_scenario(b, {**scenario, 'period': period})

# --- Calculation ---
years      = yearly.index.values
population = yearly[['estonian', 'other']].sum(axis=1).values
employed   = employment_path(by_age, rate_f, rate_m, shift_prime, shift_older)
ratio      = employed / population
if target == 'eu':
    level_hist = eu_prod['ee_level_pct_eu'].dropna()
    level_year = BASE_YEAR if BASE_YEAR in level_hist.index else int(level_hist.index.max())
    index, annual, level = catch_up_productivity(period, catch_up_year - BASE_YEAR,
                                                 level_hist[level_year] / 100, eu_growth)
else:
    index, annual = required_productivity(employed, population, growth, per_person)
avg_required  = index[-1] ** (1 / period) - 1

last_data = int(prod_growth.index.max())
bench     = {a: mean_growth(prod_growth, a, last_data) for a in BENCHMARK_START}

# --- Metrics ---
st.divider()
st.caption(f"{target_year}")
c1, c2, c3, c4 = st.columns(4)
c1.metric(T['m_employed'], f"{employed[-1] / 1000:,.0f} {T['thousand']}",
          f"{(employed[-1] - employed[0]) / 1000:+,.0f} {T['thousand']}", border=False)
if target == 'eu':
    output = round((index[-1] * employed[-1] / employed[0] - 1) * 100) + 0
    c2.metric(T['m_output'].format(year=BASE_YEAR), f"{output:+d}%" if output else "0%", border=False)
else:
    c2.metric(T['m_ratio'], f"{ratio[-1] * 100:.1f}%", f"{(ratio[-1] - ratio[0]) * 100:+.1f}pp", border=False)
if target == 'eu':   # constant during catch-up, so the first year's rate is the rate until catch-up_year
    c3.metric(T['m_required_eu'].format(cy=catch_up_year), f"{annual[0] * 100:.2f}{T['per_year']}", border=False)
else:
    c3.metric(T['m_required'], f"{avg_required * 100:.2f}{T['per_year']}", border=False)
cumul = round((index[-1] - 1) * 100) + 0   # + 0 turns -0 into 0
c4.metric(T['m_cumul'].format(year=BASE_YEAR), f"{cumul:+d}%" if cumul else "0%", border=False)
a0, a1 = BENCHMARK_START
if target == 'eu':
    st.caption(T['eu_note'].format(year=level_year, lvl=level_hist[level_year], cy=catch_up_year))
st.caption(T['benchmark'].format(a0=a0, a1=a1, last=last_data, g0=bench[a0], g1=bench[a1]))

# --- Charts ---
st.divider()
col_l, col_r = st.columns(2)
with plt.rc_context(CHART_STYLE):
    with col_l:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        if target == 'eu':
            ax.plot(level_hist.index, level_hist.values, linewidth=2.5, color='#2196F3', label=T['lab_actual'])
            ax.plot(years, level * 100, linewidth=2.5, linestyle='--', color='#FF5722', label=T['lab_req'])
            ax.axhline(100, color='black', linewidth=0.8)
            ax.set_ylim(top=max(105, level.max() * 100 + 5))
            ax.set(ylabel='%', xlabel=T['xlab'])
            ax.legend(loc='lower right', fontsize=14)
            st.caption(T['cap_eu'])
        elif per_person:
            ax.plot(years, ratio * 100, linewidth=2.5, color='#2e86c1')
            ax.set(ylabel=T['ylab_ratio'], xlabel=T['xlab'])
            st.caption(T['cap_ratio'].format(y0=BASE_YEAR, y1=target_year))
        else:
            ax.plot(years, employed / 1000, linewidth=2.5, color='#2e86c1')
            ax.set(ylabel=T['ylab_employed'], xlabel=T['xlab'])
            ax.set_ylim(bottom=0)
            st.caption(T['cap_employed'].format(y0=BASE_YEAR, y1=target_year))
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.plot(years[1:], annual * 100, linewidth=2.5, color='#e05252', label=T['lab_req'])
        for a, style in zip(BENCHMARK_START, ('--', ':')):
            ax.axhline(bench[a], color='#555555', linewidth=1.5, linestyle=style,
                       label=T['lab_bench'].format(a=a, last=last_data))
        ax.axhline(0, color='black', linewidth=0.8)
        top = max(max(bench.values()), annual.max() * 100)
        ax.set_ylim(min(0, annual.min() * 100) - 0.3, top * 1.5 + 0.3)   # headroom for the legend
        ax.set(ylabel=T['ylab_req'], xlabel=T['xlab'])
        ax.legend(loc='upper right', fontsize=14)
        st.caption(T['cap_req'].format(y0=BASE_YEAR + 1, y1=target_year))
        st.pyplot(fig)
        plt.close(fig)

with st.expander(T['method_title']):
    st.markdown(T['method'])
