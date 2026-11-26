"""Spin Flip-Out dashboard, Team 4. Run from this folder with:  uv run streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ART = Path(__file__).resolve().parent / 'artifacts'

## Spin colors (purple #5600D9 from spinbyoxxo.com.mx). Purple is the brand, orange is the single accent, grays carry everything else.
PURPLE, PURPLE_DARK, LILAC, ORANGE = '#5600D9', '#3D0099', '#EDE8FC', '#F15A22'
INK, MUTED, GRID, GREY = '#15142A', '#6B6880', '#ECEAF3', '#B8B5C6'
## One fixed color per persona (checked for color blindness), so a persona looks the same on every page
PERSONA_ORDER = ['Everyday Power Users', 'Occasional Physical', 'Cross-Border Online', 'Bank-Funded Diversified',
                 'Virtual Card Shoppers', 'OXXO Loyalists']
PCOLOR = dict(zip(PERSONA_ORDER, ['#5600D9', '#F15A22', '#1BA39C', '#C2185B', '#E6A700', '#2F80ED']))

st.set_page_config(page_title='Spin Flip-Out Dashboard', page_icon='🟣', layout='wide')

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1280px;}}
.hero {{background: linear-gradient(120deg, {PURPLE_DARK} 0%, {PURPLE} 60%, #7A3BEB 100%); color: white;
        border-radius: 18px; padding: 26px 30px 22px; margin-bottom: 14px;}}
.hero h1 {{color: white; font-size: 2.0rem; margin: 0 0 6px 0; padding: 0;}}
.hero p {{color: #E9E3FF; font-size: 1.05rem; margin: 0;}}
.chip {{display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
        color: white; border-radius: 999px; padding: 3px 12px; font-size: .82rem; margin: 12px 6px 0 0;}}
.chip.hot {{background: {ORANGE}; border-color: {ORANGE};}}
.kpi {{background: white; border: 1px solid {GRID}; border-radius: 14px; padding: 12px 16px; min-height: 104px;}}
.kpi .lab {{color: {MUTED}; font-size: .85rem;}}
.kpi .val {{color: {INK}; font-size: 1.45rem; font-weight: 700; line-height: 1.3;}}
.note {{color: {MUTED}; font-size: .86rem; margin: 6px 0 12px;}}
.note b {{color: {INK};}}
.htab {{border: 1px solid {GRID}; border-radius: 10px; overflow: auto; margin-bottom: 6px;}}
.htab table {{width: 100%; border-collapse: collapse; font-size: .88rem; margin: 0;}}
.htab th {{color: {MUTED}; font-weight: 600; background: #FAF9FD; text-align: left; padding: 7px 10px; border: 0; border-bottom: 1px solid {GRID};}}
.htab td {{text-align: left; padding: 7px 10px; border: 0; border-bottom: 1px solid {GRID}; color: {INK};}}
.kpi .sub {{color: {MUTED}; font-size: .78rem;}}
.kpi.accent {{border-top: 4px solid {PURPLE};}}
.answer {{background: {LILAC}; border-left: 5px solid {PURPLE}; border-radius: 10px; padding: 14px 18px; color: {INK};}}
.answer b {{color: {PURPLE_DARK};}}
.pcard {{background: white; border: 1px solid {GRID}; border-radius: 14px; padding: 14px 16px 12px; height: 100%;}}
.pcard h4 {{margin: 0 0 2px 0; font-size: 1.08rem; color: {INK};}}
.pcard .who {{color: {MUTED}; font-size: .86rem; margin-bottom: 8px; min-height: 2.4em;}}
.pcard .nums {{display: flex; gap: 14px; margin: 6px 0 8px;}}
.pcard .nums div {{font-size: .74rem; color: {MUTED};}}
.pcard .nums b {{display: block; font-size: 1.15rem; color: {INK};}}
.mix {{display: flex; height: 9px; border-radius: 5px; overflow: hidden; gap: 2px; margin: 4px 0 3px;}}
.mixlab {{font-size: .72rem; color: {MUTED}; margin-bottom: 8px;}}
.act {{background: #FFF3EC; border-radius: 8px; padding: 8px 10px; font-size: .86rem; color: {INK};}}
.act b {{color: {ORANGE};}}
.stTabs [role="tablist"] {{gap: 6px;}}
.stTabs [role="tab"] {{background: #F4F1FD; border-radius: 10px 10px 0 0; padding: 8px 20px;}}
.stTabs [role="tab"] p {{font-weight: 600; font-size: 1rem;}}
.stTabs [role="tab"][aria-selected="true"] {{background: {PURPLE};}}
.stTabs [role="tab"][aria-selected="true"] p {{color: white;}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    return dict(kpis=json.load(open(ART / 'kpis.json')),
                versions=pd.DataFrame(json.load(open(ART / 'versions.json'))),
                cards=pd.read_parquet(ART / 'persona_cards.parquet').set_index('persona').reindex(PERSONA_ORDER),
                z=pd.read_parquet(ART / 'persona_zmeans.parquet').reindex(PERSONA_ORDER),
                pmodel=pd.read_parquet(ART / 'persona_model.parquet').set_index('persona').reindex(PERSONA_ORDER),
                target=pd.read_parquet(ART / 'target_list.parquet'),
                gains=pd.read_parquet(ART / 'gains.parquet'),
                drivers=pd.read_parquet(ART / 'drivers.parquet'),
                gains_all=pd.read_parquet(ART / 'gains_all.parquet'),
                fkpis=json.load(open(ART / 'flipin_kpis.json')),
                fversions=pd.DataFrame(json.load(open(ART / 'flipin_versions.json'))),
                fgains=pd.read_parquet(ART / 'flipin_gains.parquet'),
                fdrivers=pd.read_parquet(ART / 'flipin_drivers.parquet'),
                fpersona=pd.read_parquet(ART / 'flipin_persona.parquet').set_index('persona').reindex(PERSONA_ORDER),
                flist=pd.read_parquet(ART / 'flipin_list.parquet'),
                payday=pd.read_parquet(ART / 'payday.parquet').set_index('persona'))


def style(fig, height=380, legend=False):
    ## Same quiet look on every chart: white surface, light grid, conclusion title on the left
    fig.update_layout(template='plotly_white', height=height, font=dict(family='sans-serif', color=INK, size=13),
                      title=dict(x=0, xanchor='left', font=dict(size=17, color=INK)), margin=dict(l=10, r=20, t=60, b=10),
                      showlegend=legend, legend=dict(orientation='h', y=-0.18, x=0), hoverlabel=dict(bgcolor='white'),
                      plot_bgcolor='white', paper_bgcolor='white')
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    t = fig.layout.title.text
    if t and not t.startswith('<b>'):
        fig.update_layout(title_text='<b>' + t.replace('<br>', '</b><br><b>') + '</b>')
    return fig


def html_table(head, rows):
    ## Plain HTML table, so cells can carry color (Streamlit's own tables cannot)
    th = ''.join(f'<th>{h}</th>' for h in head)
    tr = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f"<div class='htab'><table><tr>{th}</tr>{tr}</table></div>"


def note(text):
    st.markdown(f"<p class='note'>{text}</p>", unsafe_allow_html=True)


def pick_persona(pill_key, ms_key):
    ## Picking a persona focuses the list on it; picking it again (which clears the pill) brings back every persona
    v = st.session_state.get(pill_key)
    st.session_state[ms_key] = [v] if v else list(PERSONA_ORDER)


def sync_pill(pill_key, ms_key):
    sel_ = st.session_state.get(ms_key) or []
    st.session_state[pill_key] = sel_[0] if len(sel_) == 1 else None


def hist3(groups, cols, cap, title):
    ## Three small histograms side by side. groups: list of (name, color, DataFrame); cols: list of (column, subplot title, axis title)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.06, subplot_titles=[f'<b>{c[1]}</b>' for c in cols])
    for name, color, df_ in groups:
        for j, (col_, _, _) in enumerate(cols):
            if col_ not in df_ or df_[col_].isna().all():
                continue
            x = df_[col_].clip(upper=cap) if col_ == 'spend' else df_[col_]
            fig.add_trace(go.Histogram(x=x, name=name, histnorm='percent', marker_color=color, opacity=.75, legendgroup=name,
                                       showlegend=j == 0, xbins=dict(start=0.5, size=1) if col_ in ('months', 'active') else None,
                                       hovertemplate='%{x}: %{y:.0f}% of users<extra>' + name + '</extra>'), row=1, col=j + 1)
    for j, (col_, _, ax) in enumerate(cols):
        fig.update_xaxes(title_text=ax, title_font_size=11, row=1, col=j + 1, range=[0, 100] if col_ == 'oxxo' else None)
    fig.update_yaxes(title_text='% of users', row=1, col=1)
    fig.update_annotations(font=dict(size=12, color=MUTED))
    fig.update_layout(barmode='overlay', title=title)
    style(fig, 380, legend=True)
    fig.update_layout(margin=dict(l=10, r=10, t=105, b=10), legend=dict(y=-0.3))
    return fig


def kpi(col, label, value, sub='', accent=False):
    col.markdown(f"<div class='kpi{' accent' if accent else ''}'><div class='lab'>{label}</div>"
                 f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


d = load()
k, cards, z, pm, target, gains, versions = d['kpis'], d['cards'], d['z'], d['pmodel'], d['target'], d['gains'], d['versions']
if 'week' not in versions:
    versions['week'] = 'Nov 9'
best = versions[versions['is_best']].iloc[0]
first = versions.iloc[0]
N_SCORED = len(target)
RATE = k['flip_out_rate_oot']
flips = N_SCORED * RATE  # expected flip-outs next month if the test rate holds

## Plain language for each persona: who they are and what to do with them
WHO = {
    'Everyday Power Users': 'Use Spin almost every day, mostly in supermarkets and stores outside OXXO.',
    'Occasional Physical': 'Use the card now and then in physical stores, with long gaps between purchases.',
    'Cross-Border Online': 'Buy online, often on foreign sites, and get more declined payments than anyone.',
    'Bank-Funded Diversified': 'Top up by bank transfer instead of cash and spread spending across many stores.',
    'Virtual Card Shoppers': 'Only buy online through the virtual card and almost never visit a store.',
    'OXXO Loyalists': 'Half of their purchases happen at OXXO. Spin is still mostly an OXXO wallet for them.'}
ACTION = {
    'Everyday Power Users': 'Protect them. Rewards on groceries and transport so they keep Spin as their main card.',
    'Occasional Physical': 'Wake them up. A reminder plus cashback in nearby stores to turn gaps into a habit.',
    'Cross-Border Online': 'Fix the friction. Fewer false declines on foreign sites and a clear reason when one fails.',
    'Bank-Funded Diversified': 'Grow the wallet. Show the card for bills and subscriptions they already pay by transfer.',
    'Virtual Card Shoppers': 'Bring them offline. Tap to pay with Apple or Google Wallet for their first store purchase.',
    'OXXO Loyalists': 'First step out. A first non-OXXO purchase bonus at stores close to their usual OXXO.'}
TRAIT = {'oxxo_tx_share': 'OXXO share', 'merchant_entropy': 'Store variety', 'ecommerce_share': 'Online share',
         'virtual_card_share': 'Virtual card use', 'intl_share': 'Foreign purchases', 'payday_excess': 'Payday spike',
         'weekend_share': 'Weekend buying', 'night_share': 'Night buying', 'recency_mean': 'Days since last purchase',
         'activity_cv': 'Irregular activity', 'decline_rate': 'Declined payments', 'avg_ticket': 'Ticket size',
         'cash_funding_ratio': 'Cash top-ups', 'funded_month_share': 'Months with a top-up'}
DRIVER = {'oxxo_share_last1': 'OXXO share last month', 'oxxo_delta_3m': 'Change in OXXO share, last 3 months',
          'oxxo_months': 'Months with an OXXO purchase', 'physical_non_oxxo_share': 'Share in other physical stores',
          'ecommerce_share': 'Online share', 'oxxo_tx_share': 'OXXO share, last 6 months', 'total_tx': 'Purchases, last 6 months',
          'tx_last1': 'Purchases last month', 'merchant_entropy': 'Store variety', 'non_oxxo_tx_last3': 'Non-OXXO purchases, last 3 months',
          'tenure_months': 'Months with Spin', 'recency_mean': 'Days since last purchase', 'spend_trend': 'Spend trend',
          'active_last3': 'Active months, last 3', 'oxxo_share_last3': 'OXXO share, last 3 months', 'non_oxxo_share_last3': 'Non-OXXO share, last 3 months'}
FI_ACTION = {
    'Everyday Power Users': 'Welcome back. A bonus on the next cash top-up at OXXO, since they spend a lot and need liquidity.',
    'Occasional Physical': 'A reminder with the nearest OXXO and a small bonus on a cash top-up.',
    'Cross-Border Online': 'Show OXXO as the easy way to top up in cash for online purchases abroad.',
    'Bank-Funded Diversified': 'Offer OXXO as a backup way to top up when a bank transfer is slow.',
    'Virtual Card Shoppers': 'Show that the virtual card can be topped up with cash at any OXXO.',
    'OXXO Loyalists': 'Win them back first. They left their main store, so send a direct message with a top-up bonus.'}
DRIVER.update({'oxxo_old_share': 'OXXO share, older 3 months', 'oxxo_new_share': 'OXXO share, newest 3 months',
               'months_since_oxxo': 'Months since the last OXXO purchase',
               'payday_excess_6m': 'Payday spike, last 6 months', 'payday_excess_last3': 'Payday spike, last 3 months',
               'payday_trend': 'Change in payday spike', 'oxxo_share_std_12m': 'OXXO share swings, 12 months',
               'oxxo_share_max_12m': 'Highest OXXO share, 12 months', 'oxxo_share_min_12m': 'Lowest OXXO share, 12 months',
               'oxxo_share_mean_12m': 'OXXO share, 12-month average', 'oxxo_gap_last_12m': 'Last month vs 12-month OXXO average'})
z = z.rename(columns=TRAIT)

## Header
st.markdown(f"""<div class='hero'>
<h1>Who should Spin nudge outside OXXO next month?</h1>
<p>This dashboard helps the <b>CRM lead at Spin</b> decide <b>which users to target next month with a non-OXXO purchase
incentive, which message each persona gets, and how many of them to contact</b>.</p>
<span class='chip hot'>Model in use: {best['version']}</span><span class='chip'>{best['label']}</span>
<span class='chip'>Data up to {pd.Timestamp(k['snapshot']).strftime('%B %Y')}</span><span class='chip'>Synthetic data, anonymous IDs</span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['① Overview', '② Personas', '③ Target list', '④ Model', '⑤ Migration list'])

## 1. Overview
with tab1:
    c = st.columns(5)
    kpi(c[0], 'Users in the sample', f"{k['users_sample']:,}", 'Spin users in our 12 months of data')
    kpi(c[1], 'Users we can score', f"{N_SCORED:,}", 'Active 3+ of the last 6 months, OXXO share of 10% or more')
    kpi(c[2], 'Purchases made at OXXO', f"{k['oxxo_share_of_purchases']:.0%}", 'Share of all card purchases')
    kpi(c[3], 'Flip-Out rate', f"{RATE:.0%}", 'Scorable users who moved their buying outside OXXO in 2 months', accent=True)
    kpi(c[4], 'Personas', f"{len(PERSONA_ORDER)}", f"{k['users_in_scope']:,} users grouped by how they buy")

    cap10 = gains.loc[gains['pct_contacted'] == 10].iloc[0]
    st.write('')
    st.markdown(f"<div class='answer'><b>The short answer.</b> If Spin contacts the top 10% of users by Flip-Out score "
                f"(about {round(N_SCORED * 0.10):,} people), it reaches about <b>{cap10['capture'] * flips:,.0f} users who are "
                f"already about to buy outside OXXO</b>, against {cap10['random'] * flips:,.0f} with a random list. That is "
                f"<b>{cap10['capture'] / cap10['random']:.1f} times</b> more of the right people for the same budget. "
                f"Use the Target list tab to change the size of the campaign.</div>", unsafe_allow_html=True)
    st.write('')

    a, b = st.columns(2)
    cnt = pd.Series(k['persona_counts']).reindex(PERSONA_ORDER)
    big, small = cnt.idxmax(), cnt.idxmin()
    fig = go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation='h', marker_color=[PCOLOR[p] for p in cnt.index],
                           marker_cornerradius=4, text=[f'{v:,}  ({v / cnt.sum():.0%})' for v in cnt.values],
                           textposition='outside', textfont=dict(color=INK), cliponaxis=False,
                           hovertemplate='%{y}: %{x:,} users<extra></extra>'))
    fig.update_layout(title=f'{big} are {cnt[big] / cnt[small]:.1f}x bigger than {small}', xaxis_title='Users',
                      yaxis=dict(autorange='reversed'), xaxis=dict(range=[0, cnt.max() * 1.3]))
    a.plotly_chart(style(fig), width='stretch')

    fr = pm['flip_out_rate']
    hi, lo = fr.idxmax(), fr.idxmin()
    fig = go.Figure(go.Bar(x=fr.index, y=fr.values * 100, marker_color=[PCOLOR[p] for p in fr.index], marker_cornerradius=4,
                           text=[f'{v:.0%}' for v in fr.values], textposition='outside', cliponaxis=False,
                           hovertemplate='%{x}: %{y:.0f}% flip out<extra></extra>'))
    fig.add_hline(y=RATE * 100, line_dash='dot', line_color=MUTED)
    fig.update_layout(title=f'{hi} flip out {fr[hi] / fr[lo]:.1f}x more often than {lo}', yaxis_title='% who flip out',
                      xaxis_tickangle=-20, yaxis=dict(range=[0, fr.max() * 125]))
    b.plotly_chart(style(fig), width='stretch')
    st.caption(f'The dotted line is the rate for all scorable users ({RATE:.0%}). A flip-out is a user whose share of purchases outside OXXO rises at least 10 points above their own '
               'usual level for two months in a row. Rates come from the latest two months we can check (June and July 2026).')

    ## Team question from Evidence 2: does payday change how Spin users spend?
    st.markdown('---')
    st.subheader('Does payday change how people spend?')
    pdy = d['payday'].reindex(PERSONA_ORDER + ['All users'])
    pdy['gap'] = (pdy['payday_share'] - pdy['calendar_share']) * 100
    allr = pdy.loc['All users']
    vv = versions.set_index('version')
    a, b = st.columns([3, 2])
    pp = pdy.loc[PERSONA_ORDER]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pp.index, y=pp['calendar_share'] * 100, name='Expected from the calendar', marker_color=GREY, marker_cornerradius=4,
                         hovertemplate='%{x}: %{y:.1f}% expected<extra></extra>'))
    fig.add_trace(go.Bar(x=pp.index, y=pp['payday_share'] * 100, name='Real share on payday', marker_color=[PCOLOR[p] for p in pp.index],
                         marker_cornerradius=4, text=[f'{g_:+.1f} pts' for g_ in pp['gap']], textposition='outside', cliponaxis=False,
                         hovertemplate='%{x}: %{y:.1f}% on payday<extra></extra>'))
    fig.update_layout(barmode='group', xaxis_tickangle=-20, yaxis_title='% of purchases on payday days',
                      title=f"Every persona buys on payday about as much as the calendar predicts (all users {allr['gap']:+.1f} pts)",
                      yaxis=dict(range=[0, pdy['payday_share'].max() * 135]))
    a.plotly_chart(style(fig, 400, legend=True), width='stretch')
    pay_line = ''
    if {'v5', 'v6'} <= set(vv.index):
        pay_line = (f" When we gave the model each user's payday spike (version v6), the precision score moved from "
                    f"{vv.loc['v5', 'pr_auc']:.3f} to {vv.loc['v6', 'pr_auc']:.3f}, which is practically nothing.")
    b.markdown(f"<div class='answer'><b>Not really.</b> Payday days hold about {allr['calendar_share']:.0%} of the calendar, and users "
               f"make {allr['payday_share']:.0%} of their purchases on them. The gap is small and looks alike in every persona.{pay_line}"
               f"<br><br><b>What this means for Spin.</b> Timing a campaign around payday will not change who leaves OXXO. "
               f"How each person buys (their persona and their OXXO habit) matters much more, so the list is built on that.</div>",
               unsafe_allow_html=True)
    st.caption('Payday days are the 1st to the 3rd and the 28th to the end of each month, the same definition our model uses. '
               'Shares come from the months used to build the personas (Aug 2025 to Jan 2026).')

## 2. Personas
with tab2:
    st.subheader('Six kinds of Spin users, and what to say to each one')
    cols = st.columns(3)
    for i, p in enumerate(PERSONA_ORDER):
        r, color = cards.loc[p], PCOLOR[p]
        top = z.loc[p].sort_values(ascending=False).index[:2]
        mix = [(r['oxxo'], '#2F2B45', 'OXXO'), (r['phys'], color, 'Other stores'), (r['ecom'], GREY, 'Online')]
        tot = sum(m[0] for m in mix) or 1
        bar = ''.join(f"<div style='width:{m[0] / tot * 100:.1f}%;background:{m[1]}'></div>" for m in mix if m[0] > 0)
        lab = ' · '.join(f"{m[2]} {m[0] / tot:.0%}" for m in mix)
        cols[i % 3].markdown(f"""<div class='pcard' style='border-top:5px solid {color}'>
<h4>{p}</h4><div class='who'>{WHO[p]}</div>
<div class='nums'><div><b>{int(r['users']):,}</b>users ({r['pct_users']:.0%})</div><div><b>{r['pct_spend']:.0%}</b>of all spend</div>
<div><b>{pm.loc[p, 'flip_out_rate']:.0%}</b>flip out</div><div><b>{r['ticket']:,.0f}</b>MXN ticket</div></div>
<div class='mix'>{bar}</div><div class='mixlab'>Where they buy: {lab}</div>
<div class='mixlab'>Stands out for: <b style='color:{INK}'>{top[0]}</b> and <b style='color:{INK}'>{top[1]}</b></div>
<div class='act'><b>Suggested action.</b> {ACTION[p]}</div></div>""", unsafe_allow_html=True)
        if i == 2:
            st.write('')
            cols = st.columns(3)
    st.write('')
    st.divider()
    st.subheader('Compare personas')
    view = st.radio('View', ['Heatmap', 'Radar', 'Side by side'], horizontal=True, label_visibility='collapsed')
    if view == 'Heatmap':
        zz = z.clip(-2.5, 2.5)
        fig = go.Figure(go.Heatmap(z=zz.values, x=zz.columns, y=zz.index, zmin=-2.5, zmax=2.5, xgap=2, ygap=2,
                                   colorscale=[[0, ORANGE], [0.5, '#F2F1F5'], [1, PURPLE]],
                                   text=[[f'{v:+.1f}' for v in row] for row in z.values], texttemplate='%{text}',
                                   hovertemplate='%{y}<br>%{x}: %{z:+.1f} vs average<extra></extra>',
                                   colorbar=dict(title='vs average', tickvals=[-2, 0, 2], ticktext=['Much lower', 'Average', 'Much higher'])))
        fig.update_layout(title='Each persona is defined by two or three habits that stand far from the average user',
                          xaxis_tickangle=-30, yaxis=dict(autorange='reversed'))
        st.plotly_chart(style(fig, 470), width='stretch')
        st.caption('Purple means the persona does this much more than the average user, orange much less. '
                   'Numbers are distance from the average user in standard units (0 is average).')
    elif view == 'Radar':
        pick = st.multiselect('Personas to compare', PERSONA_ORDER, default=['OXXO Loyalists', 'Everyday Power Users', 'Virtual Card Shoppers'])
        fig = go.Figure()
        for p in pick:
            vals = z.loc[p].clip(-2.5, 2.5)
            fig.add_trace(go.Scatterpolar(r=list(vals) + [vals.iloc[0]], theta=list(z.columns) + [z.columns[0]], name=p,
                                          fill='toself', opacity=.55, line=dict(color=PCOLOR[p], width=2)))
        fig.update_layout(title='OXXO Loyalists stretch along one habit, the other personas spread across several',
                          polar=dict(radialaxis=dict(range=[-2.5, 2.8], gridcolor=GRID, tickfont=dict(color=MUTED, size=10)),
                                     angularaxis=dict(gridcolor=GRID)))
        st.plotly_chart(style(fig, 560, legend=True), width='stretch')
    else:
        ## Several personas side by side, then each one against a base persona
        pick = st.pills('Personas to compare', PERSONA_ORDER, selection_mode='multi',
                        default=['Everyday Power Users', 'Virtual Card Shoppers', 'OXXO Loyalists'], key='side_pick')
        pick = [p for p in PERSONA_ORDER if p in (pick or [])] or ['OXXO Loyalists']
        M = [('Users', lambda p: cards.loc[p, 'users'], 'n'), ('Share of all spend', lambda p: cards.loc[p, 'pct_spend'], '%'),
             ('OXXO share of purchases', lambda p: cards.loc[p, 'oxxo'], '%'), ('Online share', lambda p: cards.loc[p, 'ecom'], '%'),
             ('Other stores share', lambda p: cards.loc[p, 'phys'], '%'), ('Typical ticket (MXN)', lambda p: cards.loc[p, 'ticket'], 'n'),
             ('Purchases in 6 months', lambda p: cards.loc[p, 'tx'], 'n'), ('Declined payments', lambda p: cards.loc[p, 'decline'], '%'),
             ('Cash top-ups', lambda p: cards.loc[p, 'cash'], '%'), ('Flip-Out rate', lambda p: pm.loc[p, 'flip_out_rate'], '%'),
             ('Share of the top 10% list', lambda p: pm.loc[p, 'share_of_list'], '%')]
        val = lambda v, k: f'{v:.0%}' if k == '%' else f'{v:,.0f}'
        rows = [[n] + [val(f(p), k) for p in pick] for n, f, k in M] + [['Suggested action'] + [ACTION[p].split('.')[0] for p in pick]]
        st.markdown(html_table(['Measure'] + pick, rows), unsafe_allow_html=True)
        st.caption('Typical values (medians) for each persona. Shares of the list use the model in use and a 10% campaign.')
        base = st.selectbox('Each persona against', pick, index=len(pick) - 1)
        others = [p for p in pick if p != base]
        if others:
            def dlt(v, k):
                ## percentages differ in points; the rest in their own units
                x = round(v * 100) if k == '%' else round(v)
                u = ' pts' if k == '%' else ''
                return f"<span style='color:{MUTED}'>0{u}</span>" if x == 0 else \
                    f"<b style='color:{PURPLE if x > 0 else ORANGE}'>{'+' if x > 0 else '−'}{abs(x):,}{u}</b>"
            drows = [[n, val(f(base), k)] + [dlt(f(p) - f(base), k) for p in others] for n, f, k in M]
            st.markdown(html_table(['Measure', base] + [f"{p}<br><span style='font-weight:400'>minus {base}</span>" for p in others], drows),
                        unsafe_allow_html=True)
            st.caption(f'Purple means higher than {base}, orange means lower. Shares differ in percentage points (pts).')
        else:
            st.caption(f'Pick at least one more persona to see the differences against {base}.')
    with st.expander('How were these groups made? (plain language)'):
        st.markdown(
            "We looked at 14 habits of every user over six months: where they buy (OXXO, other stores, online), how they top up "
            "(cash or bank), how often and how much they buy, how many payments get declined, and when they buy (night, weekend, "
            "payday). A grouping method then put together users whose habits look alike and kept apart users who look different. "
            "Each group got a name from the two or three habits that stand out the most. Age, gender and region were left out on "
            "purpose, so groups reflect behavior and not demographics. Groups were built on August 2025 to January 2026 and "
            "checked again on later months: two out of three users stay in the same group.")

## 3. Target list
with tab3:
    st.session_state.setdefault('tpers', list(PERSONA_ORDER))
    st.subheader('How big should the campaign be?')
    cap = st.slider('Share of scorable users the campaign can contact', 5, 30, 10, step=1, format='%d%%')
    row = gains.loc[gains['pct_contacted'] == cap].iloc[0]
    n_cap = round(N_SCORED * cap / 100)
    got, rnd = row['capture'] * flips, row['random'] * flips
    c = st.columns(4)
    kpi(c[0], 'Users to contact', f'{n_cap:,}', f'Top {cap}% of {N_SCORED:,} by Flip-Out score')
    kpi(c[1], 'Flip-outs reached with the model', f'{got:,.0f}', f"{row['capture']:.0%} of the {flips:,.0f} expected", accent=True)
    kpi(c[2], 'Flip-outs reached at random', f'{rnd:,.0f}', f"{row['random']:.0%} of them")
    kpi(c[3], 'Gain over random', f"{row['capture'] / row['random']:.1f}x", f'{got - rnd:+,.0f} extra right people')
    st.write('')

    a, b = st.columns([3, 2])
    fig = go.Figure()
    fig.add_vrect(x0=5, x1=30, fillcolor=LILAC, opacity=.5, line_width=0)
    fig.add_trace(go.Scatter(x=gains['pct_contacted'], y=gains['capture'] * 100, name='With the model',
                             line=dict(color=PURPLE, width=3), hovertemplate='Contact %{x}%, reach %{y:.0f}%<extra>Model</extra>'))
    fig.add_trace(go.Scatter(x=gains['pct_contacted'], y=gains['random'] * 100, name='Random list',
                             line=dict(color=GREY, width=2, dash='dash'), hovertemplate='Contact %{x}%, reach %{y:.0f}%<extra>Random</extra>'))
    fig.add_trace(go.Scatter(x=[cap, cap], y=[row['random'] * 100, row['capture'] * 100], mode='markers+lines',
                             line=dict(color=ORANGE, width=2), marker=dict(size=10, color=ORANGE), showlegend=False, hoverinfo='skip'))
    fig.add_annotation(x=cap, y=row['capture'] * 100, text=f"<b>{row['capture']:.0%}</b> vs {row['random']:.0%}",
                       showarrow=False, xanchor='left', xshift=10, yshift=8, font=dict(color=INK))
    fig.update_layout(title=f"Contacting {cap}% by score reaches {row['capture'] / row['random']:.1f}x more flip-outs than at random",
                      xaxis_title='% of users contacted (shaded: campaign sizes we can afford)', yaxis_title='% of all flip-outs reached')
    a.plotly_chart(style(fig, 400, legend=True), width='stretch')

    on_list = target[target['rank'] <= n_cap]['persona'].value_counts().reindex(PERSONA_ORDER).fillna(0)
    lead = on_list.idxmax()
    fig = go.Figure(go.Bar(x=on_list.values, y=on_list.index, orientation='h', marker_color=[PCOLOR[p] for p in on_list.index],
                           marker_cornerradius=4, text=[f'{v:,.0f}' for v in on_list.values], textposition='outside',
                           cliponaxis=False, hovertemplate='%{y}: %{x:,} users on the list<extra></extra>'))
    fig.update_layout(title=f'The {lead} persona fills {on_list[lead] / on_list.sum():.0%} of the list', xaxis_title='Users on the list',
                      yaxis=dict(autorange='reversed'), xaxis=dict(range=[0, on_list.max() * 1.25]))
    b.plotly_chart(style(fig, 400), width='stretch')

    st.divider()
    st.subheader('The list')
    f1, f2 = st.columns([3, 2])
    sel = f1.multiselect('Persona', PERSONA_ORDER, key='tpers', on_change=sync_pill, args=('tpill', 'tpers'))
    dec = f2.select_slider('Score decile (1 = most likely to flip out)', options=list(range(1, 11)), value=(1, 1))
    in_dec = target[target['decile'].between(*dec)]
    lst = in_dec[in_dec['persona'].isin(sel)].copy()
    lst['action'] = lst['persona'].map(lambda p: ACTION[p].split('.')[0])
    lst['in_campaign'] = lst['rank'] <= n_cap
    st.markdown(f"**{len(lst):,}** users match these filters, **{lst['in_campaign'].sum():,}** of them fall inside a {cap}% campaign.")

    one = sel[0] if len(sel) == 1 else None
    cnt_d = in_dec['persona'].value_counts().reindex(PERSONA_ORDER).fillna(0)
    st.pills('Pick a persona to focus the list (pick it again to see all)', PERSONA_ORDER, key='tpill',
             on_change=pick_persona, args=('tpill', 'tpers'))
    a, b = st.columns([2, 3])
    fig = go.Figure(go.Bar(x=cnt_d.values, y=cnt_d.index, orientation='h', marker_cornerradius=4,
                           marker=dict(color=[PCOLOR[p] for p in PERSONA_ORDER], opacity=[1 if not one or p == one else .25 for p in PERSONA_ORDER]),
                           text=[f'{v:,.0f}' if v else '' for v in cnt_d.values], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:,} users<extra></extra>'))
    fig.update_layout(title=f'Showing {one}' if one else f'{cnt_d.idxmax()} has the most users in this list',
                      xaxis_title='Users in the list', yaxis=dict(autorange='reversed'), xaxis=dict(range=[0, max(cnt_d.max(), 1) * 1.3]))
    a.plotly_chart(style(fig, 380), width='stretch')
    hd = pd.DataFrame({'oxxo': lst['oxxo_share'] * 100, 'active': lst['active_months'], 'spend': lst['monthly_spend']})
    who = one or 'All personas'
    if len(hd):
        ttl = (f"{who}: half of these users buy {hd['oxxo'].round().median():.0f}% or more at OXXO,<br>"
               f"and the typical one spends {hd['spend'].median():,.0f} MXN a month")
    else:
        ttl = f'{who}: no users match these filters'
    s_cap = hd['spend'].quantile(.95) if len(hd) else 1
    b.plotly_chart(hist3([('Users in the list', PCOLOR[one] if one else PURPLE, hd)],
                         [('oxxo', 'OXXO share', '% of purchases'), ('active', 'Active months', 'Months (of 6)'),
                          ('spend', 'Monthly spend', f'MXN (top 5% grouped at {s_cap:,.0f})')], s_cap, ttl), width='stretch')
    st.caption('The histograms describe the users in the list right now.')

    show = lst.rename(columns={'rank': 'Rank', 'user_ref': 'User', 'persona': 'Persona', 'score': 'Flip-Out score',
                               'decile': 'Decile', 'action': 'Message', 'in_campaign': f'In the {cap}% campaign',
                               'oxxo_share': 'OXXO share', 'active_months': 'Active months (of 6)',
                               'monthly_spend': 'Monthly spend (MXN)', 'tx_last3': 'Purchases, last 3 months'})
    show['OXXO share'] = show['OXXO share'] * 100
    order = ['Rank', 'User', 'Persona', 'Flip-Out score', 'Decile', f'In the {cap}% campaign', 'Message', 'OXXO share',
             'Active months (of 6)', 'Monthly spend (MXN)', 'Purchases, last 3 months']
    on_screen = [c_ for c_ in order if c_ not in ('OXXO share', 'Active months (of 6)', 'Monthly spend (MXN)')]
    st.dataframe(show[on_screen].head(1000), hide_index=True, width='stretch', height=420,
                 column_config={'Flip-Out score': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.2f')})
    st.download_button('Download this list for the CRM tool', show[order].to_csv(index=False).encode(),
                       'spin_flipout_target_list.csv', 'text/csv', type='primary')
    st.caption('The download also includes OXXO share, active months and monthly spend for each user. '
               'The score ranks users from most to least likely to buy outside OXXO in the next two months. '
               'It ranks, it does not promise: even in the top decile not every user will flip out. User IDs are anonymous codes.')

## 4. Model
with tab4:
    st.subheader(f"Model {best['version']}: {best['label']}")
    c = st.columns(4)
    kpi(c[0], 'Precision score (PR-AUC)', f"{best['pr_auc']:.3f}", f'A random list scores {RATE:.3f}', accent=True)
    kpi(c[1], 'Flip-outs reached at 10%', f"{best['capture_10']:.1%}", 'A random list reaches 10.0%')
    kpi(c[2], 'Gain at 10%', f"{best['lift_10']:.1f}x", 'Versus a random list')
    kpi(c[3], 'Ranking score (ROC-AUC)', f"{best['roc_auc']:.3f}", '0.5 is a coin flip, 1.0 is perfect')
    st.write('')

    a, b = st.columns(2)
    v = versions
    fig = go.Figure(go.Bar(x=v['version'], y=v['pr_auc'], marker_cornerradius=4,
                           marker_color=[PURPLE if x else '#CFC6F5' for x in v['is_best']],
                           text=[f'{x:.3f}' for x in v['pr_auc']], textposition='outside', cliponaxis=False,
                           customdata=v['label'], hovertemplate='%{x}: %{customdata}<br>PR-AUC %{y:.3f}<extra></extra>'))
    fig.add_hline(y=RATE, line_dash='dot', line_color=MUTED, annotation_text=f'Random list {RATE:.3f}',
                  annotation_position='bottom left', annotation_font_color=MUTED)
    gain = best['pr_auc'] / RATE
    fig.update_layout(title=f"{best['version']} finds flip-outs {gain:.1f}x better than chance; versions differ by {v['pr_auc'].max() - v['pr_auc'].min():.3f}",
                      yaxis=dict(range=[0, v['pr_auc'].max() * 1.18], title='Precision score (PR-AUC)'))
    a.plotly_chart(style(fig, 400), width='stretch')

    dr = d['drivers'].copy()
    dr['name'] = dr['feature'].map(lambda s: DRIVER.get(s, s.replace('_', ' ').capitalize()))
    dr = dr.sort_values('importance')
    oxxo_part = dr[dr['feature'].str.contains('oxxo')]['importance'].sum() / dr['importance'].sum()
    fig = go.Figure(go.Bar(x=dr['importance'], y=dr['name'], orientation='h', marker_cornerradius=4,
                           marker_color=[PURPLE if 'oxxo' in f else '#CFC6F5' for f in dr['feature']],
                           hovertemplate='%{y}<extra></extra>'))
    fig.update_layout(title=f'OXXO habits (purple) carry {oxxo_part:.0%} of what moves the score',
                      xaxis_title='How much it moves the score (average)')
    b.plotly_chart(style(fig, 400), width='stretch')
    note(f"<b>What changed since the last version.</b> This week we ran "
         f"{(versions['week'] == best['week']).sum()} new experiments and {best['version']} came out on top: "
         f"{best['change'][0].lower() + best['change'][1:]} {best['reason']} Most of the signal still comes from "
         f"each user's recent OXXO habits.")

    st.markdown('---')
    st.subheader('Compare two versions')
    ca, cb = st.columns(2)
    vA = ca.selectbox('Version A', list(versions['version']), index=0)
    vB = cb.selectbox('Version B', list(versions['version']), index=1)
    rA, rB = versions[versions['version'] == vA].iloc[0], versions[versions['version'] == vB].iloc[0]
    dpr, dcap = rB['pr_auc'] - rA['pr_auc'], rB['capture_10'] - rA['capture_10']
    pts = round(dcap * 100, 1) + 0.0
    verdict = f'In plain words, {vB} and {vA} are <b>practically tied</b>.' if abs(dpr) < 0.005 else \
        f"In plain words, {vB} is <b>{'better' if dpr > 0 else 'worse'}</b> than {vA}."
    ga = d['gains_all']
    gA, gB = ga[ga['version'] == vA].set_index('pct_contacted'), ga[ga['version'] == vB].set_index('pct_contacted')
    st.markdown(f"**{vB} against {vA}.** The Delta column is {vB} minus {vA}: purple means {vB} is higher, orange means lower.")
    cmp_a, cmp_b = st.columns([2, 3])

    def sgn(x, txt):
        t = txt(abs(x))
        if not any(ch in '123456789' for ch in t):
            return f"<span style='color:{MUTED}'>{t}</span>"
        return f"<b style='color:{PURPLE if x > 0 else ORANGE}'>{'+' if x > 0 else '−'}{t}</b>"
    spec = [('Precision score (PR-AUC)', rA['pr_auc'], rB['pr_auc'], '{:.3f}'.format, '{:.3f}'.format),
            ('Ranking score (ROC-AUC)', rA['roc_auc'], rB['roc_auc'], '{:.3f}'.format, '{:.3f}'.format),
            ('Flip-outs reached at 10%', rA['capture_10'], rB['capture_10'], '{:.1%}'.format, lambda v: f'{v * 100:.1f} pts'),
            ('Gain over random at 10%', rA['lift_10'], rB['lift_10'], '{:.2f}x'.format, '{:.2f}x'.format),
            ('Training rows', rA['train_rows'], rB['train_rows'], '{:,.0f}'.format, '{:,.0f}'.format)]
    for c_ in (5, 20, 30):
        spec.append((f'Flip-outs reached with a {c_}% campaign (people)', round(round(gA.loc[c_, 'capture'], 4) * flips),
                     round(round(gB.loc[c_, 'capture'], 4) * flips), '{:,.0f}'.format, '{:,.0f}'.format))
    cmp_a.markdown(html_table(['Measure', vA, vB, f'Delta ({vB} − {vA})'],
                              [[n, fv(a_), fv(b_), sgn(b_ - a_, fd)] for n, a_, b_, fv, fd in spec]), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gA.index, y=gA['capture'] * 100, name=vA, line=dict(color=GREY, width=3)))
    fig.add_trace(go.Scatter(x=gB.index, y=gB['capture'] * 100, name=vB, line=dict(color=PURPLE, width=3)))
    fig.add_trace(go.Scatter(x=gA.index, y=gA['random'] * 100, name='Random list', line=dict(color=GREY, width=1.5, dash='dash')))
    fig.update_layout(title=f"{vB} and {vA} reach almost the same share of flip-outs at every campaign size" if abs(dcap) < 0.01
                      else f"{vB} reaches {abs(dcap) * 100:.1f} points {'more' if dcap > 0 else 'fewer'} flip-outs than {vA} at 10%",
                      xaxis_title='% of users contacted', yaxis_title='% of all flip-outs reached')
    cmp_b.plotly_chart(style(fig, 360, legend=True), width='stretch')
    note(f"{vA}: {rA['change'][0].lower() + rA['change'][1:]} "
         f"{vB}: {rB['change'][0].lower() + rB['change'][1:]} The precision score moves from {rA['pr_auc']:.3f} to "
         f"{rB['pr_auc']:.3f} ({dpr:+.3f}) and the flip-outs reached at 10% from {rA['capture_10']:.1%} to "
         f"{rB['capture_10']:.1%} ({pts:+.1f} points). {verdict}")

    st.markdown('**Experiment log**')
    ## Static table so the long text wraps instead of being cut
    log = pd.DataFrame({'Version': versions['version'] + versions['is_best'].map({True: ' (in use)', False: ''}),
                        'Logged': versions['week'], 'What we changed': versions['change'], 'Why': versions['reason'],
                        'Result: PR-AUC': versions['pr_auc'].map('{:.3f}'.format),
                        'Result: reached at 10%': versions['capture_10'].map('{:.1%}'.format)}).set_index('Version')
    st.table(log)

    with st.expander('How to read these numbers'):
        st.markdown(
            f"- **Test.** Every version learns from months up to March 2026 at the latest and is graded on June and July 2026, "
            f"months it never saw. That is the same situation the CRM team faces: score today, see who flips next.\n"
            f"- **Precision score (PR-AUC).** How often the users at the top of the list are real flip-outs. A random list "
            f"scores the base rate, {RATE:.3f}. Higher is better.\n"
            f"- **Reached at 10%.** If we contact the top 10% of the list, the share of all flip-outs we reach. A random "
            f"10% reaches 10%.\n"
            f"- **Which version wins.** The highest precision score at three decimals. When two versions tie, we keep the "
            f"simpler one, because it is easier to explain and to keep running.\n"
            f"- **Drivers.** The bars show how much each habit pushes a user's score up or down on average "
            f"(SHAP values from the tree model).")

    st.markdown('**How the model does inside each persona**')
    pmt = pm.reset_index()[['persona', 'eligible_oot', 'flip_out_rate', 'share_of_list', 'own_flips_captured']].copy()
    for col_ in ['flip_out_rate', 'share_of_list', 'own_flips_captured']:
        pmt[col_] = pmt[col_] * 100
    st.dataframe(pmt.rename(columns={'persona': 'Persona', 'eligible_oot': 'Scorable users (test)',
                                     'flip_out_rate': 'Flip-Out rate', 'share_of_list': 'Share of the 10% list',
                                     'own_flips_captured': 'Own flip-outs reached at 10%'}),
                 hide_index=True, width='stretch',
                 column_config={c_: st.column_config.NumberColumn(format='%.0f%%') for c_ in
                                ['Flip-Out rate', 'Share of the 10% list', 'Own flip-outs reached at 10%']})

## 5. Migration list: Flip-Out and Flip-In together
with tab5:
    fk, fv, fg, fd, fpm, fl = d['fkpis'], d['fversions'], d['fgains'], d['fdrivers'], d['fpersona'], d['flist']
    fbest = fv[fv['is_best']].iloc[0]
    F_N, F_RATE = len(fl), fk['flip_in_rate_oot']
    f_flips = F_N * F_RATE
    ## A user who already drifted away cannot also be a Flip-Out candidate, so Flip-In wins when a user is in both lists
    out_df = target[~target['user_ref'].isin(fl['user_ref'])].copy().reset_index(drop=True)
    out_df['rank'] = out_df.index + 1
    out_df['decile'] = (out_df['rank'] - 1) * 10 // len(out_df) + 1
    st.session_state.setdefault('mpers', list(PERSONA_ORDER))
    st.subheader('Users move in two directions: out of OXXO and back in')
    c = st.columns(4)
    kpi(c[0], 'Users who may leave (Flip-Out)', f'{len(out_df):,}', 'Still buy at OXXO. Same model as the Target list')
    kpi(c[1], 'Users who may come back (Flip-In)', f'{F_N:,}', 'Had at least 10% of purchases at OXXO, now 3% or less', accent=True)
    kpi(c[2], 'Flip-In rate', f'{F_RATE:.0%}', 'Of these users, share that came back to OXXO in 2 months')
    kpi(c[3], 'Flip-Ins reached at 10%', f"{fbest['capture_10']:.0%}", 'A random list reaches 10%')
    st.write('')

    out_n = out_df['persona'].value_counts().reindex(PERSONA_ORDER).fillna(0)
    in_n = fl['persona'].value_counts().reindex(PERSONA_ORDER).fillna(0)
    top_out, top_in = out_n.idxmax(), in_n.idxmax()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='May leave OXXO (Flip-Out)', x=PERSONA_ORDER, y=out_n.values, marker_color=PURPLE, marker_cornerradius=4,
                         text=[f'{v:,.0f}' for v in out_n.values], textposition='outside', cliponaxis=False))
    fig.add_trace(go.Bar(name='May come back to OXXO (Flip-In)', x=PERSONA_ORDER, y=in_n.values, marker_color=ORANGE, marker_cornerradius=4,
                         text=[f'{v:,.0f}' for v in in_n.values], textposition='outside', cliponaxis=False))
    fig.update_layout(barmode='group', title=f'{top_out} have the most users who may leave, and {top_in} the most who may come back',
                      yaxis_title='Users', xaxis_tickangle=-15, yaxis=dict(range=[0, max(out_n.max(), in_n.max()) * 1.2]))
    st.plotly_chart(style(fig, 400, legend=True).update_layout(legend_y=-0.32), width='stretch')

    st.divider()
    st.subheader('How many users to win back?')
    fcap = st.slider('Share of Flip-In candidates the campaign can contact', 5, 30, 10, step=1, format='%d%%', key='fcap')
    frow = fg[fg['pct_contacted'] == fcap].iloc[0]
    fn_cap = round(F_N * fcap / 100)
    fgot, frnd = frow['capture'] * f_flips, frow['random'] * f_flips
    c = st.columns(4)
    kpi(c[0], 'Users to contact', f'{fn_cap:,}', f'Top {fcap}% of {F_N:,} by Flip-In score')
    kpi(c[1], 'Flip-Ins reached with the model', f'{fgot:,.0f}', f"{frow['capture']:.0%} of the {f_flips:,.0f} expected", accent=True)
    kpi(c[2], 'Flip-Ins reached at random', f'{frnd:,.0f}', f"{frow['random']:.0%} of them")
    kpi(c[3], 'Gain over random', f"{frow['capture'] / frow['random']:.1f}x", f'{fgot - frnd:+,.0f} extra right people')
    st.write('')
    a, b = st.columns([3, 2])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fg['pct_contacted'], y=fg['capture'] * 100, name='With the model', line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=fg['pct_contacted'], y=fg['random'] * 100, name='Random list', line=dict(color=GREY, width=2, dash='dash')))
    fig.add_vrect(x0=5, x1=30, fillcolor=LILAC, opacity=.5, line_width=0, layer='below')
    fig.add_trace(go.Scatter(x=[fcap, fcap], y=[frow['random'] * 100, frow['capture'] * 100], mode='markers+lines',
                             line=dict(color=PURPLE, width=2), marker=dict(size=10, color=PURPLE), showlegend=False, hoverinfo='skip'))
    fig.update_layout(title=f"Contacting {fcap}% by score reaches {frow['capture'] / frow['random']:.1f}x more Flip-Ins than at random",
                      xaxis_title='% of users contacted', yaxis_title='% of all Flip-Ins reached')
    a.plotly_chart(style(fig, 380, legend=True), width='stretch')
    f_mix = fl[fl['rank'] <= fn_cap]['persona'].value_counts().reindex(PERSONA_ORDER).fillna(0)
    f_lead = f_mix.idxmax()
    fig = go.Figure(go.Bar(x=f_mix.values, y=f_mix.index, orientation='h', marker_color=[PCOLOR[p] for p in f_mix.index],
                           marker_cornerradius=4, text=[f'{v:,.0f}' for v in f_mix.values], textposition='outside', cliponaxis=False))
    fig.update_layout(title=f'The {f_lead} persona fills {f_mix[f_lead] / f_mix.sum():.0%} of the win-back list',
                      xaxis_title='Users on the list', yaxis=dict(autorange='reversed'), xaxis=dict(range=[0, f_mix.max() * 1.25]))
    b.plotly_chart(style(fig, 380), width='stretch')

    st.divider()
    st.subheader('The migration list')
    direction = st.radio('Direction', ['Both directions', 'May leave OXXO (Flip-Out)', 'May come back to OXXO (Flip-In)'], horizontal=True)
    g1, g2 = st.columns([3, 2])
    msel = g1.multiselect('Persona', PERSONA_ORDER, key='mpers', on_change=sync_pill, args=('mpill', 'mpers'))
    mdec = g2.select_slider('Score decile inside each direction (1 = most likely to move)', options=list(range(1, 11)), value=(1, 1), key='mdec')

    def prep(df_, label, msg, col_old):
        ## every persona in the chosen deciles; the persona filter comes after, so the bar chart keeps every persona visible
        df_ = df_[df_['decile'].between(*mdec)]
        return pd.DataFrame({'Direction': label, 'Rank in direction': df_['rank'], 'User': df_['user_ref'], 'Persona': df_['persona'],
                             'Score': df_['score'], 'Decile': df_['decile'], 'Message': df_['persona'].map(msg),
                             'OXXO share': df_[col_old] * 100, 'Months since last OXXO purchase': df_['months_since_oxxo'].astype(int).astype(str) if 'months_since_oxxo' in df_ else '',
                             'Monthly spend (MXN)': df_['monthly_spend'], 'Purchases, last 3 months': df_['tx_last3']})
    all_parts = {}
    if direction != 'May come back to OXXO (Flip-In)':
        all_parts['May leave OXXO'] = prep(out_df, 'May leave OXXO', {p_: ACTION[p_].split('.')[0] for p_ in PERSONA_ORDER}, 'oxxo_share')
    if direction != 'May leave OXXO (Flip-Out)':
        all_parts['May come back'] = prep(fl, 'May come back', {p_: FI_ACTION[p_].split('.')[0] for p_ in PERSONA_ORDER}, 'oxxo_old_share')
    parts = {k_: v_[v_['Persona'].isin(msel)] for k_, v_ in all_parts.items()}
    n_all = sum(len(x_) for x_ in parts.values())
    mig = pd.concat([x_.head(500) if len(parts) > 1 else x_ for x_ in parts.values()], ignore_index=True) if parts else pd.DataFrame()
    mig_all = pd.concat(parts.values(), ignore_index=True) if parts else pd.DataFrame()
    st.markdown(f"**{n_all:,}** users match these filters. Scores come from two different models, so the list is sorted "
                f"inside each direction and not across them." + (' The table shows the first 500 of each direction.' if len(parts) > 1 else ''))

    one = msel[0] if len(msel) == 1 else None
    dcol = {'May leave OXXO': PURPLE, 'May come back': ORANGE}
    st.pills('Pick a persona to focus the list (pick it again to see all)', PERSONA_ORDER, key='mpill',
             on_change=pick_persona, args=('mpill', 'mpers'))
    a, b = st.columns([2, 3])
    fig = go.Figure()
    tot_ = pd.Series(0, index=PERSONA_ORDER)
    for name, df_ in all_parts.items():
        cnt_ = df_['Persona'].value_counts().reindex(PERSONA_ORDER).fillna(0)
        tot_ += cnt_
        fig.add_trace(go.Bar(name=name, x=cnt_.values, y=PERSONA_ORDER, orientation='h', marker_cornerradius=4,
                             marker=dict(color=dcol[name], opacity=[1 if not one or p == one else .25 for p in PERSONA_ORDER]),
                             text=[f'{v:,.0f}' if v else '' for v in cnt_.values], textposition='outside', cliponaxis=False,
                             hovertemplate='%{y}: %{x:,} users<extra>' + name + '</extra>'))
    fig.update_layout(barmode='group', title=f'Showing {one}' if one else f'{tot_.idxmax()} has the most users in this list',
                      xaxis_title='Users in the list', yaxis=dict(autorange='reversed'), xaxis=dict(range=[0, max(tot_.max(), 1) * 1.3]))
    a.plotly_chart(style(fig, 380, legend=True).update_layout(legend_y=-0.3), width='stretch')
    groups, med = [], {}
    for name, df_ in parts.items():
        hd = pd.DataFrame({'oxxo': df_['OXXO share'], 'spend': df_['Monthly spend (MXN)'],
                           'months': pd.to_numeric(df_['Months since last OXXO purchase'], errors='coerce')})
        groups.append((name, dcol[name], hd))
        med[name] = hd['oxxo'].round().median() if len(hd) else None
    who = one or 'All personas'
    mo, mi = med.get('May leave OXXO'), med.get('May come back')
    mo = None if mo is None or pd.isna(mo) else mo
    mi = None if mi is None or pd.isna(mi) else mi
    if mo is not None and mi is not None:
        ttl = f'{who}: Flip-Out users buy {mo:.0f}% at OXXO today,<br>Flip-In users bought {mi:.0f}% before leaving (medians)'
    elif mo is not None:
        ttl = f'{who}: Flip-Out users buy {mo:.0f}% of their purchases at OXXO (median)'
    elif mi is not None:
        ttl = f'{who}: Flip-In users bought {mi:.0f}% at OXXO before leaving (median)'
    else:
        ttl = f'{who}: no users match these filters'
    m_cap = mig_all['Monthly spend (MXN)'].quantile(.95) if len(mig_all) else 1
    b.plotly_chart(hist3(groups, [('oxxo', 'OXXO share', '% of purchases'), ('months', 'Months since last OXXO purchase', 'Months'),
                                  ('spend', 'Monthly spend', f'MXN (top 5% grouped at {m_cap:,.0f})')], m_cap, ttl), width='stretch')
    st.caption('The histograms describe the users '
               'in the list right now. For Flip-In users, OXXO share is the share in the older 3 months, before they drifted away. '
               'Flip-Out users still buy at OXXO, so the months since their last OXXO purchase only apply to Flip-In.')

    on_screen = [c_ for c_ in mig.columns if c_ not in ('OXXO share', 'Months since last OXXO purchase', 'Monthly spend (MXN)')]
    st.dataframe(mig[on_screen].head(1000) if len(mig) else mig, hide_index=True, width='stretch', height=420,
                 column_config={'Score': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.2f')})
    st.download_button('Download this list for the CRM tool', mig_all.to_csv(index=False).encode(), 'spin_migration_list.csv', 'text/csv', type='primary')
    st.caption('The download also includes OXXO share, months since the last OXXO purchase and monthly spend for each user. '
               'Flip-Out users are the ones in the Target list, minus the users who already drifted away from OXXO (those count as Flip-In). '
               'User IDs are anonymous codes.')

    st.divider()
    st.subheader('How the Flip-In model works')
    h1, h2 = st.columns(2)
    fdr = fd.copy()
    fdr['name'] = fdr['feature'].map(lambda s_: DRIVER.get(s_, s_.replace('_', ' ').capitalize()))
    fdr = fdr.sort_values('importance')
    fig = go.Figure(go.Bar(x=fdr['importance'], y=fdr['name'], orientation='h', marker_cornerradius=4, marker_color=ORANGE))
    fig.update_layout(title=f"{fdr.iloc[-1]['name']} is the first thing that tells us who comes back", xaxis_title='How much it moves the score (average)')
    h1.plotly_chart(style(fig, 400), width='stretch')
    fpr = fpm.reset_index()
    fig = go.Figure(go.Bar(x=fpr['persona'], y=fpr['flip_in_rate'] * 100, marker_color=[PCOLOR[p] for p in fpr['persona']], marker_cornerradius=4,
                           text=[f'{v:.0%}' for v in fpr['flip_in_rate']], textposition='outside', cliponaxis=False))
    hi_p, lo_p = fpr.loc[fpr['flip_in_rate'].idxmax()], fpr.loc[fpr['flip_in_rate'].idxmin()]
    fig.update_layout(title=f"{hi_p['persona']} come back most often ({hi_p['flip_in_rate']:.0%}),<br>{lo_p['persona']} least ({lo_p['flip_in_rate']:.0%})",
                      yaxis_title='% who come back', xaxis_tickangle=-20, yaxis=dict(range=[0, fpr['flip_in_rate'].max() * 125]))
    h2.plotly_chart(style(fig, 400), width='stretch')
    note(f"<b>Flip-In model {fbest['version']}.</b> {fbest['change']} {fbest['reason']} "
         f"It scores {F_N:,} users who drifted away. In the test months (June and July 2026), {F_RATE:.0%} of them came back. "
         f"Contacting the top 10% reaches {fbest['capture_10']:.0%} of the users who came back, "
         f"<b>{fbest['lift_10']:.1f} times</b> what a random list reaches. It is a weaker signal than Flip-Out, because only about "
         f"3,000 users a month fit the definition and their return is harder to predict.")
    st.markdown('**Flip-In experiment log**')
    flog = pd.DataFrame({'Version': fv['version'] + fv['is_best'].map({True: ' (in use)', False: ''}), 'Logged': fv['week'],
                         'What we changed': fv['change'], 'Why': fv['reason'], 'Result: PR-AUC': fv['pr_auc'].map('{:.3f}'.format),
                         'Result: reached at 10%': fv['capture_10'].map('{:.1%}'.format)}).set_index('Version')
    st.table(flog)
    note(f"A random list scores {F_RATE:.3f} in PR-AUC. Definition: a displaced user had at least 10% of purchases at OXXO in the older "
         f"3 months and 3% or less in the newest 3. A Flip-In is a displaced user who makes at least 2 OXXO purchases and has at least 10% "
         f"of purchases at OXXO in the next two months. We have not agreed this definition with Spin.")
    note('<b>Note.</b> A Flip-Out is a user who still buys at OXXO and starts buying somewhere else. A Flip-In is a user who had drifted '
         'away from OXXO and comes back, for example to top up with cash. This tab puts both in one place: who may leave, who may return, '
         'and one list to act on.')

st.caption('Team 4, Data Analytics and AI Tools (Gpo 602). Spin by OXXO challenge. Data is synthetic and shown only as '
           'aggregated tables and anonymous IDs.')
