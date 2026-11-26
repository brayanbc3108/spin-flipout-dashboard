"""Builds the tables the dashboard reads, starting from the Evidence 3 notebook.

The notebook is executed cell by cell, untouched, so personas, scaler, PCA, K-Means, Flip-Out label and the selected
LightGBM model are exactly the ones we delivered. After that we add the new experiments for Evidence 4 and write
small aggregated tables to app/artifacts/.

Run from the project root:  uv run python build_model.py (inside the course project, next to the Evidence 3 notebook)
It needs data/g4_dataSample_v2.csv. The dashboard itself never reads the raw file.
"""
import os
import sys
import json
import time
import hashlib
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOTEBOOK = ROOT / 'evidences' / 'evidence-3' / 'Team4_ML_PurchaserPersona.ipynb'
OUT = HERE / 'artifacts'
OUT.mkdir(exist_ok=True)
T_NOW = pd.Timestamp('2026-07-01')
T_STACK = [pd.Timestamp(d) for d in ('2025-10-01', '2025-11-01', '2025-12-01', '2026-01-01')]
## Week 2 (Nov 17): two more snapshots. Their labels end in May 2026, before the Jun and Jul test months.
T_STACK6 = T_STACK + [pd.Timestamp('2026-02-01'), pd.Timestamp('2026-03-01')]
NEW_FEATURES = ['spend_trend', 'active_last3', 'non_oxxo_share_last3']
## Week 3 (Nov 23): payday behavior and a 12-month view of the OXXO share (exp3 in notes/model-experiments-2026-11-23)
PAY = ['payday_excess_6m', 'payday_excess_last3', 'payday_trend']
L12 = ['oxxo_share_std_12m', 'oxxo_share_max_12m', 'oxxo_share_min_12m', 'oxxo_share_mean_12m', 'oxxo_gap_last_12m']


def payday_feats(df, T, users):
    ## Payday purchases above the calendar share: 6-month average, last 3 months, and the change between the two halves
    start = T - pd.DateOffset(months=5)
    p = df[(df.mes >= start) & (df.mes <= T) & df.userId.isin(users) & df.active].copy()
    p['k'] = (p.mes.dt.year - start.year) * 12 + p.mes.dt.month - start.month
    a = p.groupby('userId')['payday_excess_m'].mean()
    l3 = p[p.k >= 3].groupby('userId')['payday_excess_m'].mean()
    f3 = p[p.k < 3].groupby('userId')['payday_excess_m'].mean()
    return pd.DataFrame({'payday_excess_6m': a, 'payday_excess_last3': l3, 'payday_trend': l3 - f3}).reindex(users)


def year_feats(df, T, users):
    ## OXXO share over the last 12 months: average, spread, highest, lowest, and how far the last month sits from the average
    start = T - pd.DateOffset(months=11)
    p = df[(df.mes >= start) & (df.mes <= T) & df.userId.isin(users)].copy()
    p['ox'] = p['oxxo_tx_ratio_m'].where(p['tx_count_m'] > 0)
    gp = p.sort_values('mes').groupby('userId')['ox']
    out = pd.DataFrame({'oxxo_share_std_12m': gp.std(), 'oxxo_share_max_12m': gp.max(), 'oxxo_share_min_12m': gp.min(),
                        'oxxo_share_mean_12m': gp.mean()}).reindex(users)
    out['oxxo_gap_last_12m'] = out['oxxo_share_mean_12m'] - gp.last().reindex(users)
    return out


def add_week3(df, X, T, users):
    X[PAY] = payday_feats(df, T, users).values
    X[L12] = year_feats(df, T, users).values
    return X


def payday_summary(df, persona):
    ## Share of purchases made on payday days against the calendar share of those days, by persona (persona window)
    p = df[df.userId.isin(persona.index) & df.active & (df.mes >= '2025-08-01') & (df.mes <= '2026-01-01')].copy()
    p['persona'] = p.userId.map(persona)
    p['pay_tx'] = p['payday_tx_ratio_m'] * p['tx_count_m']
    p['exp_tx'] = p['expected_payday_share'] * p['tx_count_m']
    agg = lambda x: pd.Series({'payday_share': x.pay_tx.sum() / x.tx_count_m.sum(), 'calendar_share': x.exp_tx.sum() / x.tx_count_m.sum()})
    t = p.groupby('persona').apply(agg)
    t.loc['All users'] = agg(p)
    return t.rename_axis('persona').reset_index()


def run_notebook():
    ## Execute every code cell of the delivered notebook in one namespace. The notebook reads its CSV from its own folder.
    nb = json.load(open(NOTEBOOK))
    os.chdir(NOTEBOOK.parent)
    ns = {'__name__': '__main__', 'display': lambda *a, **k: None}
    code = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    t0 = time.time()
    for n, (i, src) in enumerate(code, 1):
        print(f'[{time.time() - t0:6.0f}s] notebook cell {i} ({n}/{len(code)})', flush=True)
        exec(compile(src, f'notebook_cell_{i}', 'exec'), ns)
    return ns


def extra_features(df, T, users):
    ## Three features that are not in the notebook: spend trend, active months in the last three, non-OXXO share of the last three
    start = T - pd.DateOffset(months=5)
    p = df[(df['mes'] >= start) & (df['mes'] <= T) & (df['userId'].isin(users))].copy()
    p['k'] = (p['mes'].dt.year - start.year) * 12 + p['mes'].dt.month - start.month
    p['oxxo_tx'] = p['oxxo_tx_ratio_m'] * p['tx_count_m']
    last3 = p[p['k'] >= 3].groupby('userId')
    prev3 = p[p['k'] < 3].groupby('userId')
    sp_l, sp_p = last3['spend_m'].sum() / 3, prev3['spend_m'].sum() / 3
    out = pd.DataFrame(index=users)
    out['spend_trend'] = ((sp_l - sp_p) / (sp_l + sp_p).replace(0, np.nan)).reindex(users)
    out['active_last3'] = last3['active'].sum().reindex(users).fillna(0)
    tx_l = last3['tx_count_m'].sum().reindex(users)
    out['non_oxxo_share_last3'] = 1 - (last3['oxxo_tx'].sum().reindex(users) / tx_l.replace(0, np.nan))
    return out


def gains_curve(y, p, seed):
    order = np.lexsort((np.random.default_rng(seed).random(len(p)), -p))
    cum = np.cumsum(y[order]) / y.sum()
    pct = np.arange(1, 101)
    idx = np.clip(np.round(pct / 100 * len(p)).astype(int) - 1, 0, len(p) - 1)
    return pd.DataFrame({'pct_contacted': pct, 'capture': cum[idx], 'random': pct / 100})


## ---- Flip-In: users who had drifted away from OXXO and come back --------------------------------------------------
## Displaced user: used OXXO for at least 10% of purchases in the older 3 months of the 6-month window and for 3% or less
## in the newest 3 months. Flip-In event: in the next two months the user makes at least 2 OXXO purchases and OXXO is at
## least 10% of their purchases. Same scope rules and same 6 snapshots as the Flip-Out model.
FI_NEW = ['oxxo_old_share', 'oxxo_new_share', 'months_since_oxxo']
FI_SMALL = dict(num_leaves=8, min_child_samples=60, n_estimators=200, learning_rate=0.03, colsample_bytree=0.7, subsample=0.8, reg_lambda=5.0)


def window_shares(df, T):
    start = T - pd.DateOffset(months=5)
    p = df[(df['mes'] >= start) & (df['mes'] <= T)].copy()
    p['k'] = (p['mes'].dt.year - start.year) * 12 + p['mes'].dt.month - start.month
    p['ox'] = p['oxxo_tx_ratio_m'] * p['tx_count_m']
    a = p[p.k < 3].groupby('userId')[['ox', 'tx_count_m']].sum()
    b = p[p.k >= 3].groupby('userId')[['ox', 'tx_count_m']].sum()
    return pd.DataFrame({'old': a['ox'] / a['tx_count_m'].replace(0, np.nan), 'new': b['ox'] / b['tx_count_m'].replace(0, np.nan)})


def flip_in_label(df, T, users):
    ox, tx = pd.Series(0.0, index=users), pd.Series(0.0, index=users)
    for h in (1, 2):
        m = df[df['mes'] == T + pd.DateOffset(months=h)].set_index('userId').reindex(users)
        ox += (m['oxxo_tx_ratio_m'] * m['tx_count_m']).fillna(0)
        tx += m['tx_count_m'].fillna(0)
    return ((ox.round() >= 2) & (ox / tx.replace(0, np.nan) >= 0.10)).astype(int)


def months_since_oxxo(df, T, users):
    start = T - pd.DateOffset(months=5)
    p = df[(df['mes'] >= start) & (df['mes'] <= T) & (df['userId'].isin(users)) & (df['oxxo_tx_ratio_m'] * df['tx_count_m'] > 0.5)]
    last = p.groupby('userId')['mes'].max()
    return ((T.year - last.dt.year) * 12 + T.month - last.dt.month).reindex(users).fillna(6)


def flip_in_frame(g, df, T, with_label=True):
    feat = g['build_features'](df, T)
    r3 = g['scope'](feat, T)[3][1]
    w = window_shares(df, T).reindex(feat.index)
    el = r3 & (w['old'] >= 0.10) & (w['new'].fillna(0) <= 0.03)
    users = el[el].index
    X, persona = g['stage2_frame'](feat, T, users)
    X['oxxo_old_share'] = w.loc[users, 'old'].values
    X['oxxo_new_share'] = w.loc[users, 'new'].fillna(0).values
    X['months_since_oxxo'] = months_since_oxxo(df, T, users).values
    y = flip_in_label(df, T, users).values if with_label else None
    return feat, users, X.reset_index(drop=True), persona.reset_index(drop=True), y


def flip_in(g, df, base_cols):
    SEED = g['SEED']
    snaps = {T: flip_in_frame(g, df, T) for T in T_STACK6 + [g['T_OOT'], g['T_TRAIN']]}
    for T in T_STACK6:
        print('flip-in snapshot', T.date(), len(snaps[T][4]), 'users, event rate', round(snaps[T][4].mean(), 4), flush=True)
    _, _, X_o, per_o, y_o = snaps[g['T_OOT']]
    ox_cols = [c for c in base_cols + FI_NEW if 'oxxo' in c or c in ('tenure_months', 'recency_mean', 'total_tx', 'tx_last1', 'ecommerce_share',
                                                                      'physical_non_oxxo_share', 'merchant_entropy')]
    lgb_big = dict(g['best_lgb'])
    runs = [('v1', 'Same model as the Flip-Out one, trained on one snapshot (Jan 2026)', base_cols, [g['T_TRAIN']], False, lgb_big,
             'Reuse the Flip-Out setup for Flip-In.', 'Starting point, so we can tell what each later change adds.'),
            ('v2', 'Six snapshots, recent months weighted more, plus three new features', base_cols + FI_NEW, T_STACK6, True, lgb_big,
             'Train on six snapshots (newer months weigh more) and add the OXXO share of the older and newer months and the months since the last OXXO purchase.',
             'Flip-In only has about 3,000 displaced users per month, so more months help. The features describe how far the user drifted and for how long.'),
            ('v3', 'v2 with a smaller model on OXXO and activity features only', ox_cols, T_STACK6, True, FI_SMALL,
             'Use a smaller tree model and only the 16 features about OXXO use and activity.',
             'With so few users per month, a large model can learn noise. A smaller one with fewer features should hold up better.')]
    versions, fitted = [], {}
    for vid, label, cols, months, rw, params, change, reason in runs:
        Xt = pd.concat([snaps[T][2] for T in months], ignore_index=True)
        yt = np.concatenate([snaps[T][4] for T in months])
        w = np.concatenate([np.full(len(snaps[T][4]), 1 + i / (len(months) - 1)) for i, T in enumerate(months)]) if rw else None
        model = g['make_lgb'](**params).fit(Xt[cols], yt, sample_weight=w)
        p = model.predict_proba(X_o[cols])[:, 1]
        e = g['evaluate'](y_o, p)
        fitted[vid] = (model, cols, p)
        versions.append({'version': vid, 'week': 'Nov 17', 'label': label, 'change': change, 'reason': reason, 'train_rows': int(len(yt)),
                         'train_event_rate': float(yt.mean()), 'pr_auc': float(e['PR-AUC']), 'roc_auc': float(e['ROC-AUC']),
                         'capture_10': float(e['Capture in top 10%']), 'lift_10': float(e['Lift in top 10%'])})
        print('flip-in', vid, round(e['PR-AUC'], 4), round(e['Capture in top 10%'], 4), flush=True)
    best = max(enumerate(versions), key=lambda t: (round(t[1]['pr_auc'], 3), -t[0]))[1]['version']
    for v in versions:
        v['is_best'] = v['version'] == best
    model, cols, p_best = fitted[best]
    print('flip-in best version:', best, flush=True)
    json.dump(versions, open(OUT / 'flipin_versions.json', 'w'), indent=2)
    gains_curve(y_o, p_best, SEED).to_parquet(OUT / 'flipin_gains.parquet')

    import shap
    s = np.random.default_rng(SEED).choice(len(X_o), min(2000, len(X_o)), replace=False)
    sv = shap.TreeExplainer(model).shap_values(X_o.iloc[s][cols])
    sv = sv[1] if isinstance(sv, list) else sv
    if getattr(sv, 'ndim', 2) == 3:
        sv = sv[:, :, 1]
    pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False).head(12).rename('importance').rename_axis('feature') \
        .reset_index().to_parquet(OUT / 'flipin_drivers.parquet')

    cut = np.quantile(p_best, 1 - g['CAPACITY'])
    rows = []
    for name in g['PERSONA_ORDER']:
        m = (per_o == name).values
        top = m & (p_best >= cut)
        rows.append({'persona': name, 'displaced_oot': int(m.sum()), 'flip_in_rate': float(y_o[m].mean()) if m.sum() else 0.0,
                     'share_of_list': float(top.sum() / (p_best >= cut).sum()),
                     'own_flips_captured': float(y_o[top].sum() / y_o[m].sum()) if y_o[m].sum() else 0.0})
    pd.DataFrame(rows).to_parquet(OUT / 'flipin_persona.parquet')

    ## Score the displaced users at the latest month. Same anonymous IDs as the Flip-Out list.
    feat_n, users_n, X_n, per_n, _ = flip_in_frame(g, df, T_NOW, with_label=False)
    score = model.predict_proba(X_n[cols])[:, 1]
    raw = feat_n.loc[users_n]
    lst = pd.DataFrame({'user_ref': ['U' + hashlib.sha1(('spin-g4-dashboard' + u).encode()).hexdigest()[:8].upper() for u in users_n],
                        'persona': per_n.values, 'score': score, 'oxxo_old_share': X_n['oxxo_old_share'].values,
                        'oxxo_new_share': X_n['oxxo_new_share'].values, 'months_since_oxxo': X_n['months_since_oxxo'].values,
                        'active_months': raw['active_months'].values, 'monthly_spend': raw['monthly_spend'].values,
                        'tx_last3': raw['tx_last3'].values})
    lst['rank'] = lst['score'].rank(method='first', ascending=False).astype(int)
    lst['decile'] = pd.qcut(lst['rank'], 10, labels=range(1, 11)).astype(int)
    lst.sort_values('rank').to_parquet(OUT / 'flipin_list.parquet')
    kp = {'users_displaced_now': int(len(lst)), 'flip_in_rate_oot': float(y_o.mean()), 'flip_ins_oot': int(y_o.sum()), 'users_oot': int(len(y_o)),
          'best_version': best, 'definition_users': 'at least 10% of purchases at OXXO in the older 3 months and 3% or less in the newest 3',
          'definition_event': 'at least 2 OXXO purchases and at least 10% of purchases at OXXO in the next two months'}
    json.dump(kp, open(OUT / 'flipin_kpis.json', 'w'), indent=2)
    return kp


def main():
    g = run_notebook()
    df, first_seen = g['df'], g['first_seen']
    df['payday_excess_m'] = df['payday_tx_ratio_m'] - df['expected_payday_share']
    SEED, CAP = g['SEED'], g['CAPACITY']
    base_cols = list(g['FEATURE_SETS'][g['LGB_FS']])
    print('\nNotebook finished. LightGBM feature set:', g['LGB_FS'], '| selected model:', g['SELECTED'], flush=True)

    y_oot = g['y_oot'].values
    X_oot = g['X_oot'].copy()
    X_oot[NEW_FEATURES] = extra_features(df, g['T_OOT'], g['elig_oot']).values
    X_oot = add_week3(df, X_oot, g['T_OOT'], g['elig_oot'])
    lgb_params = dict(g['best_lgb'])

    def snapshot(T, users=None):
        ## Same steps the notebook uses for its own snapshots
        feat = g['build_features'](df, T)
        if users is None:
            steps = g['scope'](feat, T)
            users = steps[4][1][steps[4][1]].index
        X, persona = g['stage2_frame'](feat, T, users)
        X = X.reset_index(drop=True)
        X[NEW_FEATURES] = extra_features(df, T, users).values
        X = add_week3(df, X, T, users)
        return feat, users, X, persona.reset_index(drop=True)

    ## Version 1 is the model we delivered in Evidence 3, as it came out of the notebook
    p1 = g['scores']['LightGBM']
    v1 = g['oot_results'].loc['LightGBM']
    versions = [{'version': 'v1', 'label': 'Evidence 3 model: LightGBM, one snapshot (Jan 2026)',
                 'change': 'Model as delivered in Evidence 3.',
                 'reason': 'Starting point, so we can tell what each later change adds.',
                 'train_rows': int(len(g['y_tr'])), 'train_event_rate': float(g['y_tr'].mean()),
                 'pr_auc': float(v1['PR-AUC']), 'roc_auc': float(v1['ROC-AUC']),
                 'capture_10': float(v1['Capture in top 10%']), 'lift_10': float(v1['Lift in top 10%'])}]
    fitted = {'v1': (g['final']['LightGBM'][0], list(g['final']['LightGBM'][1]), p1)}

    ## Training data: one table per monthly snapshot, each labeled with its own next two months
    parts = {}
    for T in T_STACK6:
        feat, users, X, _ = snapshot(T)
        y = g['flip_out_label'](feat, T, users).reset_index(drop=True).values
        parts[T] = (X, y)
        print('snapshot', T.date(), X.shape, 'event rate', round(y.mean(), 4), flush=True)

    def stack(months, recent_weight=False):
        X = pd.concat([parts[T][0] for T in months], ignore_index=True)
        y = np.concatenate([parts[T][1] for T in months])
        ## Weight 1 for the oldest month up to 2 for the newest, so recent behavior counts more
        w = np.concatenate([np.full(len(parts[T][1]), 1 + i / (len(months) - 1)) for i, T in enumerate(months)]) if recent_weight else None
        return X, y, w

    runs = [('v2', 'Nov 9', 'LightGBM, four snapshots (Oct 2025 to Jan 2026)', base_cols, T_STACK, False,
             'Train on four monthly snapshots instead of one.',
             'One snapshot depends on a single season. More rows and more months should make the ranking steadier.'),
            ('v3', 'Nov 9', 'LightGBM, four snapshots plus three new features', base_cols + NEW_FEATURES, T_STACK, False,
             'Add spend trend, active months in the last three and non-OXXO share of the last three months.',
             'The model saw purchase counts moving. Spend and recent non-OXXO behavior are closer to what a flip-out is.'),
            ('v4', 'Nov 17', 'LightGBM, six snapshots (Oct 2025 to Mar 2026)', base_cols, T_STACK6, False,
             'Add the February and March 2026 snapshots to the training data.',
             'v2 stopped in January, four months before the test. Months closer to the test should look more like it.'),
            ('v5', 'Nov 17', 'LightGBM, six snapshots, recent months weighted more', base_cols, T_STACK6, True,
             'Same data as v4, but each newer month counts more in training (weight 1 for October up to 2 for March).',
             'If users change habits over the year, the model should learn more from the latest months than from the oldest.'),
            ('v6', 'Nov 23', 'v5 plus payday behavior', base_cols + PAY, T_STACK6, True,
             'Add how much more each user buys on payday than the calendar predicts, over 6 months and the last 3, and its trend.',
             'Our team question was whether payday changes spending. If it does, it should help tell who flips out.'),
            ('v7', 'Nov 23', 'v6 plus a 12-month view of the OXXO share', base_cols + PAY + L12, T_STACK6, True,
             'Add the OXXO share over the last 12 months: average, spread, highest, lowest, and the gap between the last month and the average.',
             'Six months can hide a user who was far from OXXO earlier in the year. A full year shows how stable the habit really is.')]
    for v in versions:
        v['week'] = 'Nov 9'
    for vid, week, label, cols, months, rw, change, reason in runs:
        X_all, y_all, w = stack(months, rw)
        model = g['make_lgb'](**lgb_params).fit(X_all[cols], y_all, sample_weight=w)
        p = model.predict_proba(X_oot[cols])[:, 1]
        e = g['evaluate'](y_oot, p)
        fitted[vid] = (model, cols, p)
        versions.append({'version': vid, 'week': week, 'label': label, 'change': change, 'reason': reason,
                         'train_rows': int(len(y_all)), 'train_event_rate': float(y_all.mean()),
                         'pr_auc': float(e['PR-AUC']), 'roc_auc': float(e['ROC-AUC']),
                         'capture_10': float(e['Capture in top 10%']), 'lift_10': float(e['Lift in top 10%'])})
    for v in versions:
        print(v['version'], round(v['pr_auc'], 4), round(v['capture_10'], 4), flush=True)

    ## Best version by PR-AUC at three decimals. A tie goes to the simpler version.
    best = max(enumerate(versions), key=lambda t: (round(t[1]['pr_auc'], 3), -t[0]))[1]['version']
    for v in versions:
        v['is_best'] = v['version'] == best
    model, cols, p_best = fitted[best]
    print('best version:', best, flush=True)

    gains_curve(y_oot, p_best, SEED).to_parquet(OUT / 'gains.parquet')
    pd.concat([gains_curve(y_oot, fitted[v][2], SEED).assign(version=v) for v in fitted]).to_parquet(OUT / 'gains_all.parquet')

    ## Drivers from SHAP on the out-of-time users, like figure F10 in the notebook
    import shap
    s = np.random.default_rng(SEED).choice(len(X_oot), 5000, replace=False)
    sv = shap.TreeExplainer(model).shap_values(X_oot.iloc[s][cols])
    sv = sv[1] if isinstance(sv, list) else sv
    if getattr(sv, 'ndim', 2) == 3:
        sv = sv[:, :, 1]
    drivers = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False).head(12)
    drivers.rename('importance').rename_axis('feature').reset_index().to_parquet(OUT / 'drivers.parquet')

    ## Per persona results on the out-of-time test with the model in use
    per_o = g['persona_elig_oot']
    cut = np.quantile(p_best, 1 - CAP)
    rows = []
    for name in g['PERSONA_ORDER']:
        m = (per_o == name).values
        top = m & (p_best >= cut)
        rows.append({'persona': name, 'eligible_oot': int(m.sum()), 'flip_out_rate': float(y_oot[m].mean()),
                     'share_of_list': float(top.sum() / (p_best >= cut).sum()),
                     'own_flips_captured': float(y_oot[top].sum() / y_oot[m].sum())})
    pd.DataFrame(rows).to_parquet(OUT / 'persona_model.parquet')

    ## Score everyone who can be scored at the latest month. IDs are replaced by short hashes.
    feat_n, users_n, X_n, per_n = snapshot(T_NOW)
    score = model.predict_proba(X_n[cols])[:, 1]
    ref = ['U' + hashlib.sha1(('spin-g4-dashboard' + u).encode()).hexdigest()[:8].upper() for u in users_n]
    raw = feat_n.loc[users_n]
    target = pd.DataFrame({'user_ref': ref, 'persona': per_n.values, 'score': score, 'oxxo_share': raw['oxxo_tx_share'].values,
                           'active_months': raw['active_months'].values, 'monthly_spend': raw['monthly_spend'].values,
                           'tx_last3': raw['tx_last3'].values})
    target['rank'] = target['score'].rank(method='first', ascending=False).astype(int)
    target['decile'] = pd.qcut(target['rank'], 10, labels=range(1, 11)).astype(int)
    target.sort_values('rank').to_parquet(OUT / 'target_list.parquet')

    ## Persona cards and heatmap come straight from the notebook (training window, Aug 2025 to Jan 2026)
    card = g['card'].copy()
    card['pct_users'] = card['% users'] / 100
    card['pct_spend'] = card['% approved spend'] / 100
    card.rename_axis('persona').reset_index().drop(columns=['% users', '% approved spend']).to_parquet(OUT / 'persona_cards.parquet')
    g['z_means'].to_parquet(OUT / 'persona_zmeans.parquet')

    uni = feat_n.index
    in_scope = g['persona_tr']
    tx = g['feat_tr_all'].loc[g['universe_tr']]
    kpis = {'users_sample': int(df.userId.nunique()), 'users_in_scope': int(len(in_scope)),
            'users_eligible': int(len(users_n)),
            'oxxo_share_of_purchases': float((tx['oxxo_tx_share'] * tx['total_tx']).sum() / tx['total_tx'].sum()),
            'flip_out_rate_oot': float(y_oot.mean()), 'flip_outs_oot': int(y_oot.sum()), 'users_oot': int(len(y_oot)), 'snapshot': str(T_NOW.date()), 'window': 'Feb to Jul 2026',
            'persona_window': 'Aug 2025 to Jan 2026', 'best_version': best,
            'persona_counts': {k: int(v) for k, v in in_scope.value_counts().items()}}
    json.dump(kpis, open(OUT / 'kpis.json', 'w'), indent=2)
    json.dump(versions, open(OUT / 'versions.json', 'w'), indent=2)
    payday_summary(df, in_scope).to_parquet(OUT / 'payday.parquet')
    flip_in(g, df, base_cols)
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
