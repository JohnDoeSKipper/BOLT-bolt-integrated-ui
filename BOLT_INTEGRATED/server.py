"""
BOLT Integrated — Assembly-Line Pipeline Backend  (server.py)

Data flow (one-way chain):
  CSV upload
    -> Predictor (LightGBM, trains once on history)
       -> 48-step forecast (kW) every tick
          -> Manager (optimises forecasted load with battery)
             -> Calculator (TNB bill before/after)
             -> PowerRECO (solar + battery sizing + 25-yr ROI)

The Manager never reads raw CSV again after the Predictor is trained.
The background pipeline thread runs every N seconds, advancing the
simulation clock and producing a fresh optimised schedule each tick.

Run:  python server.py
Open: http://localhost:5000
"""
from __future__ import annotations
import io, json, math, threading, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider

# ── numpy → JSON ──────────────────────────────────────────────────────────────
class _NP(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_):    return bool(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

BASE = Path(__file__).parent
app  = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")
app.json_provider_class = _NP
app.json = _NP(app)
CORS(app)

# ── BOLT module imports ───────────────────────────────────────────────────────
from predictor.forecaster import DirectMultiStepForecaster
from predictor.solar_estimator import detect_has_solar, estimate_solar_capacity_kwp
from predictor.simulation import (
    initialize_simulation, advance_one_tick, compute_running_accuracy,
    split_historical_for_simulation,
)
from calculator.tnb_tariffs import (
    auto_detect_tariff, compute_monthly_stats, calculate_bill,
    compute_nem_credit, TARIFF_META,
)
from calculator.rates import SCHEDULES, DEFAULT_SCHEDULE_KEY
from powerreco.solar_sizing import calculate_solar_sizing
from powerreco.battery_sizing import calculate_battery_sizing
from powerreco.roi_engine import calculate_roi
from powerreco.optimizer import find_optimum_from_existing_run

from site_profiles import (
    PROFILES, list_profiles, get_profile,
    profile_loads_for_manager, profile_predictor_kwargs,
    DEFAULT_PROFILE_ID,
)

from persistence import load_overrides, save_overrides, OVERRIDES_PATH

# ── Shared application state ──────────────────────────────────────────────────
# Hydrate site overrides from disk on import so a server restart preserves
# whatever the user typed into Site Setup last time.
S: dict = {
    'site_overrides':  load_overrides(),     # {site_id: {field: value}}
    'use_real_weather': True,                # default-on; toggle via /api/site/configure
}
PL: dict = {          # pipeline control state
    'running':          False,
    'status':           'idle',     # idle | training | forecasting | optimizing | billing | powerreco | error
    'tick':             0,
    'interval_seconds': 30,
    'last_tick_ts':     None,
    'log':              [],         # list of {ts, msg, level}
    '_thread':          None,
    '_stop_evt':        None,
}
_pl_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# MANAGER v6 — exact replica from Manager/app.py  (kept verbatim)
# ══════════════════════════════════════════════════════════════════════════════
def _calc_kva(kw, kvar): return math.sqrt(kw*kw + kvar*kvar)


def _in_window(hour, window):
    """True if hour falls inside (start,end). Handles wrap (e.g. 18→8)."""
    if window is None: return False
    s, e = int(window[0]), int(window[1])
    if s == e: return False
    if s < e:  return s <= hour < e
    return hour >= s or hour < e


def _eff_max_cut(load, hour):
    """Window-aware effective max_cut_pct for a load at this hour."""
    base = float(load.get('max_cut_pct', 0.10))
    if _in_window(hour, load.get('protected_window')): return 0.0
    aw = load.get('allowed_window')
    if aw is not None and not _in_window(hour, aw): return 1.0
    return base

def _normalize_cols(df):
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace('﻿','',regex=False).str.replace(r'\s+',' ',regex=True))
    return df

def _is_valid_header(cols):
    s = ' '.join(cols)
    return (any(t in s for t in ('date','time','start','end','timestamp','datetime')) and
            any(t in s for t in ('kw','kvar','kwh','power','watt','energy','var')))

def _find_dt_col(df):
    cols = list(df.columns)
    for col in cols:
        if ('date' in col and 'time' in col) or col in ('datetime','timestamp'):
            df[col]=pd.to_datetime(df[col],errors='coerce'); return col,df
    for col in cols:
        if col in ('end_time','end time','start_time','start time'):
            df[col]=pd.to_datetime(df[col],errors='coerce'); return col,df
    date_c=next((c for c in cols if c=='date' or c.startswith('date')),None)
    time_c=next((c for c in cols if c in ('time','end time','end_time','start_time') and c!=date_c),None)
    if date_c and time_c:
        df['_dt']=pd.to_datetime(df[date_c].astype(str)+' '+df[time_c].astype(str),errors='coerce')
        return '_dt',df
    for col in cols[:10]:
        try:
            p=pd.to_datetime(df[col],errors='coerce')
            if p.notna().sum()/max(len(p),1)>=0.8: df[col]=p; return col,df
        except Exception: continue
    raise ValueError(f"No datetime column found. Columns: {cols}")

def _map_power(df):
    m={}
    for col in df.columns:
        kvar='kvar' in col or ('var' in col and 'kw' not in col)
        kw=('kw' in col or 'kwh' in col or 'watt' in col or 'active' in col
            or 'power' in col or 'energy' in col) and not kvar
        imp,exp='import' in col,'export' in col
        if kvar and imp and 'kvar_import' not in m: m['kvar_import']=col
        elif kvar and exp and 'kvar_export' not in m: m['kvar_export']=col
        elif kw and imp and 'kw_import' not in m: m['kw_import']=col
        elif kw and exp and 'kw_export' not in m: m['kw_export']=col
    for k in ('kw_import','kw_export','kvar_import','kvar_export'): m.setdefault(k,None)
    return m

def _parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    df=None
    if filename.lower().endswith('.csv'):
        for hrow in range(12):
            for enc in ('utf-8-sig','utf-8','latin-1','cp1252'):
                try:
                    tmp=pd.read_csv(io.StringIO(content.decode(enc)),header=hrow)
                    tmp=_normalize_cols(tmp)
                    if len(tmp.columns)>=4 and _is_valid_header(list(tmp.columns)):
                        df=tmp; break
                except Exception: continue
            if df is not None: break
        if df is None: raise ValueError("Could not locate valid header row in CSV.")
    elif filename.lower().endswith(('.xlsx','.xls')):
        engine='xlrd' if filename.lower().endswith('.xls') else 'openpyxl'
        fb=io.BytesIO(content)
        for hrow in range(12):
            try:
                fb.seek(0)
                tmp=pd.read_excel(fb,header=hrow,engine=engine)
                tmp=_normalize_cols(tmp)
                if len(tmp.columns)>=4 and _is_valid_header(list(tmp.columns)):
                    df=tmp; break
            except Exception: continue
        if df is None: raise ValueError("Could not locate valid header row in Excel.")
    else:
        raise ValueError(f"Unsupported format '{filename}'.")
    df=df.dropna(how='all').reset_index(drop=True)
    df=df.loc[:,df.columns.notna()]
    df=df.loc[:,df.columns.astype(str)!='']
    start_col,df=_find_dt_col(df)
    df=df.dropna(subset=[start_col]).sort_values(start_col).reset_index(drop=True)
    m=_map_power(df)
    def _s(key):
        col=m.get(key)
        if col and col in df.columns:
            return pd.to_numeric(df[col],errors='coerce').fillna(0)
        return pd.Series([0.0]*len(df),index=df.index)
    result=pd.DataFrame({
        'timestamp':  df[start_col].values,
        'kw_import':  _s('kw_import').values,
        'kw_export':  _s('kw_export').values,
        'kvar_import':_s('kvar_import').values,
        'kvar_export':_s('kvar_export').values,
    })
    result['kw_net']  =result['kw_import']-result['kw_export']
    result['kvar_net']=result['kvar_import']-result['kvar_export']
    result['kva']     =result.apply(lambda r:_calc_kva(r['kw_net'],r['kvar_net']),axis=1)
    result['timestamp']=pd.to_datetime(result['timestamp'])
    result=(result.set_index('timestamp').resample('30min').mean()
            .dropna(how='all').reset_index())
    result['kva']=result.apply(lambda r:_calc_kva(r['kw_net'],r['kvar_net']),axis=1)
    if result.empty: raise ValueError("Empty DataFrame after resampling.")
    return result


def _mgr_run(df, loads, battery_kwh, priority_order,
             peak_target_pct, bat_charge_upper_pct,
             c_rate=0.5, initial_soc_pct=0.50, bat_efficiency=0.95,
             peak_reference_kva=None, lookahead_intervals=16,
             md_start_hour=14, md_end_hour=22, pre_md_hours=2):
    """Manager v6 interval-by-interval optimiser (exact replica)."""
    INTERVAL_H=0.5; BAT_EMERG=0.15; BAT_FULL=0.90
    BAT_DIS_MIN=0.15; PROX_N=0.80; PROX_MD=0.70
    CHG_GUARD=0.92; EMERG_GUARD=1.00
    load_keys=list(loads.keys())
    tot_prop=sum(loads[k].get('proportion',0) for k in load_keys) or 1
    norm={k:loads[k].get('proportion',0)/tot_prop for k in load_keys}
    df=df.copy().sort_values('timestamp').reset_index(drop=True)
    df['date']=df['timestamp'].dt.date
    df['hour']=df['timestamp'].dt.hour
    df['in_md']=df['hour'].apply(lambda h: md_start_hour<=h<md_end_hour)
    pms=(md_start_hour-pre_md_hours)%24
    df['in_pre_md']=df['hour'].apply(
        lambda h:(pms<=h<md_start_hour) if pms<md_start_hour else (h>=pms or h<md_start_hour))
    if peak_reference_kva is not None:
        df['_ref']=float(peak_reference_kva)
    else:
        dm=df.groupby('date')['kva'].max().sort_index()
        rr=dm.shift(1).rolling(30,min_periods=1).max().fillna(dm.iloc[0]*1.10)
        df['_ref']=df['date'].map(rr.to_dict())
    day_pk=df.groupby('date')['kva'].max().to_dict()
    bmax=battery_kwh; bmin=battery_kwh*0.05; bemerg=battery_kwh*BAT_EMERG
    bfull=battery_kwh*BAT_FULL; bdmin=battery_kwh*BAT_DIS_MIN
    chg_rate=battery_kwh*c_rate; dis_rate=battery_kwh*c_rate
    soc=battery_kwh*initial_soc_pct
    results=[]; n=len(df)
    for idx in range(n):
        row=df.iloc[idx]; kva0=float(row['kva']); kw=float(row['kw_net'])
        kvar=float(row['kvar_net']); date=row['date']; ts=row['timestamp']
        ref=float(row['_ref']); in_md=bool(row['in_md']); in_pre=bool(row['in_pre_md'])
        trig=ref*peak_target_pct; prox=ref*(PROX_MD if in_md else PROX_N)
        cu=ref*bat_charge_upper_pct; cc=trig*CHG_GUARD; ec=trig*EMERG_GUARD
        lkva={k:kva0*norm[k] for k in load_keys}; lfac={k:1.0 for k in load_keys}
        bchg=bdis=0.0; acts=[]; mkw=kw; mkvar=kvar; mkva=_calc_kva(mkw,mkvar)
        uh=False
        for fi in range(idx+1,min(idx+lookahead_intervals+1,n)):
            if float(df.iloc[fi]['kva'])>=float(df.iloc[fi]['_ref'])*peak_target_pct:
                uh=True; break
        # Discharge
        if (kva0>=prox or uh) and mkva>=trig and soc>bdmin and mkw>0:
            kwt=math.sqrt(max(trig**2-mkvar**2,0.0))
            dkwn=max(mkw-kwt,0.0); dkwh=dkwn*INTERVAL_H/bat_efficiency
            dkwh=min(dkwh,dis_rate*INTERVAL_H,soc-bdmin)
            dkwl=dkwh*bat_efficiency/INTERVAL_H; sb=soc; soc-=dkwh
            mkw-=dkwl; mkw=max(mkw,0.0); mkva=_calc_kva(mkw,mkvar); bdis=dkwl
            la=bool(uh and kva0<prox)
            acts.append({'type':'battery_discharge','load':'Battery',
                'discharge_kw':round(dkwl,2),'soc_before_kwh':round(sb,1),
                'soc_after_kwh':round(soc,1),'kva_before':round(kva0,2),
                'kva_after':round(mkva,2),'lookahead_triggered':la,'md_hours':in_md,
                'note':f'Discharged {round(dkwl,1)}kW; SOC {round(sb,0):.0f}→{round(soc,0):.0f}kWh'+
                       (' [look-ahead]' if la else '')+(' [MD]' if in_md else '')})
        # Load reduction (window-aware)
        cur_hour = ts.hour
        rem=max(0.0,mkva-trig)
        if rem>1e-3:
            for lk in priority_order:
                if rem<=1e-3: break
                if lk not in loads or lkva.get(lk,0)<=0: continue
                eff = _eff_max_cut(loads[lk], cur_hour)
                if eff <= 0: continue
                mp=lkva[lk]*eff
                cut=min(rem,mp); lfac[lk]=1.0-cut/lkva[lk]
                pf=mkw/mkva if mkva>0 else 1.0; qf=mkvar/mkva if mkva>0 else 0.0
                mkw-=cut*pf; mkvar-=cut*qf; mkw=max(mkw,0.0)
                mkva=_calc_kva(mkw,mkvar); rem=max(0.0,mkva-trig)
                if cut>0.05:
                    acts.append({'type':'load_reduction','load':loads[lk].get('name',lk),
                        'load_key':lk,'cut_kva':round(cut,2),
                        'factor_pct':round(lfac[lk]*100,1),
                        'max_cut_pct':loads[lk].get('max_cut_pct',0.10)*100,
                        'note':f'{loads[lk].get("name",lk)} cut {round(cut,1)}kVA'})
        # Charge
        if bdis==0 and soc<bmax:
            emg=soc<bemerg; ceil=ec if emg else cc
            mchg=math.sqrt(max(ceil**2-mkvar**2,0.0))-mkw
            mchwh=max(mchg,0.0)*bat_efficiency*INTERVAL_H
            opk=(kva0<cu) or in_pre; norm_chg=opk and (soc<bfull) and not in_md
            if mchwh>0.001 and (emg or norm_chg):
                chwh=min(mchwh,chg_rate*INTERVAL_H,bmax-soc)
                if chwh>0.01:
                    chkw=chwh/bat_efficiency/INTERVAL_H; sb=soc; soc+=chwh; bchg=chkw
                    mkw+=chkw; mkva=_calc_kva(mkw,mkvar)
                    tstr='emergency' if emg else 'pre-MD boost' if in_pre else 'normal'
                    acts.append({'type':'battery_charge','load':'Battery',
                        'charge_kw':round(chkw,2),'soc_before_kwh':round(sb,1),
                        'soc_after_kwh':round(soc,1),'kva_ceiling':round(ceil,2),
                        'kva_after_charge':round(mkva,2),'charge_trigger':tstr,
                        'note':f'{"Emergency" if emg else "Pre-MD" if in_pre else "Normal"} charge '+
                               f'{round(chkw,1)}kW; SOC {round(sb,0):.0f}→{round(soc,0):.0f}kWh'})
        soc=max(bmin,min(bmax,soc))
        lmgd={k:lkva[k]*lfac[k] for k in load_keys}
        lcut={k:(lkva[k]-lmgd[k])*INTERVAL_H for k in load_keys}
        row_out={'timestamp':ts.isoformat(),'date':str(date),
                 'kva_original':round(kva0,2),'kw_original':round(kw,2),
                 'kvar_original':round(kvar,2),'kw_managed':round(mkw,2),
                 'kvar_managed':round(mkvar,2),
                 'battery_action_kw':round(bdis-bchg,2),
                 'battery_charge_kw':round(bchg,2),'battery_discharge_kw':round(bdis,2),
                 'battery_soc_kwh':round(soc,2),
                 'battery_soc_pct':round(soc/bmax*100,1) if bmax else 0,
                 'kva_managed':round(mkva,2),'target_peak':round(trig,2),
                 'ref_peak':round(ref,2),'charge_threshold_upper':round(cu,2),
                 'day_peak':round(day_pk.get(date,kva0),2),
                 'bat_capacity':round(bmax,2),'in_md_hours':in_md,'in_pre_md':in_pre,
                 'actions':acts}
        for k in load_keys:
            row_out[f'{k}_kva']=round(lkva[k],2); row_out[f'{k}_managed']=round(lmgd[k],2)
            row_out[f'{k}_factor']=round(lfac[k],3); row_out[f'{k}_kvah_cut']=round(lcut[k],3)
        results.append(row_out)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_pf(df: pd.DataFrame | None, default=0.85) -> float:
    """Estimate site power factor from historical data."""
    if df is None: return default
    try:
        mkw  = float(df['kw_net'].mean())
        mkvar= float(df['kvar_net'].mean())
        if mkw <= 0: return default
        pf = mkw / math.sqrt(mkw**2 + mkvar**2)
        return max(0.5, min(0.999, pf))
    except Exception:
        return default

def _forecast_to_mgr_df(fr, pf: float = 0.85) -> pd.DataFrame:
    """
    Convert Predictor MultiStepForecastResult → Manager-ready DataFrame.
    Manager input: {timestamp, kw_net, kvar_net, kva}
    kVAR estimated from power factor: kVAR = kW * tan(arccos(PF))
    """
    kw   = np.array(fr.median, dtype=float).clip(min=0)
    ts   = pd.to_datetime(list(fr.timestamps))
    tan_phi = math.tan(math.acos(max(0.01, min(0.9999, pf))))
    kvar = kw * tan_phi
    kva  = np.sqrt(kw**2 + kvar**2)
    return pd.DataFrame({
        'timestamp':  ts,
        'kw_import':  kw,   'kw_export':   np.zeros(len(kw)),
        'kvar_import':kvar,  'kvar_export': np.zeros(len(kw)),
        'kw_net':kw, 'kvar_net':kvar, 'kva':kva,
    })

def _build_mgr_summary(results: list, load_keys: list) -> dict:
    """Build Manager summary + day_summaries from results list."""
    day_sums: dict = {}
    for r in results:
        d = r['date']
        if d not in day_sums:
            day_sums[d] = {'orig_peak':0,'managed_peak':0,'orig_md_peak':0,
                           'managed_md_peak':0,'total_discharge_kwh':0,'total_charge_kwh':0}
            for k in load_keys: day_sums[d][f'{k}_total_kvah_cut']=0
        s=day_sums[d]
        s['orig_peak']   =max(s['orig_peak'],   r['kva_original'])
        s['managed_peak']=max(s['managed_peak'],r['kva_managed'])
        if r.get('in_md_hours'):
            s['orig_md_peak']   =max(s['orig_md_peak'],   r['kva_original'])
            s['managed_md_peak']=max(s['managed_md_peak'],r['kva_managed'])
        s['total_discharge_kwh']+=r['battery_discharge_kw']*0.5
        s['total_charge_kwh']   +=r['battery_charge_kw']*0.5
        for k in load_keys: s[f'{k}_total_kvah_cut']+=r.get(f'{k}_kvah_cut',0)
    oo=max((s['orig_peak']    for s in day_sums.values()),default=0)
    om=max((s['managed_peak'] for s in day_sums.values()),default=0)
    mo=max((s['orig_md_peak']    for s in day_sums.values()),default=0)
    mm=max((s['managed_md_peak'] for s in day_sums.values()),default=0)
    pr=(oo-om)/oo*100 if oo else 0
    mr=(mo-mm)/mo*100 if mo else 0
    return {
        'summary':{
            'original_peak_kva':round(oo,2),'managed_peak_kva':round(om,2),
            'peak_reduction_pct':round(pr,1),
            'original_md_peak_kva':round(mo,2),'managed_md_peak_kva':round(mm,2),
            'md_peak_reduction_pct':round(mr,1),
            'total_battery_discharge_kwh':round(sum(s['total_discharge_kwh'] for s in day_sums.values()),1),
            'total_battery_charge_kwh':   round(sum(s['total_charge_kwh']    for s in day_sums.values()),1),
            'avg_original_kva': round(sum(r['kva_original'] for r in results)/len(results),2) if results else 0,
            'avg_managed_kva':  round(sum(r['kva_managed']  for r in results)/len(results),2) if results else 0,
        },
        'day_summaries':day_sums,
    }

def _sf(v, d=0.0):
    try:
        f=float(v); return d if (math.isnan(f) or math.isinf(f)) else f
    except Exception: return d

def _pl_log(msg: str, level='info'):
    entry = {'ts': datetime.now().strftime('%H:%M:%S'), 'msg': msg, 'level': level}
    with _pl_lock:
        PL['log'].append(entry)
        if len(PL['log']) > 100:
            PL['log'] = PL['log'][-100:]

def _auto_bill():
    """Recompute bill from current Manager results (called inside pipeline tick)."""
    df = S.get('df')
    mgr = S.get('manager_results')
    if df is None: return
    tariff    = S.get('tariff_code', 'C1')
    icpt_sen  = S.get('bill_icpt', 0.0)
    nem_rate  = S.get('bill_nem',  0.31)
    sched_key = S.get('tariff_schedule_key', DEFAULT_SCHEDULE_KEY)
    if mgr:
        res_df = pd.DataFrame([{k:v for k,v in r.items() if k!='actions'} for r in mgr])
        res_df['timestamp'] = pd.to_datetime(res_df['timestamp'])
        df_opt  = res_df[['timestamp']].copy()
        df_opt['kw_import']   = res_df['kw_managed'].clip(lower=0)
        df_opt['kw_export']   = 0.0
        df_opt['kvar_import'] = res_df.get('kvar_managed', pd.Series(0.0,index=res_df.index)).clip(lower=0)
        df_opt['kvar_export'] = 0.0
        df_orig = df[['timestamp','kw_import','kw_export','kvar_import','kvar_export']].copy()
        show_cmp = True
    else:
        df_opt = df_orig = df[['timestamp','kw_import','kw_export','kvar_import','kvar_export']].copy()
        show_cmp = False
    def _calc(d_in):
        stats = compute_monthly_stats(d_in, schedule_key=sched_key)
        rows=[]
        for _, row in stats.iterrows():
            bill=calculate_bill(tariff,monthly_kwh=row['total_kwh'],
                peak_kwh=row['peak_kwh'],offpeak_kwh=row['offpeak_kwh'],
                max_demand_kw=row['max_demand_kw'],icpt_sen_per_kwh=icpt_sen,
                schedule_key=sched_key)
            nem=compute_nem_credit(row['export_kwh'],nem_rate)
            rows.append({'month':str(row['month']),
                'total_kwh':round(_sf(row['total_kwh']),0),
                'peak_kwh':round(_sf(row['peak_kwh']),0),
                'offpeak_kwh':round(_sf(row['offpeak_kwh']),0),
                'max_demand_kw':round(_sf(row['max_demand_kw']),1),
                'energy_rm':round(_sf(bill['energy_charge']),2),
                'md_rm':round(_sf(bill['md_charge']),2),
                'icpt_rm':round(_sf(bill['icpt_charge']),2),
                'kwtbb_rm':round(_sf(bill['kwtbb_charge']),2),
                'svc_tax_rm':round(_sf(bill['service_tax']),2),
                'nem_credit_rm':round(_sf(nem['nem_credit_rm']),2),
                'gross_bill_rm':round(_sf(bill['total_bill']),2),
                'net_bill_rm':round(_sf(bill['total_bill'])-_sf(nem['nem_credit_rm']),2)})
        return rows
    try:
        rows_opt  = _calc(df_opt)
        rows_orig = _calc(df_orig) if show_cmp else rows_opt
        tot_opt  = sum(r['net_bill_rm'] for r in rows_opt)
        tot_orig = sum(r['net_bill_rm'] for r in rows_orig)
        n = max(len(rows_opt),1)
        S['bill_rows']   = rows_opt
        S['bill_summary']= {
            'tariff':tariff,'tariff_name':TARIFF_META.get(tariff,{}).get('name',''),
            'schedule_key': sched_key,
            'schedule_name': SCHEDULES.get(sched_key, SCHEDULES[DEFAULT_SCHEDULE_KEY]).name,
            'show_comparison':show_cmp,
            'totals':{'before_rm':round(tot_orig,2),'after_rm':round(tot_opt,2),
                      'savings_rm':round(tot_orig-tot_opt,2),
                      'ann_savings_rm':round((tot_orig-tot_opt)*12/n,2)},
            'rows_opt':rows_opt,'rows_orig':rows_orig}
    except Exception as e:
        _pl_log(f'Bill error: {e}','error')

def _auto_powerreco():
    """Recompute PowerRECO from current Manager results (called inside pipeline tick)."""
    pr = S.get('powerreco_params')
    if pr is None: return
    try:
        solar = calculate_solar_sizing(pr['roof_area'], pr['panel_w'], pr['psh'])
        mgr = S.get('manager_results')
        if mgr:
            res_df = pd.DataFrame([{k:v for k,v in r.items() if k!='actions'} for r in mgr])
            res_df['battery_discharged_kwh'] = res_df['battery_discharge_kw']*0.5
            batt = calculate_battery_sizing(res_df)
        else:
            daily = _sf(solar.get('daily_generation_kwh_avg',0))
            batt  = {'min_capacity_kwh_commercial':max(10.0,round(daily*0.10/50)*50),
                     'md_reduction_kw':0.0,'n_days_analyzed':0,
                     'spike_note':'Estimated — run pipeline first.'}
        roi = calculate_roi(
            solar_kwp=solar['system_kwp'],
            battery_kwh=float(batt['min_capacity_kwh_commercial']),
            monthly_generation_kwh=solar['monthly_generation_kwh_avg'],
            md_reduction_kw=_sf(batt.get('md_reduction_kw',0)),
            self_consumption_pct=pr.get('self_cons',0.65),
            schedule_key=S.get('tariff_schedule_key', DEFAULT_SCHEDULE_KEY),
            tariff=S.get('tariff_code', 'C1'),
            solar_cost_per_kwp=pr.get('solar_cost',3500),
            battery_cost_per_kwh=pr.get('batt_cost',2500))
        S['roi'] = roi; S['solar_result'] = solar; S['batt_result'] = batt
    except Exception as e:
        _pl_log(f'PowerRECO error: {e}','error')

def _pipeline_tick():
    """Single assembly-line tick: LiveSim -> Forecast -> Manager -> Bill -> PowerRECO.

    The simulation reveals one new actual reading per tick, warm-starts
    the forecaster every retrain_every_n ticks, and updates the per-horizon
    bias + conformal calibration. So every tick produces a DIFFERENT
    forecast (the old static behaviour just re-emitted the same numbers).
    """
    fc = S.get('forecaster')
    sim = S.get('sim_state')
    if fc is None:
        _pl_log('No forecaster — skipping tick','warn')
        return
    if sim is None:
        _pl_log('No simulation state — train the forecaster to enable live mode','warn')
        return
    if sim.is_finished:
        _pl_log(f'Simulation finished at tick {sim.tick} — no more data to reveal','warn')
        PL['status'] = 'idle'
        return

    tick = PL['tick'] + 1
    PL['status'] = 'forecasting'
    _pl_log(f'Tick {tick} — revealing next reading + regenerating forecast…','info')

    # 1. Advance the live simulation: reveal next actual, fill prior
    #    forecasts, warm-start retrain when due, recompute calibration,
    #    issue a fresh 48-step forecast.
    tick_result = advance_one_tick(fc, sim)
    if tick_result.get('did_retrain'):
        ri = tick_result.get('retrain_info', {})
        _pl_log(
            f"  ↳ warm-start retrain at tick {ri.get('at_tick')} — "
            f"+{ri.get('samples_added')} samples · mean MAPE "
            f"{_sf(ri.get('mean_mape',0)):.2f}%", 'ok')
    fr = sim.current_forecast
    S['forecast_result'] = fr
    S['live_accuracy']   = compute_running_accuracy(sim)
    # 2. Convert to Manager df using historical PF
    pf = _estimate_pf(S.get('df'))
    mgr_df = _forecast_to_mgr_df(fr, pf)
    # Peak reference from historical data (with buffer)
    if S.get('df') is not None and 'kva' in S['df'].columns:
        peak_ref = float(S['df']['kva'].max()) * 1.05
    else:
        peak_ref = None
    # 3. Manager optimisation
    PL['status'] = 'optimizing'
    _pl_log(f'Tick {tick} — running AI Manager on {len(mgr_df)} forecast intervals…','info')
    # Loads — explicit /api/pipeline/configure wins; otherwise build from
    # active site profile, then layer Site Setup overrides (EV count/kW,
    # window, HVAC protected hours, etc.) on top.
    if S.get('pipeline_loads'):
        loads = S['pipeline_loads']
    else:
        active_profile = get_profile(S.get('site_profile_id', DEFAULT_PROFILE_ID))
        loads = profile_loads_for_manager(active_profile)
        ov = (S.get('site_overrides') or {}).get(active_profile.id, {})
        if loads.get('ev'):
            ev = loads['ev']
            if 'ev_count' in ov or 'ev_kw_each' in ov or 'ev_kind' in ov:
                _c  = int(ov.get('ev_count',   ev.get('ev_chargers',[{}])[0].get('count', 4)))
                _kw = float(ov.get('ev_kw_each', ev.get('ev_chargers',[{}])[0].get('kw_each', 22.0)))
                _kd = str(ov.get('ev_kind',    ev.get('ev_chargers',[{}])[0].get('kind', 'AC')))
                ev['ev_chargers'] = [{'count': _c, 'kw_each': _kw, 'kind': _kd}]
                ev['ev_total_kw'] = _c * _kw
            if 'ev_window_start' in ov or 'ev_window_end' in ov:
                ws = int(ov.get('ev_window_start', ev.get('allowed_window',[18,8])[0]))
                we = int(ov.get('ev_window_end',   ev.get('allowed_window',[18,8])[1]))
                ev['allowed_window'] = [ws, we]
        if loads.get('hvac'):
            hv = loads['hvac']
            if 'hvac_protect_start' in ov or 'hvac_protect_end' in ov:
                ps = int(ov.get('hvac_protect_start', hv.get('protected_window',[9,17])[0]))
                pe = int(ov.get('hvac_protect_end',   hv.get('protected_window',[9,17])[1]))
                hv['protected_window'] = [ps, pe]
            if 'hvac_max_cut_pct' in ov:
                hv['max_cut_pct'] = float(ov['hvac_max_cut_pct']) / 100.0
    priority = S.get('pipeline_priority') or list(loads.keys())
    results = _mgr_run(mgr_df, loads,
                       S.get('pipeline_battery_kwh',200),
                       priority,
                       S.get('pipeline_peak_target',0.85),
                       S.get('pipeline_charge_upper',0.70),
                       c_rate          = S.get('pipeline_c_rate',0.5),
                       initial_soc_pct = S.get('pipeline_init_soc',0.50),
                       bat_efficiency  = S.get('pipeline_bat_eff',0.95),
                       peak_reference_kva = peak_ref,
                       lookahead_intervals= S.get('pipeline_lookahead',16),
                       md_start_hour   = S.get('pipeline_md_start',14),
                       md_end_hour     = S.get('pipeline_md_end',22),
                       pre_md_hours    = S.get('pipeline_pre_md_hours',2))
    S['manager_results']  = results
    S['manager_loads']    = loads
    mgr_meta = _build_mgr_summary(results, list(loads.keys()))
    S['manager_summary']  = mgr_meta['summary']
    S['manager_day_sums'] = mgr_meta['day_summaries']
    # 4. Bill
    PL['status'] = 'billing'
    _auto_bill()
    # 5. PowerRECO
    PL['status'] = 'powerreco'
    _auto_powerreco()
    PL['tick'] = tick
    PL['last_tick_ts'] = datetime.now().isoformat()
    PL['status'] = 'idle'
    orig_pk = _sf(S['manager_summary'].get('original_peak_kva',0))
    mgd_pk  = _sf(S['manager_summary'].get('managed_peak_kva',0))
    red_pct = _sf(S['manager_summary'].get('peak_reduction_pct',0))
    _pl_log(f'Tick {tick} done — orig peak {orig_pk:.1f} kVA -> managed {mgd_pk:.1f} kVA ({red_pct:.1f}% reduction)','ok')

def _pipeline_loop(stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            _pipeline_tick()
        except Exception as e:
            PL['status'] = 'error'
            _pl_log(f'Pipeline error: {e}','error')
        stop_evt.wait(PL.get('interval_seconds',30))


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
def _ok(d): return jsonify({'ok':True,**d}),200
def _err(m,code=400): return jsonify({'ok':False,'error':m}),code

@app.route('/')
def index():
    return send_from_directory(app.static_folder,'index.html')

@app.route('/api/status')
def api_status():
    return _ok({
        'stages':{'data':S.get('df') is not None,'predictor':S.get('forecaster') is not None,
                  'forecast':S.get('forecast_result') is not None,
                  'manager':S.get('manager_results') is not None,
                  'calculator':S.get('bill_rows') is not None,
                  'powerreco':S.get('roi') is not None},
        'tariff_code':S.get('tariff_code','C1'),
        'tariff_meta':{k:{'name':v['name']} for k,v in TARIFF_META.items()},
        'tariff_schedule_key': S.get('tariff_schedule_key', DEFAULT_SCHEDULE_KEY),
        'site_profile_id': S.get('site_profile_id', DEFAULT_PROFILE_ID),
        'site_profiles':   list_profiles(),
        'solar_kwp':S.get('solar_kwp',0),
    })


@app.route('/api/site/configure', methods=['POST'])
def api_site_configure():
    """Update editable per-site parameters (location, solar/battery capacity,
    EV details, costs, etc.) and optionally trigger a fresh train so the next
    pipeline tick uses the new values.

    Body:
      {
        site_id?:         "sol|e|sun|mi2|custom",    # default = active
        overrides:        {field: value, ...},
        use_real_weather: bool (default true),
        retrain:          bool — if true and data is already uploaded,
                                  re-runs the train endpoint with the new params.
      }
    Override fields accepted (any subset):
      lat, lon, timezone,
      solar_kwp, roof_area_m2, panel_w, psh, solar_cost,
      battery_kwh, c_rate, batt_cost,
      md_target_pct, charge_upper_pct, init_soc_pct,
      ev_count, ev_kw_each, ev_kind, ev_window_start, ev_window_end,
      hvac_protect_start, hvac_protect_end, hvac_max_cut_pct,
      tariff_code, icpt_sen, nem_rate, budget_rm
    """
    b = request.get_json(silent=True) or {}
    site_id = b.get('site_id', S.get('site_profile_id', DEFAULT_PROFILE_ID))
    if site_id not in PROFILES:
        return _err(f'Unknown site profile id: {site_id}', 400)
    overrides = b.get('overrides') or {}
    if not isinstance(overrides, dict):
        return _err('"overrides" must be a dict', 400)

    S.setdefault('site_overrides', {})
    cur = S['site_overrides'].get(site_id, {})
    cur.update(overrides)
    S['site_overrides'][site_id] = cur
    S['site_profile_id'] = site_id
    # Persist to disk — file location is data/site_overrides.json
    save_overrides(S['site_overrides'])

    # Mirror tariff_code into the top-level S key so /api/calculator/bill
    # picks it up on the next tick without an extra POST.
    if 'tariff_code' in overrides:
        S['tariff_code'] = overrides['tariff_code']
    if 'icpt_sen' in overrides:  S['bill_icpt'] = float(overrides['icpt_sen'])
    if 'nem_rate' in overrides:  S['bill_nem']  = float(overrides['nem_rate'])

    # Reset pipeline-load overrides so the next tick uses freshly-derived
    # loads (with the new EV / HVAC params layered onto the site profile).
    S.pop('pipeline_loads', None)
    S.pop('pipeline_priority', None)

    if 'use_real_weather' in b:
        S['use_real_weather'] = bool(b['use_real_weather'])
    else:
        S.setdefault('use_real_weather', True)

    retrain = bool(b.get('retrain', False))
    retrain_info = None
    if retrain and S.get('df') is not None:
        # Mirror train endpoint logic with overrides applied
        try:
            profile = get_profile(site_id)
            fc_kwargs = profile_predictor_kwargs(profile)
            # Apply lat/lon/timezone overrides; turn off real weather if flag off
            if S.get('use_real_weather', True):
                fc_kwargs['lat']      = float(cur.get('lat',      profile.lat))
                fc_kwargs['lon']      = float(cur.get('lon',      profile.lon))
                fc_kwargs['timezone'] = str(  cur.get('timezone', profile.timezone))
            else:
                fc_kwargs['lat'] = None
                fc_kwargs['lon'] = None
            capacity_kwp = float(cur.get('solar_kwp',
                profile.expected_solar_kwp or S.get('solar_kwp', 0) or 0))
            fc = DirectMultiStepForecaster(capacity_kwp=capacity_kwp, **fc_kwargs)
            train_df, future_df = split_historical_for_simulation(
                S['df'][['timestamp','kw_import','kw_export','kvar_import','kvar_export']]
                  if all(c in S['df'].columns for c in ('kw_export','kvar_import','kvar_export'))
                  else S['df'],
                train_fraction=0.7,
            )
            m = fc.fit(train_df, verbose=False)
            S['forecaster'] = fc
            S['sim_state'] = initialize_simulation(fc, future_df,
                                                    retrain_every_n=12,
                                                    warm_start_rounds=50)
            S['forecast_result'] = S['sim_state'].current_forecast
            S['live_accuracy']   = compute_running_accuracy(S['sim_state'])
            retrain_info = {
                'n_models': m['n_models_trained'],
                'mean_mape': round(_sf(m.get('mean_mape',0)), 3),
                'train_rows':  len(train_df),
                'future_rows': len(future_df),
            }
        except Exception as e:
            traceback.print_exc()
            return _err(f'Retrain after configure failed: {e}')

    return _ok({
        'site_id':         site_id,
        'overrides_count': len(cur),
        'overrides':       cur,
        'retrained':       retrain_info is not None,
        'retrain':         retrain_info,
    })


@app.route('/api/site/select', methods=['POST'])
def api_site_select():
    """Pick a site profile. Returns the resolved profile dict so the
    frontend can pre-fill its forms (battery_kwh, EV chargers, etc.)."""
    b = request.get_json(silent=True) or {}
    sid = b.get('id', DEFAULT_PROFILE_ID)
    if sid not in PROFILES:
        return _err(f'Unknown site profile id: {sid}', 400)
    S['site_profile_id'] = sid
    # Reset pipeline-load overrides so the new profile's loads take effect
    # on the next tick (user can override again via /api/pipeline/configure).
    S.pop('pipeline_loads', None)
    S.pop('pipeline_priority', None)
    prof = get_profile(sid)
    return _ok({
        'id':   prof.id,
        'name': prof.name,
        'description': prof.description,
        'lat': prof.lat, 'lon': prof.lon,
        'expected_tariff':   prof.expected_tariff,
        'expected_solar_kwp': prof.expected_solar_kwp,
        'expected_roof_area_m2': prof.expected_roof_area_m2,
        'battery_kwh':       prof.battery_kwh,
        'md_target_pct':     prof.md_target_pct,
        'charge_upper_pct':  prof.charge_upper_pct,
        'predictor': {
            'n_estimators':  prof.predictor.n_estimators,
            'learning_rate': prof.predictor.learning_rate,
            'bias_window':   prof.predictor.bias_window,
        },
        'loads': profile_loads_for_manager(prof),
    })

# ── Upload ────────────────────────────────────────────────────────────────────
@app.route('/api/upload',methods=['POST'])
def api_upload():
    if 'file' not in request.files: return _err('No file')
    f=request.files['file']; raw=f.read()
    try: df=_parse_upload(raw,f.filename)
    except Exception as e: return _err(f'Parse error: {e}')
    S['df']=df; S['manager_results']=None; S['bill_rows']=None
    S['forecast_result']=None; S['forecaster']=None; S['roi']=None
    S['sim_state']=None  # cleared — new upload triggers a fresh train + sim
    S.setdefault('tariff_schedule_key', DEFAULT_SCHEDULE_KEY)
    S.setdefault('site_profile_id',    DEFAULT_PROFILE_ID)
    try: tc,tr,_=auto_detect_tariff(df); S['tariff_code']=tc
    except Exception: tc='C1'; S['tariff_code']=tc
    try:
        hs,sr=detect_has_solar(df); skwp=0.0
        if hs: sol=estimate_solar_capacity_kwp(df); skwp=_sf(sol.get('capacity_kwp',0))
        S['solar_kwp']=skwp
    except Exception: hs=False; sr=''; skwp=0.0; S['solar_kwp']=0.0
    dates=sorted(df['timestamp'].dt.date.astype(str).unique().tolist())
    records=[{'timestamp':r['timestamp'].isoformat(),'kva':round(_sf(r['kva']),2),
              'kw_net':round(_sf(r['kw_net']),2),'kvar_net':round(_sf(r['kvar_net']),2),
              'kw_import':round(_sf(r['kw_import']),2),'kw_export':round(_sf(r['kw_export']),2)}
             for _,r in df.iterrows()]
    S['file_summary']={'rows':len(df),'days':len(dates),'start':dates[0] if dates else '',
        'end':dates[-1] if dates else '','max_kw_import':round(_sf(df['kw_import'].max()),2),
        'mean_kw_import':round(_sf(df['kw_import'].mean()),2),'peak_kva':round(_sf(df['kva'].max()),2)}
    return _ok({'success':True,'records':records,'dates':dates,
                'total_intervals':len(records),'peak_kva':round(_sf(df['kva'].max()),2),
                'avg_kva':round(_sf(df['kva'].mean()),2),
                'summary':S['file_summary'],'tariff':tc,
                'tariff_meta':{k:{'name':v['name']} for k,v in TARIFF_META.items()},
                'solar':{'has_solar':hs,'reason':sr,'capacity_kwp':skwp}})

# ── Predictor — train ─────────────────────────────────────────────────────────
@app.route('/api/predictor/train',methods=['POST'])
def api_predictor_train():
    """
    Train the forecaster on the first `train_fraction` of the uploaded data
    and reserve the rest as a future stream for the live simulation.

    This is what enables "live updating" pipeline ticks: each tick reveals
    the next future reading to the model, warm-start retrains every 12
    ticks, and produces a fresh forecast → Manager → bill → ROI chain.
    """
    if S.get('df') is None: return _err('Upload data first',400)
    b=request.get_json(silent=True) or {}
    kwp=float(b.get('capacity_kwp',S.get('solar_kwp',0)))
    n=int(b.get('n_estimators',250)); lr=float(b.get('learning_rate',0.05))
    train_fraction = float(b.get('train_fraction', 0.7))
    train_fraction = max(0.3, min(0.95, train_fraction))
    retrain_every  = int(b.get('retrain_every_n', 12))   # warm-start every 6h of revealed data
    warm_rounds    = int(b.get('warm_start_rounds', 50))

    # Site profile drives per-site forecaster tuning (h_cap_*, bias_window,
    # min_child_samples).  User overrides via /api/site/configure layer on top.
    profile = get_profile(S.get('site_profile_id', DEFAULT_PROFILE_ID))
    site_id = profile.id
    overrides = (S.get('site_overrides') or {}).get(site_id, {})
    # Use_real_weather: explicit body param > stored state > default True
    use_real_weather = bool(b.get('use_real_weather',
                                    S.get('use_real_weather', True)))
    S['use_real_weather'] = use_real_weather

    try:
        # Split historical: past = train, future = the stream we'll replay.
        train_df, future_df = split_historical_for_simulation(
            S['df'][['timestamp','kw_import','kw_export','kvar_import','kvar_export']]
              if all(c in S['df'].columns for c in ('kw_export','kvar_import','kvar_export'))
              else S['df'],
            train_fraction=train_fraction,
        )

        fc_kwargs = profile_predictor_kwargs(profile)
        fc_kwargs.update({'n_estimators': n, 'learning_rate': lr})
        # Lat/lon from override layer, falling back to profile preset
        if use_real_weather:
            fc_kwargs['lat']      = float(overrides.get('lat',      profile.lat))
            fc_kwargs['lon']      = float(overrides.get('lon',      profile.lon))
            fc_kwargs['timezone'] = str(  overrides.get('timezone', profile.timezone))
        else:
            fc_kwargs['lat'] = None
            fc_kwargs['lon'] = None
        # Solar capacity override wins over the request body's `capacity_kwp`
        if 'solar_kwp' in overrides:
            kwp = float(overrides['solar_kwp'])
        fc = DirectMultiStepForecaster(capacity_kwp=kwp, **fc_kwargs)
        m = fc.fit(train_df, verbose=False)
        S['forecaster'] = fc

        # Spin up the live simulation. Each pipeline tick consumes one row
        # from future_df, warm-starts retrain every retrain_every_n ticks,
        # and updates the forecaster's bias_profile + conformal_half_width.
        S['sim_state'] = initialize_simulation(
            fc, future_df,
            retrain_every_n=retrain_every,
            warm_start_rounds=warm_rounds,
        )
        S['forecast_result'] = S['sim_state'].current_forecast
        S['live_accuracy']   = compute_running_accuracy(S['sim_state'])

        return _ok({
            'n_models':     m['n_models_trained'],
            'mean_mape':    round(_sf(m.get('mean_mape',0)),3),
            'mape_at_h24':  round(_sf(m.get('mape_at_h24',0)),3),
            'train_rows':   len(train_df),
            'future_rows':  len(future_df),
            'live_ready':   True,
            'retrain_every_n': retrain_every,
        })
    except Exception as e:
        traceback.print_exc()
        return _err(f'Training failed: {e}')

# ── Pipeline configure ────────────────────────────────────────────────────────
@app.route('/api/pipeline/configure',methods=['POST'])
def api_pipeline_configure():
    b=request.get_json(silent=True) or {}
    # Loads
    loads_raw=b.get('loads',None)
    if loads_raw:
        S['pipeline_loads']={k:{'name':v.get('name',k),'proportion':float(v.get('proportion',0)),
            'max_cut_pct':float(v.get('max_cut_pct',0.10)),'color':v.get('color','#00d4ff')}
            for k,v in loads_raw.items()}
    # Priority order (JS sends low-priority first = cut first)
    if 'priority_order' in b: S['pipeline_priority']=b['priority_order']
    # Battery params
    for key,default in [('battery_kwh',200),('peak_target',0.85),('charge_upper',0.70),
                        ('c_rate',0.5),('init_soc',0.50),('bat_eff',0.95),
                        ('lookahead',16),('md_start',14),('md_end',22),('pre_md_hours',2)]:
        field=f'pipeline_{key}'; src=key.replace('_',' ').replace('pipeline ','')
        if key in b: S[field]=float(b[key]) if key not in ('lookahead','md_start','md_end','pre_md_hours') else int(b[key])
    if 'peak_ref_kva' in b: S['pipeline_peak_ref']=float(b['peak_ref_kva']) if b['peak_ref_kva'] else None
    # Interval
    if 'interval_seconds' in b: PL['interval_seconds']=max(5,int(b['interval_seconds']))
    # Bill params
    if 'tariff' in b: S['tariff_code']=b['tariff']
    if 'icpt_sen' in b: S['bill_icpt']=float(b['icpt_sen'])
    if 'nem_rate'  in b: S['bill_nem'] =float(b['nem_rate'])
    # Tariff schedule (drives BOTH Calculator and ROI engine)
    if 'tariff_schedule_key' in b:
        k = b['tariff_schedule_key']
        if k in SCHEDULES:
            S['tariff_schedule_key'] = k
    # PowerRECO params
    if any(k in b for k in ('roof_area','panel_w','psh','solar_cost','batt_cost','self_cons','use_new_tariff')):
        S['powerreco_params']={
            'roof_area':   float(b.get('roof_area',  500)),
            'panel_w':     int(  b.get('panel_w',    415)),
            'psh':         float(b.get('psh',         4.5)),
            'solar_cost':  float(b.get('solar_cost', 3500)),
            'batt_cost':   float(b.get('batt_cost',  2500)),
            'self_cons':   float(b.get('self_cons',   0.65)),
            'use_new_tariff': bool(b.get('use_new_tariff',True)),
        }
    return _ok({'configured':True,'interval_seconds':PL['interval_seconds']})

# ── Pipeline start ────────────────────────────────────────────────────────────
@app.route('/api/pipeline/start',methods=['POST'])
def api_pipeline_start():
    if S.get('forecaster') is None: return _err('Train the Predictor first',400)
    if PL.get('running'): return _ok({'already_running':True})
    b=request.get_json(silent=True) or {}
    if 'interval_seconds' in b: PL['interval_seconds']=max(5,int(b['interval_seconds']))
    stop_evt=threading.Event()
    PL['_stop_evt']=stop_evt; PL['running']=True; PL['status']='idle'
    PL['log']=[]; PL['tick']=0
    _pl_log('Pipeline started','ok')
    t=threading.Thread(target=_pipeline_loop,args=(stop_evt,),daemon=True)
    PL['_thread']=t; t.start()
    return _ok({'started':True,'interval_seconds':PL['interval_seconds']})

# ── Pipeline stop ─────────────────────────────────────────────────────────────
@app.route('/api/pipeline/stop',methods=['POST'])
def api_pipeline_stop():
    evt=PL.get('_stop_evt')
    if evt: evt.set()
    PL['running']=False; PL['status']='idle'
    _pl_log('Pipeline stopped','warn')
    return _ok({'stopped':True})

# ── Pipeline status (main polling endpoint) ───────────────────────────────────
@app.route('/api/pipeline/status')
def api_pipeline_status():
    fr  = S.get('forecast_result')
    mgr = S.get('manager_results')
    fc  = S.get('forecaster')
    # Forecast chart data
    fc_data=None
    if fr is not None and fc is not None:
        try:
            hist=fc.history.tail(96)
            fc_data={'hist_labels':hist['timestamp'].astype(str).tolist(),
                     'hist_kw':[round(_sf(v),2) for v in hist['kw_import']],
                     'fc_labels':[str(t) for t in fr.timestamps],
                     'fc_median':[round(_sf(v),2) for v in fr.median],
                     'fc_p10':[round(_sf(v),2) for v in fr.p10],
                     'fc_p90':[round(_sf(v),2) for v in fr.p90]}
        except Exception: pass
    # Manager chart data
    mgr_data=None
    if mgr:
        try:
            res_df=pd.DataFrame([{k:v for k,v in r.items() if k!='actions'} for r in mgr])
            res_df['timestamp']=pd.to_datetime(res_df['timestamp'])
            step=max(1,len(res_df)//300); sub=res_df.iloc[::step]
            load_keys=[k for k in res_df.columns if k.endswith('_managed') and 'battery' not in k]
            lk_clean=[k.replace('_managed','') for k in load_keys]
            loads_info=S.get('pipeline_loads') or S.get('manager_loads') or {}
            mgr_data={'labels':sub['timestamp'].astype(str).tolist(),
                      'kva_original':[round(_sf(v),2) for v in sub['kva_original']],
                      'kva_managed': [round(_sf(v),2) for v in sub['kva_managed']],
                      'battery_soc_pct':[round(_sf(v),1) for v in sub['battery_soc_pct']],
                      'battery_soc_kwh':[round(_sf(v),2) for v in sub['battery_soc_kwh']],
                      'battery_charge_kw':[round(_sf(v),2) for v in sub['battery_charge_kw']],
                      'battery_discharge_kw':[round(_sf(v),2) for v in sub['battery_discharge_kw']],
                      'target_peak':[round(_sf(v),2) for v in sub['target_peak']],
                      'charge_threshold_upper':[round(_sf(v),2) for v in sub.get('charge_threshold_upper',pd.Series(0.0,index=sub.index))],
                      'in_md_hours':sub.get('in_md_hours',pd.Series(False,index=sub.index)).tolist(),
                      'load_keys':lk_clean,'loads':loads_info,
                      'summary':S.get('manager_summary',{}),
                      'day_summaries':S.get('manager_day_sums',{})}
            for k,kc in zip(load_keys,lk_clean):
                mgr_data[kc+'_managed']=[round(_sf(v),2) for v in sub.get(k,pd.Series(0.0,index=sub.index))]
        except Exception as e:
            _pl_log(f'Status build error: {e}','error')
    # ROI
    roi_data=None
    if S.get('roi') and S.get('solar_result') and S.get('batt_result'):
        r=S['roi']; sl=S['solar_result']; bt=S['batt_result']
        roi_data={'solar':{'system_kwp':sl['system_kwp'],'n_panels':sl['n_panels'],
                            'panel_wattage_w':sl['panel_wattage_w'],
                            'annual_generation_kwh':sl['annual_generation_kwh'],
                            'usable_area_m2':sl['usable_area_m2'],
                            'daily_generation_kwh_avg':round(_sf(sl['daily_generation_kwh_avg']),1),
                            'month_labels':sl['month_labels'],
                            'monthly_breakdown_kwh':[round(_sf(v),1) for v in sl['monthly_breakdown_kwh']]},
                  'battery':{'min_kwh_commercial':float(bt['min_capacity_kwh_commercial']),
                             'md_reduction_kw':round(_sf(bt.get('md_reduction_kw',0)),1),
                             'n_days_analyzed':int(bt.get('n_days_analyzed',0)),
                             'spike_note':bt.get('spike_note','')},
                  'roi':{'total_capex_rm':round(_sf(r['total_capex_rm']),0),
                         'simple_payback_years':r['simple_payback_years'],
                         'npv_25yr_rm':round(_sf(r['npv_25yr_rm']),0),
                         'irr_pct':r.get('irr_pct'),
                         'annual_energy_savings_rm':round(_sf(r['annual_energy_savings_rm']),0),
                         'annual_md_savings_rm':round(_sf(r['annual_md_savings_rm']),0),
                         'annual_nem_credit_rm':round(_sf(r['annual_nem_credit_rm']),0),
                         'co2_offset_tonnes_yr':round(_sf(r['co2_offset_tonnes_yr']),1),
                         'md_rate_used':r.get('md_rate_used',''),
                         'cumulative_npv':[round(_sf(v),0) for v in r['cumulative_npv']]}}
    # Live simulation progress + accuracy (drives the "smarter every day"
    # KPIs the dashboard renders).
    sim = S.get('sim_state')
    sim_progress = None
    if sim is not None:
        sim_progress = {
            'tick':          sim.tick,
            'total_ticks':   sim.total_ticks,
            'progress_pct':  round(sim.progress * 100, 1),
            'retrains':      len(sim.retrain_log),
            'last_retrain':  sim.retrain_log[-1] if sim.retrain_log else None,
            'is_finished':   bool(sim.is_finished),
            'bias_h1':       round(_sf(sim.current_bias), 2),
        }
    live_accuracy = S.get('live_accuracy')
    if live_accuracy:
        live_accuracy = {
            'n_verified':       live_accuracy.get('n_verified'),
            'overall_mae':      round(_sf(live_accuracy.get('overall_mae',0)), 2)
                                  if live_accuracy.get('overall_mae') is not None else None,
            'overall_mape':     round(_sf(live_accuracy.get('overall_mape',0)), 2)
                                  if live_accuracy.get('overall_mape') is not None else None,
            'within_80ci_pct':  round(_sf(live_accuracy.get('within_80ci_pct',0)), 1)
                                  if live_accuracy.get('within_80ci_pct') is not None else None,
            # Per-horizon table (kept compact — h1, h12, h24, h48 are the
            # operationally important ones for an MD-shaving demo)
            'by_horizon': {
                str(h): {
                    'mae':  round(_sf(v['mae']),  2),
                    'mape': round(_sf(v['mape']), 2),
                    'n':    v['n'],
                }
                for h, v in (live_accuracy.get('by_horizon') or {}).items()
                if h in (1, 12, 24, 48)
            },
        }

    return _ok({'pipeline':{'running':PL.get('running',False),'status':PL.get('status','idle'),
                             'tick':PL.get('tick',0),'interval_seconds':PL.get('interval_seconds',30),
                             'last_tick_ts':PL.get('last_tick_ts'),'log':PL.get('log',[])[-30:]},
                'stages':{'data':S.get('df') is not None,'predictor':S.get('forecaster') is not None,
                          'forecast':fr is not None,'manager':mgr is not None,
                          'calculator':S.get('bill_rows') is not None,'powerreco':S.get('roi') is not None},
                'forecast':fc_data,'manager':mgr_data,'bill':S.get('bill_summary'),
                'roi':roi_data,
                'tariff_meta':{k:{'name':v['name']} for k,v in TARIFF_META.items()},
                'tariff_code':S.get('tariff_code','C1'),
                'tariff_schedule_key': S.get('tariff_schedule_key', DEFAULT_SCHEDULE_KEY),
                'tariff_schedules': {k: {'name': v.name, 'effective_from': v.effective_from}
                                     for k, v in SCHEDULES.items()},
                'site_profile_id': S.get('site_profile_id', DEFAULT_PROFILE_ID),
                'site_profiles':   list_profiles(),
                'sim_progress':  sim_progress,
                'live_accuracy': live_accuracy,
                'file_summary':S.get('file_summary',{})})

# ── Bill settings update (standalone) ────────────────────────────────────────
@app.route('/api/calculator/bill',methods=['POST'])
def api_calc_bill():
    b=request.get_json(silent=True) or {}
    if 'tariff'  in b: S['tariff_code']=b['tariff']
    if 'icpt_sen'in b: S['bill_icpt']=float(b['icpt_sen'])
    if 'nem_rate' in b: S['bill_nem'] =float(b['nem_rate'])
    _auto_bill()
    return _ok({'bill':S.get('bill_summary',{})})

# ── PowerRECO settings update (standalone) ────────────────────────────────────
@app.route('/api/powerreco/optimize', methods=['POST'])
def api_powerreco_optimize():
    """Sweep (solar × battery) grid against the active tariff schedule
    and return the best NPV / payback / under-budget configurations."""
    if not S.get('solar_result') or not S.get('batt_result'):
        return _err('Run PowerRECO once first to seed solar + battery sizing.', 400)
    b = request.get_json(silent=True) or {}
    pr = S.get('powerreco_params') or {}
    budget = b.get('budget_rm')
    try:
        opt = find_optimum_from_existing_run(
            solar_sizing=S['solar_result'],
            battery_sizing=S['batt_result'],
            schedule_key=S.get('tariff_schedule_key', DEFAULT_SCHEDULE_KEY),
            tariff=S.get('tariff_code', 'C1'),
            self_consumption_pct=pr.get('self_cons', 0.65),
            solar_cost_per_kwp=pr.get('solar_cost', 3500.0),
            battery_cost_per_kwh=pr.get('batt_cost', 2500.0),
            budget_rm=float(budget) if budget else None,
            solar_grid_kwp=b.get('solar_grid_kwp'),
            battery_grid_kwh=b.get('battery_grid_kwh'),
        )
        S['sizing_sweep'] = opt
        return _ok({'optimum': opt})
    except Exception as e:
        traceback.print_exc()
        return _err(f'Optimizer failed: {e}')


@app.route('/api/powerreco/run',methods=['POST'])
def api_powerreco_run():
    b=request.get_json(silent=True) or {}
    S['powerreco_params']={
        'roof_area':  float(b.get('roof_area',  500)),
        'panel_w':    int(  b.get('panel_w',    415)),
        'psh':        float(b.get('psh',         4.5)),
        'solar_cost': float(b.get('solar_cost', 3500)),
        'batt_cost':  float(b.get('batt_cost',  2500)),
        'self_cons':  float(b.get('self_cons',   0.65)),
        'use_new_tariff':bool(b.get('use_new_tariff',True)),
    }
    _auto_powerreco()
    roi_data=None
    if S.get('roi') and S.get('solar_result') and S.get('batt_result'):
        r=S['roi']; sl=S['solar_result']; bt=S['batt_result']
        roi_data={'solar':{'system_kwp':sl['system_kwp'],'n_panels':sl['n_panels'],
                            'panel_wattage_w':sl['panel_wattage_w'],
                            'annual_generation_kwh':sl['annual_generation_kwh'],
                            'usable_area_m2':sl['usable_area_m2'],
                            'daily_generation_kwh_avg':round(_sf(sl['daily_generation_kwh_avg']),1),
                            'month_labels':sl['month_labels'],
                            'monthly_breakdown_kwh':[round(_sf(v),1) for v in sl['monthly_breakdown_kwh']]},
                  'battery':{'min_kwh_commercial':float(bt['min_capacity_kwh_commercial']),
                             'md_reduction_kw':round(_sf(bt.get('md_reduction_kw',0)),1),
                             'n_days_analyzed':int(bt.get('n_days_analyzed',0)),
                             'spike_note':bt.get('spike_note','')},
                  'roi':{'total_capex_rm':round(_sf(r['total_capex_rm']),0),
                         'simple_payback_years':r['simple_payback_years'],
                         'npv_25yr_rm':round(_sf(r['npv_25yr_rm']),0),
                         'irr_pct':r.get('irr_pct'),
                         'annual_energy_savings_rm':round(_sf(r['annual_energy_savings_rm']),0),
                         'annual_md_savings_rm':round(_sf(r['annual_md_savings_rm']),0),
                         'annual_nem_credit_rm':round(_sf(r['annual_nem_credit_rm']),0),
                         'co2_offset_tonnes_yr':round(_sf(r['co2_offset_tonnes_yr']),1),
                         'md_rate_used':r.get('md_rate_used',''),
                         'cumulative_npv':[round(_sf(v),0) for v in r['cumulative_npv']]}}
    return _ok({'roi':roi_data})

if __name__=='__main__':
    print('\nBOLT Integrated  |  http://localhost:5000\n')
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
