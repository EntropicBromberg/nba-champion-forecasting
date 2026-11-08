"""Read-only, explicitly experimental comparisons of published real model outputs."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from scripts.publish_forecast import public_forecast
from scripts.publish_experiments import fingerprint

ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/EntropicBromberg/nba-champion-forecasting/main/reports/experimental_forecasts.json'
TITLES={'uncertainty':'Team-strength uncertainty','injury':'Injury adjustment','stress':'Injury + recovery stress'}
DESCRIPTIONS={
    'uncertainty':'Allows team quality to vary across simulated season paths, then keeps each draw consistent through games and playoff series.',
    'injury':'Adjusts the primary model for reported player absences until their estimated return dates, using prior-season player value and projected minutes.',
    'stress':'Combines uncertain team quality with player-role and roster-depth penalties, then lets the assumed impact fade gradually after return.',
}


def validate_bundle(bundle,primary):
    if bundle['schema_version']!=1 or bundle['primary_fingerprint']!=fingerprint(primary):
        raise ValueError('Experimental outputs do not match the primary forecast')
    if set(bundle['models'])!=set(TITLES):raise ValueError('Unexpected scenario identities')
    primary_by_team={r['team']:r for r in primary['odds']}
    for identifier,model in bundle['models'].items():
        if model['id']!=identifier or model['status'] not in ['available','unavailable']:
            raise ValueError('Invalid scenario status')
        if model['status']=='unavailable':continue
        if model['games_through']!=primary['games_through']:raise ValueError('Mismatched game cutoff')
        cleaned=public_forecast(dict(season=primary['season'],trained_through=primary['trained_through'],
            training_seasons=25,checked_at=model['computed_at'],games_through=model['games_through'],
            simulations=model['simulations'],odds=model['odds']))
        if {r['team']:r['name'] for r in cleaned['odds']}!={t:r['name'] for t,r in primary_by_team.items()}:
            raise ValueError('Mismatched teams')
        expected_baseline='uncertainty' if identifier=='stress' else 'primary'
        if model['baseline']!=expected_baseline:raise ValueError('Wrong comparison baseline')
        baseline=primary_by_team
        if identifier=='stress':
            uncertainty=bundle['models']['uncertainty']
            if uncertainty['status']!='available':raise ValueError('Missing stress baseline')
            baseline={r['team']:r for r in uncertainty['odds']}
            if not 0<=model['settings']['severity']<=4 or not 1<=model['settings']['recovery_days']<=180:
                raise ValueError('Invalid stress settings')
        for row in model['odds']:
            p=float(row['baseline_title_probability'])
            if not math.isfinite(p) or not math.isclose(p,baseline[row['team']]['title_probability'],abs_tol=1e-8):
                raise ValueError('Mismatched paired baseline')
    return bundle


@st.cache_data(ttl=1800,show_spinner=False)
def load_bundle(primary):
    local=None
    try:local=validate_bundle(json.loads((ROOT/'reports/experimental_forecasts.json').read_text()),primary)
    except (OSError,ValueError,KeyError,TypeError):pass
    try:
        response=requests.get(URL,timeout=10);response.raise_for_status()
        remote=validate_bundle(response.json(),primary)
        # Keep a later local result if a CDN briefly returns an older publication.
        if local:
            for key,model in local['models'].items():
                other=remote['models'][key]
                if model['status']=='available' and (other['status']!='available' or
                    datetime.fromisoformat(model['computed_at'])>datetime.fromisoformat(other['computed_at'])):
                    remote['models'][key]=model
        return validate_bundle(remote,primary),None
    except (requests.RequestException,ValueError,KeyError,TypeError):
        return local,('Showing saved experimental outputs.' if local else 'Experimental outputs are not yet available for this primary forecast.')


def render_variations(primary,selected):
    st.header('Experimental variations')
    st.markdown('Explore how different assumptions change the forecast. These research scenarios are **not promoted replacements** for the primary XGBoost model.')
    bundle,warning=load_bundle(primary)
    if warning:st.caption(warning)
    if bundle is None:return
    baseline_names={r['team']:r for r in primary['odds']}
    tabs=st.tabs(list(TITLES.values()))
    for tab,(identifier,title) in zip(tabs,TITLES.items()):
        with tab:
            st.markdown(DESCRIPTIONS[identifier])
            model=bundle['models'][identifier]
            if model['status']!='available':
                st.info('This variation is not available for the current forecast cutoff.')
                continue
            stamp=datetime.fromisoformat(model['computed_at']).astimezone(timezone.utc)
            st.caption(f'Games through {model["games_through"]} · Computed {stamp:%Y-%m-%d %H:%M UTC} · {model["simulations"]:,} paths')
            if (datetime.now(timezone.utc)-stamp).total_seconds()>48*3600:
                st.warning('This experimental run is more than 48 hours old; it has not been refreshed.')
            if identifier=='uncertainty':
                st.caption('Uses a separate team-strength estimator and game model; differences from XGBoost are not attributable to uncertainty alone.')
            elif identifier=='injury':
                st.caption('Estimated return dates are uncertain. Existing absences may already affect team statistics, so penalties can double-count an injury.')
            else:
                settings=model['settings']
                st.caption(f'Published assumptions: severity {settings["severity"]:g}×; recovery decay {settings["recovery_days"]:g} days. Compared with the uncertainty baseline; these settings are fixed for this run.')
                if settings.get('team_specific_adjustments'):st.caption('This run also includes team-specific stress assumptions.')
            rows=pd.DataFrame(model['odds']).set_index('team')
            rows['Primary XGBoost (%)']=[100*baseline_names[t]['title_probability'] for t in rows.index]
            rows['Experimental (%)']=100*rows.title_probability
            rows['Baseline (%)']=100*rows.baseline_title_probability
            rows['Change vs baseline (pp)']=rows['Experimental (%)']-rows['Baseline (%)']
            baseline_label='Uncertainty baseline' if identifier=='stress' else 'Primary XGBoost'
            chart=rows.loc[rows['name'].isin(selected),['name','Experimental (%)','Baseline (%)']].reset_index(drop=True)
            if len(chart):
                chart=chart.rename(columns={'Experimental (%)':title,'Baseline (%)':baseline_label}).melt(id_vars='name',var_name='Forecast',value_name='Championship probability (%)')
                fig=px.bar(chart,x='name',y='Championship probability (%)',color='Forecast',barmode='group',
                           labels={'name':'Team'},color_discrete_sequence=['#159b86','#a4afc1'])
                fig.update_layout(height=390,legend_title_text='',margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig,width='stretch')
            columns=['name','Primary XGBoost (%)','Experimental (%)','Change vs baseline (pp)']
            if identifier=='stress':columns.insert(2,'Baseline (%)')
            display=rows[columns].sort_values('Experimental (%)',ascending=False).rename(columns={'name':'Team','Baseline (%)':'Uncertainty baseline (%)'})
            config={c:st.column_config.NumberColumn(format='%.2f') for c in display.columns if c!='Team'}
            st.dataframe(display,hide_index=True,width='stretch',column_config=config)
            st.download_button('Download this variation',display.to_csv(index=False),
                               f'{identifier}-{model["games_through"]}.csv','text/csv',key='download_'+identifier)
