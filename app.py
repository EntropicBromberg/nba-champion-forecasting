"""Two-page Streamlit app: current-season title odds and historical correlations."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.express as px
import streamlit as st
from data_pipeline import DATA, FEATURES, TARGET
from live import refresh_or_cached
from experimental_live import forecast as experimental_forecast
from injury_live import forecast as injury_forecast
from injury_uncertainty import forecast as stress_forecast,current_report

SEASON = f'{TARGET-1}–{str(TARGET)[-2:]}'
st.set_page_config(page_title=f'NBA {SEASON}', page_icon='🏀', layout='wide')

# This source release intentionally excludes third-party data and fitted models.
from pathlib import Path
if not all((Path(__file__).resolve().parent/path).exists() for path in
           ['artifacts/xgboost.json','data/teams.csv','data/matchups.csv']):
    st.title('NBA Champion Forecasting')
    st.info('Real-data forecasts require locally acquired data and a trained model. See docs/SETUP.md.')
    st.markdown('Try the offline synthetic demonstration: `python -m streamlit run demo.py`. All demo teams and statistics are invented.')
    st.stop()

@st.cache_data(ttl=1800,show_spinner=False)
def load_live():
    return refresh_or_cached()

@st.cache_data(ttl=1800,show_spinner=False)
def load_experimental(primary):
    return experimental_forecast(primary)

@st.cache_data(ttl=1800,show_spinner=False)
def load_injury(primary):
    return injury_forecast(primary)

@st.cache_data(ttl=1800,show_spinner=False)
def load_stress_report():
    return current_report()

@st.cache_data(ttl=1800,show_spinner=False)
def load_stress(primary,report,severity,recovery,team_multipliers):
    return stress_forecast(primary,report,severity,recovery,team_multipliers)

def clear_forecasts():
    load_live.clear()
    load_experimental.clear()
    load_injury.clear()
    load_stress_report.clear()
    load_stress.clear()

def model_comparison(primary,stress_settings):
    """Join by team identity, never row/rank; each forecast gets the same primary inputs."""
    teams=pd.DataFrame(primary['teams'])[['team','name']].set_index('team')
    forecasts={'Production (%)':primary}
    problems=[]
    for label,loader in [
        ('Uncertainty (%)',lambda:load_experimental(primary)),
        ('Injury-adjusted production (%)',lambda:load_injury(primary)),
        ('Uncertainty + injury stress (%)',lambda:load_stress(primary,load_stress_report(),*stress_settings))]:
        try:
            result=loader()
            if result['games_through']!=primary['games_through']:raise ValueError('Forecast dates do not match')
            forecasts[label]=result
        except Exception as error:
            forecasts[label]=None;problems.append(f'{label}: {error}')
    for label,result in forecasts.items():
        if result is None:
            teams[label]=float('nan');continue
        rows=pd.DataFrame(result['odds'])
        if rows.team.duplicated().any() or set(rows.team)!=set(teams.index):
            teams[label]=float('nan');problems.append(f'{label}: team identities do not match');continue
        teams[label]=100*rows.set_index('team').title_probability
    return teams.rename(columns={'name':'Team'}).sort_values('Production (%)',ascending=False).reset_index(drop=True),problems


@st.cache_data
def historical_data(version):
    return pd.read_csv(DATA/'matchups.csv')

@st.fragment(run_every='30m')
def probabilities():
    st.title(f'{SEASON} championship probabilities')
    st.button('Refresh now',on_click=clear_forecasts)
    try:
        with st.spinner('Refreshing season stats and championship probabilities…'):
            snapshot,warning=load_live()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
    if warning:
        st.warning(warning)
    view=st.radio('Forecast view',['Production (primary)','Experimental — uncertainty-aware','Experimental — injury-adjusted','Experimental — uncertainty + injury stress'],index=3 if st.query_params.get('forecast') in ['injury','injury-stress'] else 0,horizontal=True,key='forecast_view')
    experimental=view!='Production (primary)'
    if view=='Experimental — uncertainty + injury stress':
        st.warning('Manual injury stress scenario — unvalidated. The uncertainty model is unchanged; penalties and recovery assumptions are set below.')
        with st.expander('Set injury penalties',expanded=True):
            with st.form('injury_stress_settings'):
                left,right=st.columns(2)
                with left:
                    severity=st.slider('Injury severity multiplier',0.,4.,2.,.25,help='0 exactly reproduces the uncertainty-aware forecast. Default 2 is a harsh scenario assumption.')
                    recovery=st.slider('Long-absence recovery decay (days)',15,180,90,15,help='After returning from a long absence, impairment begins at 65% of the absence penalty, then decays. At 90 days it retains 37% of that return-day penalty.')
                with right:
                    codes=sorted(t['team'] for t in snapshot['teams'])
                    override_team=st.selectbox('Team-specific adjustment',codes,index=codes.index('OKC'))
                    team_multiplier=st.slider('Additional multiplier for selected team',0.,3.,1.,.25)
                st.form_submit_button('Apply injury assumptions')
            st.caption('Prior BPM and projected minutes → lost player value. Top-three roles receive 1.75×; weak depth and concentrated star production increase the penalty. Long absences still matter after the estimated return date. These controls change assumptions, not trained weights.')
        try:
            with st.spinner('Simulating uncertainty with persistent injury and recovery penalties…'):
                alternate=load_stress(snapshot,load_stress_report(),severity,recovery,{override_team:team_multiplier} if team_multiplier!=1 else {})
        except Exception as error:
            st.error(f'Injury stress forecast unavailable: {error}')
            return
        odds=pd.DataFrame(alternate['odds'])
        checked=datetime.fromisoformat(alternate['computed_at']).astimezone(ZoneInfo('America/Vancouver'))
        st.caption(f"Games through {alternate['games_through']} · Computed {checked:%b %d, %Y %H:%M %Z} · Injury feed {alternate['report']['source_timestamp']} · 10,000 paired paths")
        williams=next((r for r in alternate['adjustments'] if r['player']=='Jalen Williams' and r['applied']),None)
        if williams:
            st.info(f"OKC · Jalen Williams: role rank {williams['role_rank']} · Estimated return {williams['return_date']} · Recovery penalty at playoff start: {williams['playoff_player_penalty']:.2f} strength points. This is a scenario assumption, not a medical forecast.")
        with st.expander('Player penalties and method'):
            affected=pd.DataFrame(alternate['adjustments'])
            st.dataframe(affected[['team','player','status','top3','depth_bpm','top3_share','return_date','applied','playoff_player_penalty']],hide_index=True,width='stretch')
            st.caption(alternate['assumptions'])
    elif view=='Experimental — injury-adjusted':
        st.warning('Experimental injury scenario — unvalidated. Production remains the established forecast. Return dates are estimates, not confirmed recovery dates.')
        try:
            with st.spinner('Refreshing injury reports and simulating dated absences…'):
                alternate=load_injury(snapshot)
        except Exception as error:
            st.error(f'Injury forecast unavailable: {error}. No injury-adjusted odds are being substituted.')
            return
        odds=pd.DataFrame(alternate['odds'])
        checked=datetime.fromisoformat(alternate['computed_at']).astimezone(ZoneInfo('America/Vancouver'))
        st.caption(f"Games through {alternate['games_through']} · Injury scenario computed {checked:%b %d, %Y %H:%M %Z} · Injury feed {alternate['report']['source_timestamp']}")
        st.caption('50,000 paths · Absences affect remaining games until estimated return; playoff availability is evaluated separately.')
        for injury in alternate['adjustments']:
            if injury['applied'] and injury['strength_penalty']>=1 and injury.get('active_until','')>f'{TARGET-1}-11-15':
                st.info(f"{injury['team']} · {injury['player']}: {injury['status']} · Estimated return {injury['active_until']} · Applied to remaining games before that date.")
        with st.expander('Injury reports and assumptions',expanded=False):
            affected=pd.DataFrame(alternate['adjustments'])
            st.dataframe(affected[['team','player','status','estimated_return','applied','strength_penalty','reason']],hide_index=True,width='stretch')
            st.caption(alternate['assumptions'])
            st.markdown('[Injury report source](https://www.espn.com/nba/injuries)')
    elif experimental:
        st.warning('Experimental — not promoted. Title-odds calibration is not established; production remains the primary forecast.')
        try:
            with st.spinner('Building the experimental team-strength forecast…'):
                alternate=load_experimental(snapshot)
        except Exception as error:
            st.warning(f'Experimental forecast unavailable: {error}. Select Production (primary) for the established forecast.')
            return
        odds=pd.DataFrame(alternate['odds'])
        checked=datetime.fromisoformat(alternate['computed_at']).astimezone(ZoneInfo('America/Vancouver'))
        st.caption(f"Games through {alternate['games_through']} · Experimental forecast computed {checked:%b %d, %Y %H:%M %Z}")
        st.caption('Team data only · 10,000 persistent-strength simulation paths · Experimental game model trained through 2025–26. Inputs are aligned to the primary forecast date; this does not replace production.')
    else:
        odds=pd.DataFrame(snapshot['odds'])
        checked=datetime.fromisoformat(snapshot['checked_at']).astimezone(ZoneInfo('America/Vancouver'))
        st.caption(f"Games through {snapshot['games_through']} · Last successful refresh {checked:%b %d, %Y %H:%M %Z} · Trained on {snapshot['training_seasons']} seasons through 2025–26")
        st.caption('Current-season estimates from 50,000 season, play-in and playoff simulations. Early stats blend with last season; estimates update every 30 minutes while open and on launch. Published completed games only. Production does not consume live injury reports; select the injury-adjusted experimental view for that scenario.')
    st.subheader('All forecasts side by side')
    settings=(severity,recovery,{override_team:team_multiplier} if team_multiplier!=1 else {}) if view=='Experimental — uncertainty + injury stress' else (2.,90,{})
    with st.spinner('Comparing all forecasts on the same season inputs…'):
        comparison,problems=model_comparison(snapshot,settings)
    st.dataframe(comparison,hide_index=True,width='stretch',height=600,column_config={
        column:st.column_config.NumberColumn(format='%.2f%%') for column in comparison.columns if column!='Team'})
    st.caption('Championship probabilities for the same 30 teams and completed-game snapshot. Sorted by production. Other columns are experimental scenarios; they are not combined into an ensemble. Injury stress uses the controls above when selected, otherwise severity 2 and 90-day recovery decay.')
    for problem in problems:st.warning(f'Comparison unavailable — {problem}')
    st.download_button('Download model comparison',comparison.to_csv(index=False),'nba_model_comparison.csv','text/csv')
    st.subheader('Selected forecast')
    plot=odds.copy()
    plot['Probability (%)']=100*plot.title_probability
    ordered=plot.sort_values('Probability (%)')
    fig=px.bar(ordered,x='Probability (%)',y='name',orientation='h',color='conference',
        color_discrete_map={'E':'#4385f5','W':'#f29a38'},
        labels={'name':'','conference':'Conference'},text='Probability (%)')
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside',cliponaxis=False)
    fig.update_layout(height=850,margin=dict(l=0,r=50,t=10,b=0),
        xaxis_range=[0,max(plot['Probability (%)'].max()*1.15,1)],
        yaxis=dict(categoryorder='array',categoryarray=ordered.name.tolist()))
    st.plotly_chart(fig,width='stretch')
    display=odds[['name','wins','losses','projected_wins','playoff_probability','title_probability']].rename(columns={
        'name':'Team','wins':'W','losses':'L','projected_wins':'Projected wins',
        'playoff_probability':'Playoff probability (%)','title_probability':'Title probability (%)'})
    for col in ['Playoff probability (%)','Title probability (%)']:
        display[col]*=100
    if view=='Experimental — uncertainty + injury stress':
        display['Uncertainty baseline (%)']=100*odds.uncertainty_title_probability
        display['Injury change (pp)']=odds.injury_change_pp
        display['Playoff strength penalty']=odds.playoff_strength_penalty
    elif view=='Experimental — injury-adjusted':
        display['Production title (%)']=100*odds.production_title_probability
        display['Injury change (pp)']=odds.injury_change_pp
    st.dataframe(display,hide_index=True,width='stretch',column_config={
        'Projected wins':st.column_config.NumberColumn(format='%.1f'),
        'Playoff probability (%)':st.column_config.NumberColumn(format='%.1f%%'),
        'Title probability (%)':st.column_config.NumberColumn(format='%.2f%%')})
    st.caption('These are model estimates, not sportsbook odds. The live-season extension has not been backtested at early-season checkpoints. A displayed 0% can mean no simulated titles. Unassigned NBA Cup games use an average opponent; tied records use random tiebreaks.')
    st.download_button('Download probabilities',odds.to_csv(index=False),f'nba_{TARGET-1}_{str(TARGET)[-2:]}_title_odds'+('_experimental.csv' if experimental else '.csv'),'text/csv')

page=st.sidebar.radio(f'NBA · {SEASON}',['Championship probabilities','Feature correlations'])
if page=='Championship probabilities':
    probabilities()
else:
    st.title('Feature correlations')
    st.caption('Candidate regular-season feature differences, 2001–02 through 2025–26. Both matchup orderings are included. Correlation describes association, not causation.')
    matchups=historical_data((DATA/'matchups.csv').stat().st_mtime_ns)
    chosen=st.selectbox('Feature',['All features']+FEATURES)
    corr=matchups.loc[matchups.season.lt(TARGET),FEATURES+['y']].corr()
    if chosen!='All features':
        corr=corr[[chosen]].sort_values(chosen,ascending=False)
    fig=px.imshow(corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,text_auto='.2f',aspect='auto',labels={'color':'Pearson r'})
    fig.update_layout(height=650,margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig,width='stretch')
    st.caption('d_ columns are A − B after within-season z-scoring. home_court is 1/0 (0.5 in the 2020 bubble); round is 1–4; y indicates an A series win. Optional career playoff experience is omitted because complete career data is unavailable.')
