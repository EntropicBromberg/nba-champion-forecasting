"""Public Streamlit view of the real primary forecast published by the pipeline."""
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from scripts.publish_forecast import public_forecast
from experimental_dashboard import render_variations,load_bundle

ROOT=Path(__file__).resolve().parent
FORECAST_URL='https://raw.githubusercontent.com/EntropicBromberg/nba-champion-forecasting/main/reports/forecast.json'
REPOSITORY='https://github.com/EntropicBromberg/nba-champion-forecasting'


def load_forecast():
    """Fetch validated outputs; keep the last committed real forecast on failure."""
    local=public_forecast(json.loads((ROOT/'reports/forecast.json').read_text()))
    try:
        response=requests.get(FORECAST_URL,timeout=10)
        response.raise_for_status()
        latest=public_forecast(response.json())
        if datetime.fromisoformat(latest['checked_at'])<datetime.fromisoformat(local['checked_at']):
            return local,'The remote forecast is older than the bundled forecast; showing the newer saved result.'
        return latest,None
    except (requests.RequestException,ValueError,KeyError,TypeError) as error:
        return local,f'Forecast refresh unavailable ({type(error).__name__}); showing the last committed primary forecast.'


@st.cache_data(ttl=1800,show_spinner=False)
def cached_forecast():
    return load_forecast()


def refresh_views():
    cached_forecast.clear()
    load_bundle.clear()


def render_dashboard():
    st.title('NBA championship probabilities')
    st.caption('Primary model · XGBoost · 2026–27 season')
    st.button('Refresh forecast',on_click=refresh_views)
    with st.spinner('Loading the latest published forecast…'):
        data,warning=cached_forecast()
    if warning:st.warning(warning)
    checked=datetime.fromisoformat(data['checked_at']).astimezone(timezone.utc)
    st.caption(f'Games through {data["games_through"]} · Model computed {checked:%Y-%m-%d %H:%M UTC} · {data["simulations"]:,} simulated season paths')
    age=(datetime.now(timezone.utc)-checked).total_seconds()/3600
    if age>48:st.warning('This forecast is more than 48 hours old. The timestamp reflects the last successful model run; a new forecast has not been confirmed.')
    frame=pd.DataFrame(data['odds'])
    frame['Championship probability (%)']=100*frame.title_probability
    leader=frame.iloc[0]
    left,middle,right=st.columns(3)
    left.metric('Leading contender',leader['name'])
    middle.metric('Leading title probability',f'{100*leader.title_probability:.2f}%')
    right.metric('Teams modeled',len(frame))
    selected=st.multiselect('Teams to compare',frame['name'].tolist(),default=frame['name'].head(10).tolist())
    chart=frame.loc[frame['name'].isin(selected)].sort_values('Championship probability (%)')
    if len(chart):
        fig=px.bar(chart,x='Championship probability (%)',y='name',orientation='h',
                   labels={'name':'Team'},text_auto='.2f',color_discrete_sequence=['#26b69e'])
        fig.update_layout(height=max(350,32*len(chart)),margin=dict(l=0,r=30,t=10,b=0),showlegend=False)
        st.plotly_chart(fig,width='stretch')
    else:st.info('Select at least one team to compare probabilities.')
    st.subheader('All 30 teams')
    display=frame[['name','Championship probability (%)']].rename(columns={'name':'Team'})
    st.dataframe(display,hide_index=True,width='stretch',
                 column_config={'Championship probability (%)':st.column_config.NumberColumn(format='%.2f%%')})
    st.download_button('Download probabilities',display.to_csv(index=False),
                       f'nba-title-probabilities-{data["games_through"]}.csv','text/csv')
    with st.expander('Model and interpretation'):
        st.markdown('A regularized XGBoost series classifier feeds season, play-in, and playoff simulations. The model uses 25 completed seasons; current-season statistics update forecast inputs. These are the same real primary probabilities published on GitHub, not synthetic demonstration values.')
        st.markdown('The forecasting pipeline is scheduled daily. This app checks for newly published outputs every 30 minutes; clicking refresh does not retrain the model. Failed source or training runs preserve the previous forecast and timestamp.')
        st.markdown('Historical series accuracy does not establish calibrated in-season championship odds. Future strength is held fixed within a path, tied records are simplified, and displayed zeros can reflect simulation sampling. Injury and uncertainty scenarios remain separate research experiments.')
        st.markdown(f'[Methodology]({REPOSITORY}/blob/main/docs/METHODOLOGY.md) · [Source code]({REPOSITORY})')

    render_variations(data,selected)
