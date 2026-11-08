"""Offline Streamlit demonstration using invented teams and an illustrative rule."""
import streamlit as st
import plotly.express as px
from synthetic_demo import synthetic_forecast

st.set_page_config(page_title='NBA forecasting methods — synthetic demo',layout='wide')
st.title('Forecast simulation and stress scenarios')
st.warning('Synthetic demonstration: all teams and statistics are invented. These are not NBA forecasts, trained-model results, or validated injury effects.')
st.caption('The actual primary simulation engine runs with an illustrative logistic matchup rule. Invented 41–41 records leave play-in and playoff paths to simulate.')
penalty=st.slider('Synthetic Team 01 strength penalty (standardized units)',0.,3.,0.,.25)

@st.cache_data
def run(penalty):
    return synthetic_forecast(simulations=5000,strength_penalty=penalty)

odds=run(penalty)
st.caption('5,000 simulated paths with fixed random seeds. Sampling error and model-assumption uncertainty are different quantities.')
chart=odds.assign(title_percent=100*odds.title_probability).sort_values('title_percent',ascending=False)
st.plotly_chart(px.bar(chart,x='name',y='title_percent',labels={'name':'Invented team','title_percent':'Simulated championship share (%)'}),width='stretch')
st.dataframe(chart[['name','title_percent','playoff_probability','simulation_se']],hide_index=True,width='stretch')
st.markdown('For real-data acquisition and training, see `docs/SETUP.md`. This demonstration needs no network, dataset download, or trained artifact.')
