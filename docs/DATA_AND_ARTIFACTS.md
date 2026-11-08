# Data sources and forecast outputs

The historical pipeline uses [Basketball Reference](https://www.basketball-reference.com), [llimllib/nba_data](https://github.com/llimllib/nba_data), and [hoopR data](https://github.com/sportsdataverse/hoopR-data). Current forecasts also use ESPN public schedule feeds. The [source manifest](../data/sources.json) records the historical sources.

This repository includes code, specifications, authored research summaries, aggregate evaluation scores, and model-generated championship probabilities. It does not bundle source HTML, game logs, player tables, injury-feed text, recovery histories, or trained models. Those assets are acquired/generated locally under the applicable source terms.

The README forecast publishes only team identifiers/names, model probabilities, and forecast dates. These are derived model outputs, not a redistribution of the input datasets. The automated refresh uses temporary runner storage and publishes no source-data caches or fitted artifacts. Publishing these outputs does not grant rights in the underlying providers' content.

The scheduled workflow attempts a daily refresh. If acquisition, validation, or training fails, it leaves the last successful forecast and its original timestamp unchanged. A displayed forecast is only as recent as its timestamp; it is not a real-time feed. The interactive Streamlit application requires separate hosting and locally available inputs.

External libraries retain their licenses. Source availability and permissions can change; review the relevant provider terms before acquisition or redistribution. No source access controls are bypassed.
