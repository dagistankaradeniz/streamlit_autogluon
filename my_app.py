import streamlit as st
from autogluon.tabular import TabularDataset, TabularPredictor

st.set_page_config(
    page_title="dk - AutoGluon AutoMl",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('## `AutoGluon` TabularPredictor')
with st.spinner('Preparing...'):
    data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
    train_data = TabularDataset(f'{data_url}train.csv')
    st.markdown('#### `train_data.head()`')
    st.write(train_data.head())

    label = 'signature'
    st.markdown('#### `label.describe`')
    st.write(train_data[label].describe())

st.markdown('#### Training / Fitting')
with st.spinner('Preparing...'):
    predictor = TabularPredictor(label=label).fit(train_data)

    test_data = TabularDataset(f'{data_url}test.csv')
    st.markdown('#### `test_data.head()`')
    st.write(test_data.head())

st.markdown('#### Predicting')
with st.spinner('Preparing...'):
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    st.markdown('#### `y_pred.head()`')
    st.write(y_pred.head())

    st.markdown('#### Evaluate test data')
    st.write(predictor.evaluate(test_data, silent=True))

    st.markdown('#### test data leaderboard')
    st.write(predictor.leaderboard(test_data, silent=True))
