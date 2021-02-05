# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
# Imports from this application
from app import app, pipeline
import keras
import numpy as np

model = keras.models.load_model('data/kerasmodel')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Model parameters
# columns = ['host_total_listings_count', 
# 'latitude', 
# 'longitude', 
# 'accommodates',
# 'bedrooms', 
# 'beds', 
# 'minimum_nights', 
# 'maximum_nights', 
# 'availability_30', 
# 'availability_365',
# 'number_of_reviews', 
# 'bathrooms']

def get_prediction(host_listings, latitude,
                  longitude, accommodates,
                  bedrooms, beds, num_reviews,
                  avail_30, avail_365, max_nights, min_nights,
                  num_bath):
    """
    Function for checking if all features are correctly entered and to return
    a message to the user to fix their input if not. Otherwise returns a price
    prediction by passing features to the price prediction model.
    """
    df = pd.DataFrame(data={'host_listings': host_listings,
                            'latitude': latitude,
                            'longitude': longitude,
                            'accommodates': accommodates,
                            'bedrooms': bedrooms,
                            'beds': beds,
                            'num_reviews': num_reviews,
                            'avail_30': avail_30,
                            'avail_365': avail_365,
                            'max_nights': max_nights,
                            'min_nights': min_nights,
                            'num_bath': num_bath,
                            },
                      index=[0]
                      )
    if df.isnull().values.any():
        return dcc.Markdown('All fields must be filled!')
    else:
        X = np.asarray(df).astype(np.float64)
        prediction = model.predict(X)
        # prediction = pipeline.predict(df)
        return dcc.Markdown('You should charge ${} per night.'.format(round(float(prediction[0])), 2))


def get_options(options):
    """
    Get options for dcc.Dropdown from a list of options
    """
    opts = []
    for opt in options:
        opts.append({'label': opt.title(), 'value': opt})
    return opts


# Column for first 5 features
column1 = dbc.Col(
    [
        dcc.Input(id='host_listings', type='number',
                  placeholder='Host Listings', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='latitude', type='number',
                  placeholder='Latitude', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='longitude', type='number',
                  placeholder='Longitude', min=1,
                  step=1),
        html.Br(), html.Br(),
        dcc.Input(id='accommodates', type='number',
                  placeholder='Accommodates', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='bedrooms', type='number',
                  placeholder='Bedrooms', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='beds', type='number',
                  placeholder='Beds', min=1, step=1),
        html.Br(), html.Br(),
    ]
)

# Column for last 5 features
column2 = dbc.Col(
    [
        dcc.Input(id='num_reviews', type='number',
                  placeholder='Number of Reviews', step=1),
        html.Br(), html.Br(),
        dcc.Input(id='avail_30', type='number',
                  placeholder='Availability 30', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='avail_365', type='number',
                  placeholder='Availability 365', min=1, step=0.5),
        html.Br(), html.Br(),
        dcc.Input(id='min_nights', type='number',
                  placeholder='Min nights', min=1, step=0.5),
        html.Br(), html.Br(),
        dcc.Input(id='max_nights', type='number',
                  placeholder='Max nights', min=1, step=0.5),
        html.Br(), html.Br(),
        dcc.Input(id='num_bath', type='number',
                  placeholder='Number of Bathrooms', min=1, step=0.5),
        html.Br(), html.Br(),
    ]
)

# Column for age slider
column_slider = dbc.Col(
    [
        html.Br(),
        html.Br(),
        html.Center(dcc.Markdown('Account Age')),
        html.Center(dcc.Slider(
            id='account_age',
            min=0,
            max=14,
            step=1,
            marks={
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5',
                6: '6',
                7: '7',
                8: '8',
                9: '9',
                10: '10',
                11: '11',
                12: '12',
                13: '13',
                14: '14',
            },
            value=7
        ),),
        html.Br(),
        html.Hr(),
    ]
)

# Column for displaying the app callback result after clicking
# on the prediction button
prediction_column = dbc.Col(
    html.Center(id='output-submit')
)

# Column for price prediction button
column_button = dbc.Col(
    [
        html.Hr(),
        html.Center((dbc.Button('Make Price Prediction', color='primary',
                                id='btn-submit', n_clicks=0)))
    ]
)

# Webpage layout
layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([column1, column2]),
        # dbc.Row([column_slider]),
        dbc.Row([prediction_column]),
        dbc.Row([column_button])
    ],
    style={'margin': 'auto'}
)


# App callback to get values from user and return a prediction
@app.callback(
    Output('output-submit', 'children'),
    Input('btn-submit', 'n_clicks'),
    State('host_listings', 'value'),
    State('latitude', 'value'),
    State('longitude', 'value'),
    State('accommodates', 'value'),
    State('bedrooms', 'value'),
    State('beds', 'value'),
    State('num_reviews', 'value'),
    State('avail_30', 'value'),
    State('avail_365', 'value'),
    State('max_nights', 'value'),
    State('min_nights', 'value'),
    State('num_bath', 'value'),
)
def update_output(clicks, host_listings, latitude,
                  longitude, accommodates,
                  bedrooms, beds, num_reviews,
                  avail_30, avail_365, max_nights, min_nights,
                  num_bath):
    if clicks:
        # Return prediction when button is clicked
        return get_prediction(host_listings, latitude,
                  longitude, accommodates,
                  bedrooms, beds, num_reviews,
                  avail_30, avail_365, max_nights, min_nights,
                  num_bath)
