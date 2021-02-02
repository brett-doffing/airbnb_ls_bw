# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
# Imports from this application
from app import app

column1 = dbc.Col(
    [
        dcc.Input(id='occupancy', type='number',
                  placeholder='Occupancy', min=1, max=10, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='cleaning_fee', type='number',
                  placeholder='Cleaning Fee'),
        html.Br(), html.Br(),
        dcc.Input(id='listing_count', type='number',
                  placeholder='Number of Listings', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='availability_90', type='number',
                  placeholder='Days Available Next 90 Days', min=1, max=90,
                  step=1),
        html.Br(), html.Br(),
        dcc.Input(id='extra_person_fee', type='number',
                  placeholder='Extra Person Fee'),
        html.Br(), html.Br(),
        dcc.Input(id='num_reviews', type='number',
                  placeholder='Number of Reviews', step=1),
        html.Br(), html.Br(),
        dcc.Input(id='num_bath', type='number',
                  placeholder='Number of Bathrooms', min=1, step=0.5),
        html.Br(), html.Br(),
        dcc.Input(id='security_deposit', type='number',
                  placeholder='Security Deposit'),
        html.Br(), html.Br(),
        dcc.Input(id='min_nights', type='number',
                  placeholder='Minimum Number of Nights', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='rooms',
            options=[
                {'label': 'Entire House/apt', 'value': 1},
                {'label': 'Private room', 'value': 2},
                {'label': 'Shared room', 'value': 3}
            ],
            value=1
        ),
        html.Br(), html.Br(),
        dcc.Link(dbc.Button('Make Price Prediction', color='primary',
                            id='btn-submit', n_clicks=0), href='/predictions'),
        # html.Button('Submit', id='btn-submit', n_clicks=0),
        html.Br(),
        html.Hr(),
        html.Div(id='output-submit'),
    ]
)

layout = dbc.Row([column1])


@app.callback(
    Output('output-submit', 'children'),
    Input('btn-submit', 'n_clicks'),
    State('occupancy', 'value'),
    State('cleaning_fee', 'value'),
    State('listing_count', 'value'),
    State('availability_90', 'value'),
    State('extra_person_fee', 'value'),
    State('num_reviews', 'value'),
    State('num_bath', 'value'),
    State('security_deposit', 'value'),
    State('min_nights', 'value'),
    State('rooms', 'value')
)
def update_output(clicks, occupancy, cleaning_fee, listing_count,
                  availability_90, extra_person_fee, num_reviews,
                  num_bath, security_deposit, min_nights, rooms):
    if clicks:
        df = pd.DataFrame(data={'occupancy': occupancy,
                                'cleaning_fee': cleaning_fee,
                                'listing_count': listing_count,
                                'availability_90': availability_90,
                                'extra_person_fee': extra_person_fee,
                                'num_reviews': num_reviews,
                                'num_bath': num_bath,
                                'security_deposit': security_deposit,
                                'min_nights': min_nights,
                                'rooms': rooms},
                          index=[0])
        df.to_csv('assets/predict.csv', index=False)

        # return '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(occupancy, cleaning_fee, listing_count, availability_90, extra_person_fee, num_reviews, num_bath, security_deposit, min_nights, rooms)
