# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
# Imports from this application
from app import app


def get_prediction(occupancy, listing_count,
                   availability_90, num_reviews,
                   num_bath, num_bed, num_bedroom,
                   min_nights, room_type, property_type, bathroom_type,
                   has_about, has_neighborhood_overview,
                   account_age, has_profile_pic, identity_verified,
                   is_superhost, instant_bookable,
                   neighbourhood):
    """
    Function for checking if all features are correctly entered and to return
    a message to the user to fix their input if not. Otherwise returns a price
    prediction by passing features to the price prediction model.
    """
    features = [occupancy, listing_count,
                availability_90, num_reviews,
                num_bath, num_bed, num_bedroom,
                min_nights, room_type, property_type, bathroom_type,
                has_about, has_neighborhood_overview,
                account_age, has_profile_pic, identity_verified,
                is_superhost, instant_bookable,
                neighbourhood]
    if None in features:
        return dcc.Markdown('All fields must be filled!')
    else:
        test = sum(features)
        prediction = 'TEST: {}'.format(test)
        return dcc.Markdown('You should charge {}'.format(prediction))


def get_options(options):
    """
    Get options for dcc.Dropdown from a list of options
    """
    opts = []
    for opt in options:
        opts.append({'label': opt.title(), 'value': opt})
    return opts


# Column for displaying the app callback result after clicking
# on the prediction button
prediction_column = dbc.Col(
    html.Center(id='output-submit')
)

# Column for first 5 features
column1 = dbc.Col(
    [
        dcc.Input(id='occupancy', type='number',
                  placeholder='Max Occupancy', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='listing_count', type='number',
                  placeholder='Number of Listings', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='availability_90', type='number',
                  placeholder='Next 90 Days Available', min=1,
                  step=1),
        html.Br(), html.Br(),
        dcc.Input(id='min_nights', type='number',
                  placeholder='Minimum Nights', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='bathroom_type',
            options=get_options(['private', 'regular', 'shared']),
            searchable=False,
            clearable=False,
            placeholder='Bathroom Type'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='neighbourhood',
            options=get_options(['Yau Tsim Mong', 'Yuen Long', 'Wan Chai', 'Central & Western',
                                 'Eastern', 'Kowloon City', 'Sha Tin', 'Sham Shui Po', 'Islands',
                                 'Sai Kung', 'Wong Tai Sin', 'North', 'Tsuen Wan', 'Kwun Tong',
                                 'Southern', 'Tuen Mun', 'Kwai Tsing', 'Tai Po']),
            searchable=False,
            clearable=False,
            placeholder='Neighbourhood'
        ),
        html.Br(), html.Br(),
        dcc.RadioItems(
            id='identity_verified',
            options=[
                {'label': 'Identity Verified', 'value': 1},
                {'label': 'Identity Not Verified', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block', 'padding': '10px'}
        ),
        html.Br(), html.Br(),
        dcc.RadioItems(
            id='is_superhost',
            options=[
                {'label': 'Superhost', 'value': 1},
                {'label': 'Not Superhost', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block', 'padding': '10px'}
        ),
        html.Br(), html.Br(),
        dcc.RadioItems(
            id='instant_bookable',
            options=[
                {'label': 'Instant Booking', 'value': 1},
                {'label': 'No Instant Booking', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block', 'padding': '10px'}
        ),
    ]
)

# Column for last 5 features
column2 = dbc.Col(
    [
        dcc.Input(id='num_reviews', type='number',
                  placeholder='Number of Reviews', step=1),
        html.Br(), html.Br(),
        dcc.Input(id='num_bath', type='number',
                  placeholder='Number of Bathrooms', min=1, step=0.5),
        html.Br(), html.Br(),
        dcc.Input(id='num_bed', type='number',
                  placeholder='Number of Beds', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='num_bedroom', type='number',
                  placeholder='Number of Bedrooms', min=1, step=0.5),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='room_type',
            options=get_options(
                ['Entire House/apt', 'Private room', 'Shared room', 'Hotel room']),
            searchable=False,
            clearable=False,
            placeholder='Room Type'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='property_type',
            options=get_options(['apartment', 'guesthouse', 'bed and breakfast', 'bungalow',
                                 'hotel', 'guest suite', 'condominium', 'boutique hotel', 'house',
                                 'hostel', 'serviced apartment', 'loft', 'townhouse', 'Entire loft',
                                 'Entire apartment', 'Entire condominium',
                                 'Entire serviced apartment', 'treehouse', 'chalet', 'Entire house',
                                 'Entire guest suite', 'Private room', 'aparthotel', 'cottage',
                                 'castle', 'cabin', 'tiny house', 'resort', 'minsu', 'hut',
                                 'kezhan', 'villa', 'earth house', 'nature lodge', 'Tiny house',
                                 'Entire home/apt', 'igloo', 'Entire guesthouse', 'Casa particular',
                                 'Entire townhouse', 'Entire place', 'Entire bungalow',
                                 'Earth house', 'casa particular', 'Entire cottage', 'pension',
                                 'Campsite', 'Island', 'Boat', 'Cave', 'Tent', 'Castle', 'Pension',
                                 'Dome house', 'Entire villa', 'Farm stay', 'Shared room']),
            searchable=False,
            clearable=False,
            placeholder='Property Type'
        ),
        html.Br(), html.Br(),
        dcc.RadioItems(
            id='has_about',
            options=[
                {'label': 'Has About', 'value': 1},
                {'label': 'No About', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block', 'padding': '10px'}
        ),
        html.Br(), html.Br(),
        dcc.RadioItems(
            id='has_neighborhood_overview',
            options=[
                {'label': 'Has Neighborhood Overview', 'value': 1},
                {'label': 'No Neighborhood Overview', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block', 'padding': '10px'}
        ),
        html.Br(), html.Br(),
        dcc.RadioItems(
            id='has_profile_pic',
            options=[
                {'label': 'Has Profile Pic', 'value': 1},
                {'label': 'No Profile Pic', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block', 'padding': '10px'}
        ),
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
        ),)
    ]
)

# Column for price prediction button
column_button = dbc.Col(
    [
        html.Br(),
        html.Hr(),
        html.Center((dbc.Button('Make Price Prediction', color='primary',
                                id='btn-submit', n_clicks=0)))
    ]
)

# Webpage layout
layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([prediction_column]),
        dbc.Row([column1, column2]),
        dbc.Row([column_slider]),
        dbc.Row([column_button])
    ],
    style={'margin': 'auto'}
)


# App callback to get values from user and return a prediction
@app.callback(
    Output('output-submit', 'children'),
    Input('btn-submit', 'n_clicks'),
    State('occupancy', 'value'),
    State('listing_count', 'value'),
    State('availability_90', 'value'),
    State('num_reviews', 'value'),
    State('num_bath', 'value'),
    State('num_bed', 'value'),
    State('num_bedroom', 'value'),
    State('min_nights', 'value'),
    State('room_type', 'value'),
    State('property_type', 'value'),
    State('bathroom_type', 'value'),
    State('has_about', 'value'),
    State('has_neighborhood_overview', 'value'),
    State('account_age', 'value'),
    State('has_profile_pic', 'value'),
    State('identity_verified', 'value'),
    State('is_superhost', 'value'),
    State('instant_bookable', 'value'),
    State('neighbourhood', 'value'),
)
def update_output(clicks, occupancy, listing_count,
                  availability_90, num_reviews,
                  num_bath, num_bed, num_bedroom,
                  min_nights, room_type, property_type, bathroom_type,
                  has_about, has_neighborhood_overview,
                  account_age, has_profile_pic, identity_verified,
                  is_superhost, instant_bookable,
                  neighbourhood):
    if clicks:
        # Return prediction when button is clicked
        return get_prediction(occupancy, listing_count,
                              availability_90, num_reviews,
                              num_bath, num_bed, num_bedroom,
                              min_nights, room_type, property_type,
                              bathroom_type, has_about, has_neighborhood_overview,
                              account_age, has_profile_pic, identity_verified,
                              is_superhost, instant_bookable,
                              neighbourhood)
