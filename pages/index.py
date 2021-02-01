# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Imports from this application
from app import app

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            # Airbnb Predictions

            """
        ),
        html.Div(
            children=[dcc.Link(dbc.Button('Load Interactive Chart', color='primary'), href='/insights')],  # fill out your Input however you need
            style=dict(display='flex', justifyContent='center')
        )
        # dcc.Link(dbc.Button('Load Interactive Chart', color='primary'), href='/insights'),
    ],
)

layout = dbc.Row([column1])