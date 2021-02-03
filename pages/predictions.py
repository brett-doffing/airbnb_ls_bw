# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import sqlalchemy as db
from dash.dependencies import Input, Output, State
# Imports from this application
from app import app, db, Listing

#df = pd.read_csv('assets/predict.csv')
features = Listing.query.all()[0]

#df = pd.DataFrame(features)

# fig = go.Figure(data=[go.Table(
#     header=dict(values=list(map(lambda x:x.title(), df.columns)),
#                 fill_color='paleturquoise',
#                 align='left'),
#     cells=dict(values=list(map(lambda x:df[x], df)),
#                fill_color='lavender',
#                align='left'))
# ])

column1 = dbc.Col(
    [
    #    dcc.Graph(figure=fig),
        dcc.Markdown(str(features.id))
    ],
)

layout = dbc.Row([column1])