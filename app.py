import dash
import dash_bootstrap_components as dbc
from flask_sqlalchemy import SQLAlchemy

"""
https://github.com/facultyai/dash-bootstrap-components

dash-bootstrap-components provides Bootstrap components.

Plotly Dash is great! However, creating the initial layout can require a lot 
of boilerplate. dash-bootstrap-components reduces this boilerplate by providing 
standard layouts and high-level components.

A good way to start customising the stylesheet is to use an alternative 
pre-compiled theme. Bootswatch is a great place to find new themes. Links to 
CDNs for each of the Bootswatch styles are also included , and can be used 
with the external_stylesheets argument of the Dash constructor:

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

Go to https://bootswatch.com/ to preview these Bootswatch themes:

dbc.themes.BOOTSTRAP
dbc.themes.CERULEAN
dbc.themes.COSMO
dbc.themes.CYBORG
dbc.themes.DARKLY
dbc.themes.FLATLY
dbc.themes.JOURNAL
dbc.themes.LITERA
dbc.themes.LUMEN
dbc.themes.LUX
dbc.themes.MATERIA
dbc.themes.MINTY
dbc.themes.PULSE
dbc.themes.SANDSTONE
dbc.themes.SIMPLEX
dbc.themes.SKETCHY
dbc.themes.SLATE
dbc.themes.SOLAR
dbc.themes.SPACELAB
dbc.themes.SUPERHERO
dbc.themes.UNITED
dbc.themes.YETI
"""

external_stylesheets = [
    dbc.themes.UNITED,  # Bootswatch theme
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css',  # for social media icons
]

meta_tags = [
    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}
]

app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets, meta_tags=meta_tags)
app.config.suppress_callback_exceptions = True  # see https://dash.plot.ly/urls
app.title = 'AirBnB Predictions'  # appears in browser title bar
server = app.server
server.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3?check_same_thread=False"
server.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(server)


class Listing(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    occupancy = db.Column(db.Integer())
    cleaning_fee = db.Column(db.Float())
    listing_count = db.Column(db.Integer())
    availability_90 = db.Column(db.Integer())
    extra_person_fee = db.Column(db.Float())
    num_reviews = db.Column(db.Integer())
    num_bath = db.Column(db.Float())
    security_deposit = db.Column(db.Float())
    min_nights = db.Column(db.Integer())
    room = db.Column(db.Integer())

db.create_all()
