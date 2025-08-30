from dash import Dash, dcc, html
from tabs.tab0_home import home_tab_content
from tabs.tab1_data import data_tab_content
from tabs.tab2_model import model_tab_content
from tabs.tab3_train import train_tab_content
from tabs.tab4_post import post_tab_content

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Home', children=home_tab_content),
        dcc.Tab(label='Data', children=data_tab_content),
        dcc.Tab(label='Modelling', children=model_tab_content),
        dcc.Tab(label='Training', children=train_tab_content),
        dcc.Tab(label='Posterior', children=post_tab_content)
    ])
])

if __name__ == '__main__':
    # type this in terminal
    # conda activate poem
    # HOST=127.0.0.1 (ou 0.0.0.0)
    # python app.py
    # dash tourne alors sur l'adresse affichée
    app.run(debug=True)
