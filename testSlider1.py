import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objects as go
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

# Define graphs
numnodes = 30
numedge = 3

# Physical network structure
G1 = nx.barabasi_albert_graph(numnodes, numedge)
G2 = nx.barabasi_albert_graph(numnodes, numedge)

# Diffusion model: SIR model
model1 = ep.SIRModel(G1)
model2 = ep.SIRModel(G2)

config1 = mc.Configuration()
config2 = mc.Configuration()

config1.add_model_parameter('beta', 0.1)  # infection rate
config1.add_model_parameter('gamma', 0)  # recovery rate
config2.add_model_parameter('beta', 0.1)  # infection rate
config2.add_model_parameter('gamma', 0)  # recovery rate

# Initialize infection node
initial_infected_node = 5
for node in G1.nodes():
    if node == initial_infected_node:
        config1.add_node_configuration("status", node, 1)  # infected
    else:
        config1.add_node_configuration("status", node, 0)  # susceptible

for node in G2.nodes():
    if node == initial_infected_node:
        config2.add_node_configuration("status", node, 1)  # infected
    else:
        config2.add_node_configuration("status", node, 0)  # susceptible

model1.set_initial_status(config1)
model2.set_initial_status(config2)

# Compute graph layouts
pos1 = nx.spring_layout(G1)
pos2 = nx.spring_layout(G2)

# Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Slider(
        id='iteration-slider',
        min=0,
        max=20,
        step=1,
        value=0,
        marks={i: str(i) for i in range(21)},
    ),
    dcc.Graph(id='network-graph')
])


# Initial computation of model iterations
iterations1 = [model1.status]
iterations2 = [model2.status]
for _ in range(20):
    model1.iteration()
    model2.iteration()
    iterations1.append(model1.status.copy())
    iterations2.append(model2.status.copy())

@app.callback(
    Output('network-graph', 'figure'),
    Input('iteration-slider', 'value')
)
def update_network(iteration):
    node_status1 = iterations1[iteration]
    colors1 = ['rgb(0,255,0)' if s == 0 else 'rgb(255,0,0)' if s == 1 else 'rgb(0,0,255)' for s in node_status1.values()]
    colors1[initial_infected_node] = 'rgb(0,0,0)'
    node_status2 = iterations2[iteration]
    colors2 = ['rgb(0,255,0)' if s == 0 else 'rgb(255,0,0)' if s == 1 else 'rgb(0,0,255)' for s in node_status2.values()]
    colors2[initial_infected_node] = 'rgb(0,0,0)'

    
    offset_y = 2

    node_trace1 = go.Scatter(
        x=[pos1[node][0] for node in G1.nodes()],
        y=[pos1[node][1] for node in G1.nodes()],
        mode="markers",
        marker=dict(
            size=20,
            color=colors1
        ),
        hoverinfo='text',
        text=list(G1.nodes()),
    )

    node_trace2 = go.Scatter(
        x=[pos2[node][0] for node in G2.nodes()],
        y=[pos2[node][1] + offset_y for node in G2.nodes()],
        mode="markers",
        marker=dict(
            size=20,
            color=colors2
        ),
        hoverinfo='text',
        text=list(G2.nodes()),
    )

    edge_trace1 = go.Scatter(
        x=[pos1[e[0]][0] if idx % 2 == 0 else pos1[e[1]][0] for idx, e in enumerate(G1.edges())],
        y=[pos1[e[0]][1] if idx % 2 == 0 else pos1[e[1]][1] for idx, e in enumerate(G1.edges())],
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    edge_trace2 = go.Scatter(
        x=[pos2[e[0]][0] if idx % 2 == 0 else pos2[e[1]][0] for idx, e in enumerate(G2.edges())],
        y=[pos2[e[0]][1] + offset_y if idx % 2 == 0 else pos2[e[1]][1] + offset_y for idx, e in enumerate(G2.edges())],
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    fig = go.Figure(data=[edge_trace1, node_trace1, edge_trace2, node_trace2],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ))

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
