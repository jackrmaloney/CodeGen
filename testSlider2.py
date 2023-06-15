import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(children=[
    html.H1(children='Epidemic Simulation'),
    dcc.Graph(id='graph')
])

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    dash.dependencies.Input('graph', 'id')  # This is a dummy input, replace this with actual inputs if needed
)
def update_graph(input_data):
    numnodes = 30
    numedge = 3

    # Physical network structure
    G1 = nx.barabasi_albert_graph(numnodes, numedge)

    # Diffusion model: SIR model
    model = ep.SIRModel(G1)

    config = mc.Configuration()
    config.add_model_parameter('beta', 0.1)  # infection rate
    config.add_model_parameter('gamma', 0)  # recovery rate

    # Initialize infection node
    initial_infected_node = 5
    for node in G1.nodes():
        if node == initial_infected_node:
            config.add_node_configuration("status", node, 1)  # infected
        else:
            config.add_node_configuration("status", node, 0)  # susceptible

    model.set_initial_status(config)

    iterations = 20  # Or however many iterations you want

    susceptible_counts = []
    infected_counts = []
    recovered_counts = []
    for i in range(iterations):
        iteration = model.iteration()

        statuses = iteration['status']
        susceptible_counts.append(list(statuses.values()).count(0))
        infected_counts.append(list(statuses.values()).count(1))
        recovered_counts.append(list(statuses.values()).count(2))

    time_steps = np.arange(iterations)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time_steps, y=susceptible_counts, mode='lines+markers', name='Susceptible'))
    figure.add_trace(go.Scatter(x=time_steps, y=infected_counts, mode='lines+markers', name='Infected'))
    figure.add_trace(go.Scatter(x=time_steps, y=recovered_counts, mode='lines+markers', name='Recovered'))

    figure.update_layout(
        title='SIR Model Simulation',
        xaxis_title='Iteration',
        yaxis_title='Count',
        xaxis=dict(tickmode='linear'),
    )

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
