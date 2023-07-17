# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import pandas as pd
import dash
import plotly.graph_objs as go
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
from utils import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
pd.options.display.float_format = '{:.4f}'.format
app = Dash(__name__, external_stylesheets=[external_stylesheets, dbc.themes.BOOTSTRAP])

# get the necessary data
# points.csv is a csv file containing braking, steering and accelerating points for every corner of the track
# this file also contains a 'range' of the lap position for every corner
dfPoints = pd.read_csv('Files\points.csv', index_col=False)
dfCar = pd.read_csv('Files\Otte sessie\car.csv')
dfLap = pd.read_csv('Files\Otte sessie\lap.csv')
dfInput = pd.read_csv('Files\Otte sessie\input.csv')
dfSession = pd.read_csv('Files\Otte sessie\session.csv')
dfgoodCar = pd.read_csv('Files\Goeie sessie\car.csv')

dfGas = dfInput['gas']
dfBrake = dfInput['brake']
dfSteer = dfInput['steer']
dfWorldCoordinates = dfCar['world location']
dfgoodWorldCoordinates = dfgoodCar['world location']
dfGoodCoordinates = pd.DataFrame(remove_y_coordinate(dfWorldCoordinates))
dfGoodCoordinates.columns = ['x', 'y']
dfLapPos = dfLap['lap position']
dfLapCount = dfLap['lap count']
dfCarTimestamp = dfCar['timestamp']
last_index = len(dfCarTimestamp)

# create one big dataframe that is synced by timestamp
dfAll = pd.concat([dfCarTimestamp, dfGoodCoordinates, dfGas, dfBrake, dfSteer, dfLapPos, dfLapCount], axis=1)

# get the 'chunks' of the lap position on which we split the corners
chunks = create_chunks(dfPoints)

# dataframe[i][j] gives all the data from dfAll on i=lap and j=corner
# for Zandvoort track 0 < j < 12 since there are 13 corners
dataFrame = split_data(dfAll, chunks)

# points = [lap2, lap3, lap4, ...]
# where lap1 = [[brakepoint corner 1, steerpoint corner 1, gaspoint corner 1], [brakepoint corner 2, etc.]]
# points[i][j][k] where i = lap, j = corner, k = input
points = get_all_points(dataFrame)

# perfect_points[i][j] where i=type of input (brake, steer, acc) and j=corner
# so perfect_points[0][0] = coordinates of perfect brake point for corner 1
perfect_points = all_to_coordinates(dfPoints)

# distances[i][j][k] where i = lap, j = corner, k = input
# same for average points
distances, points_x, points_y = calc_all_distances_and_averages(points, perfect_points)

# averages[i][j] where i = input and j = corner
average_distances, average_x, average_y = get_averages(distances, points_x, points_y)

full_throttle = check_full_throttle(dataFrame)

largest_values, largest_indices, input_type, input_type_int = get_worst(average_distances)

outputs = []
good_outputs = []
count = 0
for i in range(len(largest_indices)):
    if largest_values[i] == 0:
        count += 1
    elif largest_values[i] != 0:
        line = f"Turn{largest_indices[i]}{input_type[i]}{'Later' if largest_values[0] < 0 else 'Earlier'}"
        outputs.append(line)
        line2 = f"Turn{largest_indices[i]}{input_type[i]}Good"
        good_outputs.append(line2)

# remove the faulty zero entries from the input types to make sure we get the correct input types later on
input_type = input_type[:-count]
input_type_int = input_type_int[:-count]

turns = []
for i in range(len(outputs)):
    numbers = ''.join([char for char in outputs[i] if char.isdigit()])
    turns.append(numbers)

trackname = str(dfSession['track'].values[0])
formatted_trackname = trackname.replace("_", " ")
carname = str(dfSession['car'].values[0])
formatted_carname = carname.replace("_", " ")

fastest_lap_time = dfLap['best lap'].values[last_index - 1]
worst_lap_time = format_laptime(max(list(set(dfLap['last lap'].values))))

formatted_best_laptime = format_laptime(fastest_lap_time)
average_laptime = format_laptime(get_average_laptime(dfLap))
corrected_lapcount = dfLapCount.values[-1] - 2

########################################################################################################################
# initialization of the graphs to be shown
########################################################################################################################

# perfect_cooridnates_x/y[i][j] where i = input (brake steer acc) and j = corner
perfect_coordinates_x, perfect_coordinates_y = find_perfect_coordinates(perfect_points)

sortx, sorty = get_map_coordinates(dfgoodWorldCoordinates)

# ranges of coordinates to show a certain corner
x_ranges = [
    [260, 120],
    [160, 110],
    [230, 125],
    [None, None],
    [None, None],
    [-370, -580],
    [-480, -560],
    [-290, -420],
    [-370, -480],
    [100, -90],
    [115, -30],
    [40, -50],
    [260, 70]
]

# reverse these ranges since we need to mirror the graph, so it is displayed properly
for i in range(len(x_ranges)):
    x_ranges[i] = x_ranges[i][::-1]

# ranges of coordinates to show a certain corner
y_ranges = [
    [230, 420],
    [60, 200],
    [-60, 100],
    [None, None],
    [None, None],
    [190, 320],
    [-80, 45],
    [-80, 25],
    [40, 200],
    [-110, 10],
    [-180, -50],
    [-420, -270],
    [-460, -280]
]


bad_graphs = []
nr = 0

for turn in turns[:4]:
    turn = int(turn) - 1
    filtered_x = []
    filtered_y = []
    for coord_x, coord_y in zip(sortx, sorty):
        if x_ranges[turn][0] <= coord_x <= x_ranges[turn][1] and y_ranges[turn][0] <= coord_y <= y_ranges[turn][1]:
            filtered_x.append(coord_x)
            filtered_y.append(coord_y)

    perf_x, perf_y = [perfect_coordinates_x[input_type_int[nr]][turn]], [
        perfect_coordinates_y[input_type_int[nr]][turn]]
    act_x, act_y = [average_x[input_type_int[nr]][turn]], [average_y[input_type_int[nr]][turn]]
    nr += 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_x, y=filtered_y, mode='markers', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=perf_x, y=perf_y, mode='markers', marker=dict(color='green', size=20)))
    fig.add_trace(go.Scatter(x=act_x, y=act_y, mode='markers', marker=dict(color='red', size=20)))
    fig.update_layout(xaxis=dict(visible=False, autorange='reversed'))
    fig.update_layout(yaxis=dict(tickmode='linear', dtick=100, visible=False))
    fig.update_layout(width=300, height=300, showlegend=False, margin=dict(l=0, r=0, b=0, t=0))

    # some tweaks for readability of the graphs for certain corners (to make the corner appear as a driver would take it)
    if turn == 1:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]
    elif turn == 2:
        fig.update_layout(xaxis=dict(tickangle=0), yaxis=dict(tickangle=90))
        for trace in fig.data:
            trace.x, trace.y = trace.y, trace.x
            trace.x = [-x for x in trace.x if x is not None]
    elif turn == 6:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]
    elif turn == 9:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]
    elif turn == 11:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]

    bad_graphs.append(fig)


nr = len(input_type_int) - 1
good_graphs = []
rev_turns = turns[::-1]
for turn in rev_turns[0:4]:
    turn = int(turn) - 1
    filtered_x = []
    filtered_y = []
    for coord_x, coord_y in zip(sortx, sorty):
        if x_ranges[turn][0] <= coord_x <= x_ranges[turn][1] and y_ranges[turn][0] <= coord_y <= y_ranges[turn][1]:
            filtered_x.append(coord_x)
            filtered_y.append(coord_y)

    perf_x, perf_y = [perfect_coordinates_x[input_type_int[nr]][turn]], [
        perfect_coordinates_y[input_type_int[nr]][turn]]
    act_x, act_y = [average_x[input_type_int[nr]][turn]], [average_y[input_type_int[nr]][turn]]
    nr -= 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_x, y=filtered_y, mode='markers', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=perf_x, y=perf_y, mode='markers', marker=dict(color='green', size=20)))
    fig.add_trace(go.Scatter(x=act_x, y=act_y, mode='markers', marker=dict(color='red', size=20)))
    fig.update_layout(xaxis=dict(visible=False, autorange='reversed'))
    fig.update_layout(yaxis=dict(tickmode='linear', dtick=100, visible=False))
    fig.update_layout(width=300, height=300, showlegend=False, margin=dict(l=0, r=0, b=0, t=0))

    # some tweaks for readability of the graphs for certain corners (to make the corner appear as a driver would take it)
    if turn == 1:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]
    elif turn == 2:
        fig.update_layout(xaxis=dict(tickangle=0), yaxis=dict(tickangle=90))
        for trace in fig.data:
            trace.x, trace.y = trace.y, trace.x
            trace.x = [-x for x in trace.x if x is not None]
    elif turn == 6:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]
    elif turn == 9:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]
    elif turn == 11:
        for trace in fig.data:
            trace.x = [-x for x in trace.x if x is not None]
            trace.y = [-y for y in trace.y if y is not None]

    good_graphs.append(fig)


bad_image_sources = [
    f'assets/icons/Turn{largest_indices[0]}/{outputs[0]}.png',
    f'assets/icons/Turn{largest_indices[1]}/{outputs[1]}.png',
    f'assets/icons/Turn{largest_indices[2]}/{outputs[2]}.png',
    f'assets/icons/Turn{largest_indices[3]}/{outputs[3]}.png'
]

good_image_sources = [
    f'assets/icons/Turn{turns[-1]}/{good_outputs[-1]}.png',
    f'assets/icons/Turn{turns[-2]}/{good_outputs[-2]}.png',
    f'assets/icons/Turn{turns[-3]}/{good_outputs[-3]}.png',
    f'assets/icons/Turn{turns[-4]}/{good_outputs[-4]}.png'
]

for i in range(len(input_type) - 1, -1, -1):
    if largest_values[i] == 0:
        del input_type[i]


########################################################################################################################
# creating the actual layout of the webapp
########################################################################################################################

app.layout = html.Div(
    children=[

        html.Div(children=[
            html.H1('TrackMaster, your personalized sim racing dashboard',
                    style={
                        'text-align': 'center',
                        'color': 'white',
                        'background-color': '#313639'
                    }
                    ),
        ]
        ),
        html.Div(
            children=[
                html.H2("Corners where you can", className="header"),
                html.H2("gain the most", className='header'),
                html.Div(
                    [
                        html.Img(
                            src=bad_image_sources[0],
                            id="image",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[0]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[0]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph',
                                                            config={'displayModeBar': False},
                                                            figure=bad_graphs[0]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                )

                                            ],
                                            className='parent'
                                        ),
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close", className="ml-auto", style={'border': 'none'})
                                ),
                            ],
                            id="modal",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),

                html.Div(
                    [
                        html.Img(
                            src=bad_image_sources[1],
                            id="image1",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[1]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image1",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[1]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph1',
                                                            config={'displayModeBar': False},
                                                            figure=bad_graphs[1]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                ),
                                            ],
                                            className='parent'
                                        ),
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close1", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal1",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),

                html.Div(
                    [
                        html.Img(
                            src=bad_image_sources[2],
                            id="image2",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[2]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image2",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[2]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph2',
                                                            config={'displayModeBar': False},
                                                            figure=bad_graphs[2]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                ),
                                            ],
                                            className='parent'
                                        ),
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close2", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal2",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),

                html.Div(
                    [
                        html.Img(
                            src=bad_image_sources[3],
                            id="image3",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[3]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image3",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[3]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph3',
                                                            config={'displayModeBar': False},
                                                            figure=bad_graphs[3]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                ),
                                            ],
                                            className='parent'
                                        ),
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close3", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal3",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),
            ],
            className="column"

        ),

        html.Div(
            children=[
                html.H2("Corners where you're", className="header"),
                html.H2("doing great", className='header'),
                html.Div(
                    [
                        html.Img(
                            src=good_image_sources[0],
                            id="image4",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[-1]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image4",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[-1]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph4',
                                                            config={'displayModeBar': False},
                                                            figure=good_graphs[0]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                ),
                                            ],
                                            className='parent'
                                        )
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close4", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal4",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),

                html.Div(
                    [
                        html.Img(
                            src=good_image_sources[1],
                            id="image5",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[-1]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image5",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[-2]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph5',
                                                            config={'displayModeBar': False},
                                                            figure=good_graphs[1]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                )
                                            ],
                                            className='parent'
                                        )
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close5", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal5",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),

                html.Div(
                    [
                        html.Img(
                            src=good_image_sources[2],
                            id="image6",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[-3]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image6",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[-3]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph6',
                                                            config={'displayModeBar': False},
                                                            figure=good_graphs[2]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                )
                                            ],
                                            className='parent'
                                        )
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close6", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal6",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),

                html.Div(
                    [
                        html.Img(
                            src=good_image_sources[3],
                            id="image7",
                            className="clickable-image",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalBody(
                                    [
                                        html.H1(f"Turn {turns[-4]}", style={'text-align': 'center'}),
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="modal-image7",
                                                    className="modal-content-image",
                                                ),
                                                dbc.Container(
                                                    [
                                                        html.Img(src=f'assets/icons/{input_type[-4]}.png',
                                                                 className='mb-3'),
                                                        dcc.Graph(
                                                            id='example-graph7',
                                                            config={'displayModeBar': False},
                                                            figure=good_graphs[3]
                                                        ),
                                                    ],
                                                    className='dflex flex-column',
                                                    style={'width': 'auto'}
                                                )
                                            ],
                                            className='parent'
                                        )
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="modal-close7", className="ml-auto",
                                               style={'border': 'none'})
                                ),
                            ],
                            id="modal7",
                            centered=True,
                            className='transparent-modal',
                            size='lg',
                        ),
                    ],
                    className="image-container",
                ),
            ],
            className="column"
        ),

        html.Div(
            children=[
                html.H2("Session statistics", className="header"),
                html.H3(f"Circuit: {formatted_trackname}", className='info'),

                html.H3(f"Car: {formatted_carname}", className='info'),
                html.H3(f"Fastest lap time: {formatted_best_laptime}", className="time"),
                html.H3(f'Average lap time: {average_laptime}', className="time"),
                html.H3(f'Number of racing laps: {corrected_lapcount}', className='info'),
                html.Div(children=[

                    dbc.Button(
                        html.Img(
                            id='mappie',
                            src='assets/icons/map.png',
                            className='map'
                        ),
                        id='open-modal-map',
                        className='button'
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalBody(
                                [
                                    html.H2("The Zandvoort circuit", style={'text-align': 'center'}),
                                    html.Img(id='expanded-image-map', className='map-fit'),
                                ]
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Close", id='close-modal-map', className="ml-auto"),
                            ),
                        ],
                        id="image-modal-map",
                        className='transparent-modal'
                    )

                ], className='map_parent'),
            ],
            className="column"
        ),

    ],
)


def create_modal_callback(image_id, modal_id, modal_image_id, just_id, callback_turn, in_type):
    @app.callback(
        Output(modal_id, "is_open"),
        Output(modal_image_id, "src"),
        Input(image_id, "n_clicks"),
        Input(f"modal-close{just_id}", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_modal(image_clicks, close_clicks):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == image_id:
            return True, f'assets/Pictures/{"Brake" if in_type == "Brake" else "Track"}Turn{callback_turn}.jpg'

        return False, None

    return open_modal


open_modal0 = create_modal_callback("image", "modal", "modal-image", "", largest_indices[0], input_type[0])
open_modal1 = create_modal_callback("image1", "modal1", "modal-image1", "1", largest_indices[1], input_type[1])
open_modal2 = create_modal_callback("image2", "modal2", "modal-image2", "2", largest_indices[2], input_type[2])
open_modal3 = create_modal_callback("image3", "modal3", "modal-image3", "3", largest_indices[3], input_type[3])
open_modal4 = create_modal_callback("image4", "modal4", "modal-image4", "4", turns[-1], input_type[-1])
open_modal5 = create_modal_callback("image5", "modal5", "modal-image5", "5", turns[-2], input_type[-2])
open_modal6 = create_modal_callback("image6", "modal6", "modal-image6", "6", turns[-3], input_type[-3])
open_modal7 = create_modal_callback("image7", "modal7", "modal-image7", "7", turns[-4], input_type[-4])


@app.callback(
    Output("image-modal-map", "is_open"),
    Output("expanded-image-map", "src"),
    Input("open-modal-map", "n_clicks"),
    Input("close-modal-map", "n_clicks"),
    State("image-modal-map", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(open_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'open-modal-map':
        return not is_open, "assets/icons/map.png"

    return is_open, None


if __name__ == '__main__':
    app.run_server(debug=False)
