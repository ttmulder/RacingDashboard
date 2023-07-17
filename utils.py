import ast
import datetime
import numpy as np
import math


# properly sort the coordinates, so they can be displayed in a plot
def get_map_coordinates(unsorted):
    coordinates = remove_y_coordinate(unsorted)

    x_coordinates = [x[0] for x in coordinates]
    y_coordinates = [y[1] for y in coordinates]
    sorted_arrays = sorted(zip(x_coordinates, y_coordinates))
    sorted_x_coordinates = [x for x, _ in sorted_arrays]
    sorted_y_coordinates = [y for _, y in sorted_arrays]

    return sorted_x_coordinates, sorted_y_coordinates


# function to remove the unnecessary height coordinate
def remove_y_coordinate(world_coordinates):
    coordinates = []
    for i in range(len(world_coordinates)):
        temp = [world_coordinates[i]]
        tuple_val = ast.literal_eval(temp[0])
        float_list = [float(value) for value in tuple_val]
        del float_list[1]
        coordinates.append(float_list)

    return coordinates


def make_coordinate_pair(ding):
    for i in range(len(ding)):
        temp = [ding[i]]
        tuple_val = ast.literal_eval(temp[0])
        float_list = [float(value) for value in tuple_val]


# create a 'chunk' of the range of lap position for every corner
def create_chunks(dataframe):
    chunks = []
    starters = dataframe['start']
    enders = dataframe['end']
    for i in range(len(starters)):
        temp = [starters[i], enders[i]]
        chunks.append(temp)
    return chunks


# split the entire dataframe per lap and per corner
def split_data(dataframe, chunks):
    dfs = []
    laps = set(dataframe['lap count'])

    for lap in laps:
        dfLaps = dataframe[dataframe['lap count'] == lap]

        temp2 = []
        for chunk in chunks:
            val1, val2 = map(float, chunk)
            dfTurn = dfLaps[(dfLaps['lap position'] >= val1) & (dfLaps['lap position'] <= val2)]
            temp2.append(dfTurn)

        dfs.append(temp2)

    return dfs


def get_point(dataframe, input):
    data = dataframe[str(input)].values

    # Find the change point by thresholds
    if input == 'gas':
        for i in range(len(data)):
            # find where the amount of gas input passes a certain threshold (0.35)
            if data[i] >= 0.35 > data[i - 1] > 0.25:
                for j in range(len(data)):
                    # find the point where the increase to threshold started
                    if data[i - j] == 0:
                        result = i - j + 1
                        return [dataframe.iloc[result]['x'], dataframe.iloc[result]['y'],
                                dataframe.iloc[result]['lap position']]
        return 0
    elif input == 'brake':
        for i in range(len(data)):
            # find where the amount of brake input passes a certain threshold (0.25)
            if data[i] >= 0.25 > data[i - 1] > 0.10:
                for j in range(len(data)):
                    # find the point where the increase to threshold started
                    if data[i - j] == 0:
                        result = i - j + 1
                        return [dataframe.iloc[result]['x'], dataframe.iloc[result]['y'],
                                dataframe.iloc[result]['lap position']]
    elif input == 'steer':
        for i in range(len(data)):
            # find where the amount of steer input passes a certain threshold (+35 degrees or -35 degrees)
            if data[i] >= 35 > data[i - 1] > 25:
                for j in range(len(data)):
                    # find the point where the increase to threshold started
                    if data[i - j] < 5:
                        result = i - j + 1
                        return [dataframe.iloc[result]['x'], dataframe.iloc[result]['y'],
                                dataframe.iloc[result]['lap position']]
            elif data[i] <= -35 < data[i - 1] < -25:
                for j in range(len(data)):
                    # find the point where the increase to threshold started
                    if data[i - j] > -5:
                        result = i - j + 1
                        return [dataframe.iloc[result]['x'], dataframe.iloc[result]['y'],
                                dataframe.iloc[result]['lap position']]


def get_all_points(dataframe):
    """
    get the gas, brake, steer points for all corners of all laps (except out-lap and in-lap)
    :param dataframe: the dataFrame that we get from @split_data
    :return: array of arrays with the points
    """
    inputs = ['brake', 'steer', 'gas']
    points = []
    for i in range(1, len(dataframe) - 1):
        lap = []
        for j in range(len(dataframe[i])):
            corner = []
            for input in inputs:
                corner.append(get_point(dataframe[i][j], input))
            lap.append(corner)
        points.append(lap)
    return points


def to_coordinates(string):
    if string == '(x.x-x.x)':
        return
    else:
        array_nums = [float(num) for num in string.strip('()').split('/')]
        return array_nums


def all_to_coordinates(dataframe):
    points = ['brakepoint', 'steeringpoint', 'accelerationpoint']
    result = []
    for point in points:
        data = dataframe[point].values
        temp = []
        for i in range(len(data)):
            temp.append(to_coordinates(data[i]))
        result.append(temp)
    return result


# calculate the distance between the perfect input point and the actual input point
def calc_distance(perf_point, act_point):
    if perf_point is None or act_point is None or perf_point == 0 or act_point == 0:
        return None
    distance = math.sqrt((act_point[0] - perf_point[0]) ** 2 + (act_point[1] - perf_point[1]) ** 2)
    if act_point[2] < perf_point[2]:
        distance = -distance
    return distance


def remove_none_entries(arr):
    return [num for num in arr if num is not None]


def calc_all_distances_and_averages(points, perfect_points):
    distances = []
    average_points_x = []
    average_points_y = []
    for i in range(len(points)):
        temp2_avg_x = []
        temp2_avg_y = []
        temp2 = []
        for j in range(len(points[i])):
            temp = []
            temp_avg_x = []
            temp_avg_y = []
            for k in range(len(points[i][j])):
                dis = calc_distance(perfect_points[k][j], points[i][j][k])
                temp.append(dis)
                if points[i][j][k] is None or points[i][j][k] == 0:
                    temp_avg_x.append(None)
                    temp_avg_y.append(None)
                else:
                    temp_avg_x.append(points[i][j][k][0])
                    temp_avg_y.append(points[i][j][k][1])
            temp2.append(temp)
            temp2_avg_x.append(temp_avg_x)
            temp2_avg_y.append(temp_avg_y)
        distances.append(temp2)
        average_points_x.append(temp2_avg_x)
        average_points_y.append(temp2_avg_y)
    return distances, average_points_x, average_points_y


def get_averages(distances, points_x, points_y):
    average_distances = []
    average_x = []
    average_y = []
    for k in range(0, 3):
        per_inputs_dis = []
        per_inputs_x = []
        per_inputs_y = []
        for j in range(len(distances[0])):
            per_lap_dis = []
            per_lap_x = []
            per_lap_y = []
            for i in range(len(distances)):
                if distances[i][j][k] is not None:  # Check for None values
                    per_lap_dis.append(distances[i][j][k])
                    per_lap_x.append(points_x[i][j][k])
                    per_lap_y.append(points_y[i][j][k])
            if per_lap_dis:  # Check if per_lap is not empty
                # Calculate average and add to per_inputs
                per_inputs_dis.append(sum(per_lap_dis) / len(per_lap_dis))
                per_inputs_x.append(sum(per_lap_x) / len(per_lap_x))
                per_inputs_y.append(sum(per_lap_y) / len(per_lap_y))
            else:
                per_inputs_dis.append(0)
                per_inputs_x.append(None)
                per_inputs_y.append(None)
        average_distances.append(per_inputs_dis)
        average_x.append(per_inputs_x)
        average_y.append(per_inputs_y)
    return average_distances, average_x, average_y


def get_worst(averages):
    # Flatten the nested array to a 1D array
    flattened_array = np.array(averages).flatten()

    # Get the indices of the largest absolute values
    largest_indices = np.argsort(np.abs(flattened_array))
    largest_values = flattened_array[largest_indices]
    input_type = []
    input_type_int = []
    for i in range(len(largest_indices)):
        if largest_indices[i] <= 12:
            largest_indices[i] = largest_indices[i] + 1
            input_type.append('Brake')
            input_type_int.append(0)
        elif 12 < largest_indices[i] <= 25:
            largest_indices[i] = largest_indices[i] - 12
            input_type.append('Steer')
            input_type_int.append(1)
        elif largest_indices[i] > 25:
            largest_indices[i] = largest_indices[i] - 25
            input_type.append('Acc')
            input_type_int.append(2)
    largest_values = largest_values[::-1]
    largest_indices = largest_indices[::-1]
    input_type = input_type[::-1]
    input_type_int = input_type_int[::-1]
    return largest_values, largest_indices, input_type, input_type_int


def get_average_laptime(dataframe):
    laptimes = sorted(list(set(dataframe['last lap'].values)))
    # trim the average laptime by removing the first lap (always value 0) and the fastest and slowest lap
    # we do this to account for cooldown laps etc.
    laptimes = laptimes[2:-1]
    average = np.mean(laptimes)
    return average


def format_laptime(laptime):
    formatted_time = datetime.timedelta(milliseconds=int(laptime))

    # Extract minutes, seconds, and milliseconds
    minutes = formatted_time.seconds // 60
    seconds = formatted_time.seconds % 60
    milliseconds = formatted_time.microseconds // 1000

    formatted_laptime = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    return formatted_laptime


def check_full_throttle(dataframe):
    full_throttle = True
    for i in range(1, len(dataframe) - 1):
        if min(dataframe[i][3]['gas'].values) > 0.95 and min(dataframe[i][4]['gas'].values) > 0.95:
            full_throttle = True
        else:
            return False
    return full_throttle


def find_perfect_coordinates(perfect_points):
    perfect_coordinates_x = []
    perfect_coordinates_y = []
    # for every input: brake, steer, acc
    for i in range(len(perfect_points)):
        temp_x = []
        temp_y = []
        # for every corner
        for j in range(len(perfect_points[i])):
            temp2 = perfect_points[i][j]
            if type(temp2) is type(None):
                temp_x.append(None)
                temp_y.append(None)
            else:
                temp_x.append(temp2[0])
                temp_y.append(temp2[1])
        perfect_coordinates_x.append(temp_x)
        perfect_coordinates_y.append(temp_y)
    return perfect_coordinates_x, perfect_coordinates_y
