import csv
import numpy as np

def pad_worm_csv(fname: str, pad_method):
    """
    Takes in a csv file of worm position data, and pads the ends of each row (representing the position of the worm
    for one frame) using some extrapolation method.
    """
    # read in the csv, and then split it up into the headers, frame ids/time, and actual position data
    worm_positions = []
    with open(fname, 'r') as w_csv:
        reader = csv.reader(w_csv, delimiter=',')
        for row in reader:
            worm_positions.append(row)
    headers = worm_positions[:5]
    worm_positions = worm_positions[5:]

    # pad each row
    max_len = max([len(position) for position in worm_positions])
    pad_positions = []
    for i in range(len(worm_positions)):
        position = [float(coord) for coord in worm_positions[i]]
        pad_len = (max_len - len(position)) // 2
        pad_positions.append(pad_method(position, pad_len))

    assert(all(len(position) == max_len for position in pad_positions))

    last_coord_idx = int(headers[4][-1][:-1])
    pad_len = (max_len - len(headers[4])) // 2
    for i in range(pad_len):
        headers[4].append(f'{i + last_coord_idx + 1}x')
        headers[4].append(f'{i + last_coord_idx + 1}y')

    # generate the padded csv
    new_fname = fname + 'padded.csv'
    with open(new_fname, 'w+', newline='') as pad_csv:
        writer = csv.writer(pad_csv, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for header_row in headers:
            writer.writerow(header_row)

        for position in pad_positions:
            writer.writerow(position)

def linear_pad(position: list[float], n: int):
    """
    Mutates existing list to contain all given positions, plus n extra points that are linearly extrapolated from the
    last point in the original position list.

    >>> body_frame = [1, 1, 2, 2]
    >>> linear_pad(body_frame, 2)
    [1, 1, 2, 2, 3, 3, 4, 4]
    """
    last = np.array([position[-2], position[-1]])
    second_last = np.array([position[-4], position[-3]])

    linear_approx = last - second_last

    for i in range(n):
        new_point = last + (i + 1) * linear_approx
        position.append(new_point[0])
        position.append(new_point[1])

    return position
#
fname = input('file name: ')
pad_worm_csv(fname, linear_pad)
