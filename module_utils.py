import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

import svgwrite
from cairosvg import svg2png
from sklearn.preprocessing import StandardScaler


def plot_losses(lossD, lossG):
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(lossD, label="Discriminator loss")
    ax.plot(lossG, label="Generator loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("BCE Loss")
    plt.legend()
    plt.savefig("train_losses.png")


class sketchLoader():
    """Data loader class for the model. Loads data from given .npz file and
    gives access to batches. Also provides methods for scaling input data.
    """
    def __init__(self, batch_size, tensor_file):
        self.batch_size = batch_size
        tensor_dict = torch.load(tensor_file)
        self.train_data = tensor_dict['train']
        self.train_idx = 0
        self.validation_data = tensor_dict['validation']
        self.validation_idx = 0
        self.test_data = tensor_dict['test']
        self.test_idx = 0

        all_sketches = np.concatenate((self.train_data,
                                       self.validation_data,
                                       self.test_data))
        all_strokes = np.reshape(all_sketches, (-1, 3))
        self.scaler = StandardScaler()
        self.scaler.fit(all_strokes)

    def load_next_batch(self, type='train'):
        if type == 'train':
            data = self.train_data
            i = self.train_idx
            self.train_idx += 1
        elif type == 'validation':
            data = self.validation_data
            i = self.validation_idx
            self.validation_idx += 1
        elif type == 'test':
            data = self.test_data
            i = self.test_idx
            self.test_idx += 1

        try:
            batch = data[i:i+self.batch_size]
        except IndexError:
            batch = data[i:]

        return batch

    def transform_data(self):
        self.train_data = torch.tensor(
            [self.scaler.transform(sketch)
             for sketch
             in self.train_data],
            dtype=torch.float
            )
        self.validation_data = torch.tensor(
            [self.scaler.transform(sketch)
             for sketch
             in self.validation_data],
            dtype=torch.float
             )
        self.test_data = torch.tensor(
            [self.scaler.transform(sketch)
             for sketch
             in self.test_data],
            dtype=torch.float
            )

    def inverse_transform(self, sketch):
        return self.scaler.inverse_transform(sketch)


def get_priors(batch_size, embed_dim, hidden_dim):
    """Returns a list of tensors, each with the specified dims. Currently used
    to generate an initial input, hidden state, and cell state.
    """
    batch_input = torch.randn(batch_size, embed_dim, dtype=torch.float)
    batch_hidden = torch.randn(batch_size, hidden_dim, dtype=torch.float)
    batch_cell = torch.randn(batch_size, hidden_dim, dtype=torch.float)

    return [batch_input, batch_hidden, batch_cell]


# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.8, svg_filename='sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0])/factor
        y = float(data[i, 1])/factor
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    # display(SVG(dwg.tostring()))


def save_svg2png(svg_filename, png_filename):
    svg2png(
        open(svg_filename, 'rb').read(),
        write_to=open(png_filename, 'wb')
        )


# helper function for draw_strokes
def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0])/factor
        y = float(data[i, 1])/factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=100.0, grid_space_x=200.0):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max+x_min)*0.5
        return x_start-center_loc, x_end
    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0]*grid_space+grid_space*0.5
        grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x+loc_x
        new_y_pos = grid_y+loc_y
        result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos+delta_pos[0]
        y_pos = new_y_pos+delta_pos[1]
    return np.array(result)
