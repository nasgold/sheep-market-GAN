"""Classes that together define a GAN. The encoder is only used if pre-training
the generator on reconstruction loss.
"""

import torch
import torch.nn as nn


class sketchGenerator(nn.Module):
    def __init__(self, n_dims_per_stroke, n_cells, n_dims_hidden_layer):
        super(sketchGenerator, self).__init__()
        self.cells = n_cells
        self.hidden_dim = n_dims_hidden_layer
        self.lstm_cell = nn.LSTMCell(
            n_dims_per_stroke,
            n_dims_hidden_layer
            )
        self.lstm_cell_f = nn.LSTMCell(
            n_dims_per_stroke,
            n_dims_hidden_layer
            )
        self.lstm_cell_b = nn.LSTMCell(
            n_dims_per_stroke,
            n_dims_hidden_layer
            )
        self.linear_encode = nn.Linear(
            n_dims_hidden_layer*2,
            n_dims_hidden_layer
            )
        self.linear_output = nn.Linear(
            n_dims_hidden_layer,
            n_dims_per_stroke
            )
        self.activation = nn.LeakyReLU()

    def encode(self, batch_of_sketches):
        batch_size = batch_of_sketches.shape[0]

        hidden_states_f = torch.zeros(batch_size, self.hidden_dim)
        cell_states_f = torch.zeros(batch_size, self.hidden_dim)
        for i in range(self.cells):
            hidden_states_f, cell_states_f = self.lstm_cell_f(
                batch_of_sketches[:, i, :],
                (hidden_states_f, cell_states_f)
                )

        hidden_states_b = torch.zeros(batch_size, self.hidden_dim)
        cell_states_b = torch.zeros(batch_size, self.hidden_dim)
        for i in range(0, self.cells, -1):
            hidden_states_b, cell_states_b = self.lstm_cell_b(
                batch_of_sketches[:, i, :],
                (hidden_states_b, cell_states_b)
                )

        hidden_states_2 = torch.cat(
            (hidden_states_f, hidden_states_b),
            axis=1
            )
        cell_states_2 = torch.cat(
            (cell_states_f, cell_states_b),
            axis=1
            )
        hidden_states = self.activation(self.linear_encode(hidden_states_2))
        cell_states = self.activation(self.linear_encode(cell_states_2))

        return hidden_states, cell_states

    def generate(self, priors):
        init_inputs, hidden_states, cell_states = priors
        cur_inputs = init_inputs
        outputs = []

        for i in range(self.cells):
            hidden_states, cell_states = self.lstm_cell(
                cur_inputs,
                (hidden_states, cell_states)
                )
            cur_outputs = self.activation((self.linear_output(hidden_states)))
            outputs.append(cur_outputs)

            cur_inputs = cur_outputs

        return torch.stack(outputs, 1)


class sketchDiscriminator(nn.Module):
    def __init__(self, n_dims_per_stroke, n_cells, n_dims_hidden_layer):
        super(sketchDiscriminator, self).__init__()
        self.cells = n_cells
        self.hidden_size = n_dims_hidden_layer
        self.lstm_cell = nn.LSTMCell(n_dims_per_stroke, n_dims_hidden_layer)
        self.linear_output = nn.Linear(n_dims_hidden_layer, 1)
        self.activation = nn.LeakyReLU(0.10)
        self.sigmoid = nn.Sigmoid()

    def classify(self, strokes):
        batch_size = strokes.shape[0]
        hidden_states = torch.zeros(batch_size, self.hidden_size)
        cell_states = torch.zeros(batch_size, self.hidden_size)
        annotations = []

        for i in range(self.cells):
            cur_strokes = strokes[:, i, :].float()
            hidden_states, cell_states = self.lstm_cell(
                cur_strokes,
                (hidden_states, cell_states)
                )
            annotations.append(hidden_states)

        raw_output = self.linear_output(hidden_states).squeeze()
        raw_probs = self.activation(raw_output)

        return self.sigmoid(raw_probs)
