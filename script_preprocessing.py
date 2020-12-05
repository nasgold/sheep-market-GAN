import numpy as np
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from module_utils import draw_strokes, save_svg2png


def main():
    data_file = 'sheep_market.npz'
    data_dict = np.load(data_file, allow_pickle=True, encoding='bytes')
    # each is a list of [n_strokes x 3] dimension drawings
    train_data = data_dict['train']
    train_end = len(train_data)
    validation_data = data_dict['valid']
    validation_end = train_end + len(validation_data)
    test_data = data_dict['test']
    all_data = np.concatenate((train_data, validation_data, test_data))

    ex_sketch_idxs = np.random.randint(0, len(all_data), size=4)
    write_dir = "sketches"
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    for i in range(len(ex_sketch_idxs)):
        idx = ex_sketch_idxs[i]
        cur_sketch = all_data[idx]
        svg_filename = f"real_ex_{str(i).zfill(4)}.svg"
        svg_filepath = os.path.join(write_dir, svg_filename)
        draw_strokes(cur_sketch, svg_filename=svg_filepath)
        png_filepath = svg_filepath[:-3] + "png"
        save_svg2png(svg_filepath, png_filepath)
        os.remove(svg_filepath)

    all_tensors = []
    all_lens = []
    for sketch in all_data:
        all_tensors.append(torch.from_numpy(sketch))
        all_lens.append(len(sketch))

    padded_tensors = pad_sequence(
                            all_tensors,
                            batch_first=True
                            )

    train_tensors = padded_tensors[:train_end]
    validation_tensors = padded_tensors[train_end:validation_end]
    test_tensors = padded_tensors[validation_end:]

    tensor_dict = {
                'train': train_tensors,
                'validation': validation_tensors,
                'test': test_tensors
                }

    torch.save(tensor_dict, 'sheep_market_preprocessed.pt')


if __name__ == '__main__':
    main()
