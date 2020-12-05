"""Training regime for a GAN using data from The Sheep Market.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict

from module_GAN import sketchGenerator, sketchDiscriminator
from module_utils import (
    sketchLoader, get_priors, draw_strokes, plot_losses, save_svg2png
    )


def main():
    opts = EasyDict()
    opts.n_dims_per_stroke = 3
    opts.n_cells = 250
    opts.n_dims_hidden_layer = 128
    opts.n_epochs = 2**9
    opts.n_epochs_netG_pre = 2**9
    opts.batch_size = 2**6
    opts.lrD = 0.0001
    opts.lrG = 0.0001
    opts.lr_preG = .0001
    opts.grad_max = 10
    opts.write_dir = "sketches"

    netD = sketchDiscriminator(
        opts.n_dims_per_stroke,
        opts.n_cells,
        opts.n_dims_hidden_layer
        )
    netG = sketchGenerator(
        opts.n_dims_per_stroke,
        opts.n_cells,
        opts.n_dims_hidden_layer
        )

    real_label = 1
    gen_label = 0
    criterion = nn.BCELoss()
    pre_criterion = nn.MSELoss()

    optimizerD = optim.Adam(
        netD.parameters(),
        lr=opts.lrD
        )
    optimizerG = optim.Adam(
        netG.parameters(),
        lr=opts.lrG
        )
    optimizer_preG = optim.Adam(
        netG.parameters(),
        lr=opts.lr_preG
        )

    pre_sketch_list = []
    sketch_list = []
    lossesD = []
    lossesG = []

    tensor_file = 'sheep_market_preprocessed.pt'
    dataloader = sketchLoader(opts.batch_size, tensor_file)
    dataloader.transform_data()

    fixed_priors_batch_size = 1
    fixed_priors = get_priors(
        fixed_priors_batch_size,
        opts.n_dims_per_stroke,
        opts.n_dims_hidden_layer
        )

    #############################
    # netG pre-training
    #############################
    for epoch in range(opts.n_epochs_netG_pre):
        netG.zero_grad()

        real_sketches = dataloader.load_next_batch()
        hidden_states, cell_states = netG.encode(real_sketches)
        init_inputs = torch.randn(
            opts.batch_size,
            opts.n_dims_per_stroke,
            dtype=torch.float
            )
        priors = [init_inputs, hidden_states, cell_states]
        gen_sketches = netG.generate(priors)

        netG_pre_err = pre_criterion(gen_sketches, real_sketches)
        netG_pre_err.backward()
        optimizer_preG.step()
        netG_epoch_pre_err = netG_pre_err.item()

        if epoch % 2**7 == 0:
            with torch.no_grad():
                gen_sketch_data = netG.generate(fixed_priors).squeeze()
                gen_sketch_data = dataloader.inverse_transform(gen_sketch_data)
                pre_sketch_list.append(gen_sketch_data)

        epoch_end_str = f"pre-epoch: {epoch} __"
        epoch_end_str += f" netG pre-training loss: {netG_epoch_pre_err} __"
        print(epoch_end_str)
        print("\n")

    #############################
    # WARNING: GAN AT WORK (SCHOOL?)
    #############################
    for epoch in range(opts.n_epochs):
        sketch_data = dataloader.load_next_batch()

        # first train the discriminator with real data
        netD.zero_grad()
        netD_real_target = torch.full(
                                (opts.batch_size, ),
                                real_label,
                                dtype=torch.float
                                )
        netD_real_output = netD.classify(sketch_data)
        netD_real_err = criterion(netD_real_output, netD_real_target)
        netD_real_err.backward()

        # now generate some sketches
        priors = get_priors(
                        opts.batch_size,
                        opts.n_dims_per_stroke,
                        opts.n_dims_hidden_layer
                        )
        gen_sketch_data = netG.generate(priors)

        # train the discriminator on the gen data
        netD_gen_target = torch.full(
                                (opts.batch_size, ),
                                gen_label,
                                dtype=torch.float
                                )
        netD_gen_output = netD.classify(gen_sketch_data.detach())
        netD_gen_err = criterion(netD_gen_output, netD_gen_target)
        netD_gen_err.backward()
        optimizerD.step()
        netD_epoch_err = (netD_real_err + netD_real_err).item()

        # now update generator using same generated sketch data
        netG.zero_grad()
        netG_target = torch.full(
                            (opts.batch_size, ),
                            real_label,
                            dtype=torch.float
                            )
        netD_gen_output_forG = netD.classify(gen_sketch_data)
        netG_err = criterion(netD_gen_output_forG, netG_target)
        netG_err.backward()
        optimizerG.step()
        netG_epoch_err = netG_err.item()

        # check to make sure the gradients' 2norms don't look off
        netD_grad_2norms = []
        netG_grad_2norms = []
        for p in netD.parameters():
            try:
                cur_2norm = p.grad.data.norm().item()
                netD_grad_2norms.append(cur_2norm)
            except AttributeError:
                continue
        for p in netG.parameters():
            try:
                cur_2norm = p.grad.data.norm().item()
                netG_grad_2norms.append(cur_2norm)
            except AttributeError:
                continue
        grad_2norms = np.concatenate((netD_grad_2norms, netG_grad_2norms))
        if grad_2norms.max() > opts.grad_max:
            "WARNING: Potential gradient explosion."

        if epoch % 2**7 == 0:
            with torch.no_grad():
                gen_sketch_data = netG.generate(fixed_priors).squeeze()
                gen_sketch_data = dataloader.inverse_transform(gen_sketch_data)
                sketch_list.append(gen_sketch_data)

        lossesD.append(netD_epoch_err)
        lossesG.append(netG_epoch_err)

        epoch_end_str = f"epoch: {epoch} __"
        epoch_end_str += f" netD training loss: {netD_epoch_err} __"
        epoch_end_str += f" netG training loss: {netG_epoch_err} __"
        print(epoch_end_str)
        print("\n")

    # plot the losses for each net across training iterations
    plot_losses(lossesD, lossesG)

    for i in range(len(pre_sketch_list)):
        cur_sketch = pre_sketch_list[i]
        svg_filename = f"pre_gen_ex_{str(i).zfill(4)}.svg"
        svg_filepath = os.path.join(opts.write_dir, svg_filename)
        draw_strokes(cur_sketch, svg_filename=svg_filepath)
        png_filepath = svg_filepath[:-3] + "png"
        save_svg2png(svg_filepath, png_filepath)
        os.remove(svg_filepath)

    for i in range(len(sketch_list)):
        cur_sketch = pre_sketch_list[i]
        svg_filename = f"gen_ex_{str(i).zfill(4)}.svg"
        svg_filepath = os.path.join(opts.write_dir, svg_filename)
        draw_strokes(cur_sketch, svg_filename=svg_filepath)
        png_filepath = svg_filepath[:-3] + "png"
        save_svg2png(svg_filepath, png_filepath)
        os.remove(svg_filepath)


if __name__ == '__main__':
    main()
