import os
import time
import numpy as np

import progress.bar

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from . import utils


def train(opts, model, train_data, val_data, num_epochs, resume_from_epoch=None):
    train_loader = DataLoader(train_data,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=opts.dataloader_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=opts.batch_size,
                            shuffle=False,
                            num_workers=opts.dataloader_workers,
                            pin_memory=True)

    print('Training images: {}'.format(len(train_data)))
    print('Validation images: {}'.format(len(val_data)))

    log_dir = os.path.join(opts.results_dir, opts.experiment_name)
    writer = SummaryWriter(log_dir)

    ### LOAD FROM CHECKPOINT ###
    if resume_from_epoch is not None:
        try:
            initial_epoch = model.load_checkpoint(resume_from_epoch) + 1
            iterations = (initial_epoch - 1) * opts.batch_size
        except FileNotFoundError:
            print('No model available for epoch {}, starting fresh'.format(
                resume_from_epoch))
            initial_epoch = 1
            iterations = 0
    else:
        initial_epoch = 1
        iterations = 0

    ### TRAIN AND VALIDATE ###
    best_total_val_loss = 1e12

    for epoch in range(initial_epoch, num_epochs + 1):
        epoch_start = time.perf_counter()

        # TRAIN
        epoch_train_loss = None
        model.set_mode('train')

        bar = progress.bar.Bar('Epoch {} train'.format(
            epoch), max=len(train_loader))
        for data in train_loader:
            model.set_data(data)
            model.optimize()
            if epoch_train_loss is None:
                epoch_train_loss = model.get_errors()
            else:
                epoch_train_loss = utils.concatenate_dicts(
                    epoch_train_loss, model.get_errors())

            iterations += 1
            bar.next()
        bar.finish()

        # VISUALIZE
        for label, image in model.get_images().items():
            image = np.array(image).transpose([2, 0, 1])
            writer.add_image('train/'+label, image, epoch)

        train_end = time.perf_counter()

        # VALIDATE
        epoch_val_loss = None
        model.set_mode('eval')

        bar = progress.bar.Bar(
            'Epoch {} val  '.format(epoch), max=len(val_loader))
        for data in val_loader:
            model.set_data(data)
            model.test(compute_loss=True)
            if epoch_val_loss is None:
                epoch_val_loss = model.get_errors()
            else:
                epoch_val_loss = utils.concatenate_dicts(
                    epoch_val_loss, model.get_errors())

            bar.next()
        bar.finish()

        for label, image in model.get_images().items():
            image = np.array(image).transpose([2, 0, 1])
            writer.add_image('val/'+label, image, epoch)

        epoch_end = time.perf_counter()

        epoch_avg_val_loss = utils.compute_dict_avg(epoch_val_loss)
        epoch_avg_train_loss = utils.compute_dict_avg(epoch_train_loss)
        train_fps = len(train_data)/(train_end-epoch_start)
        val_fps = len(val_data)/(epoch_end-train_end)

        print('End of epoch {}/{} | iter: {} | time: {:.3f} s | train: {:.3f} fps | val: {:.3f} fps'.format(
            epoch, num_epochs, iterations, epoch_end - epoch_start,
            train_fps, val_fps))

        # LOG ERRORS
        errors = utils.tag_dict_keys(epoch_avg_train_loss, 'train')
        errors.update(utils.tag_dict_keys(epoch_avg_val_loss, 'val'))
        for key, value in sorted(errors.items()):
            writer.add_scalar(key, value, epoch)
            print('{:20}: {:.3e}'.format(key, value))

        writer.add_scalar('fps/train', train_fps, epoch)
        writer.add_scalar('fps/val', val_fps, epoch)

        # SAVE MODELS
        model.save_checkpoint(epoch, 'latest')

        if epoch % opts.checkpoint_interval == 0:
            model.save_checkpoint(epoch, epoch)

        curr_total_val_loss = 0
        for key, val in epoch_avg_val_loss.items():
            if 'eval_loss' in key:
                try:
                    curr_total_val_loss += val[-1]
                except IndexError:
                    curr_total_val_loss += val

        if epoch == 1 or curr_total_val_loss < best_total_val_loss:
            model.save_checkpoint(epoch, 'best')
            best_total_val_loss = curr_total_val_loss


def test(opts, model, test_data, which_epoch='best', batch_size=1, expdir=None,
         save_loss=False, save_images=True):
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=opts.dataloader_workers,
                             pin_memory=True)

    model.load_checkpoint(which_epoch)
    model.set_mode('eval')

    output_dir = os.path.join(opts.results_dir,
                              opts.experiment_name if expdir is None else expdir,
                              'test_{}'.format(which_epoch))
    os.makedirs(output_dir, exist_ok=True)

    test_start = time.perf_counter()

    test_loss = None

    bar = progress.bar.Bar('Test', max=len(test_loader))
    for idx, data in enumerate(test_loader):
        model.set_data(data)
        model.test(compute_loss=save_loss)

        if save_loss:
            if test_loss is None:
                test_loss = model.get_errors()
            else:
                test_loss = utils.concatenate_dicts(
                    test_loss, model.get_errors())

        if save_images:
            output = model.get_images()

            for img_label, img in output.items():
                this_output_dir = os.path.join(output_dir, img_label)
                os.makedirs(this_output_dir, exist_ok=True)

                output_file = os.path.join(
                    this_output_dir, '{:05}.png'.format(idx))
                # print("Saving to {}".format(output_file))
                img.save(output_file)

        bar.next()
    bar.finish()

    test_end = time.perf_counter()
    test_fps = len(test_data)/(test_end-test_start)
    print('Processed {} images | time: {:.3f} s | test: {:.3f} fps'.format(
        len(test_data), test_end-test_start, test_fps))

    if save_loss:
        loss_file = os.path.join(output_dir, 'loss.csv')
        header = [key for key in test_loss]
        entries = [test_loss[key] for key in test_loss]
        entries = np.atleast_2d(np.array(entries)).T.tolist()

        print("Saving test loss to {}".format(loss_file))
        with open(loss_file, 'wt') as file:
            file.write(','.join(header) + '\n')
            for entry in entries:
                line = ','.join([str(val) for val in entry]) + '\n'
                file.write(line)
