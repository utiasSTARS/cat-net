import os
import time
import numpy as np

from torch.utils.data import DataLoader

from . import config
from . import utils
from .visualizer import Visualizer


def train(model, train_data, val_data, resume_from_epoch=None):
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.dataloader_workers,
                              pin_memory=config.use_cuda)
    val_loader = DataLoader(val_data,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.dataloader_workers,
                            pin_memory=config.use_cuda)

    print('Training images: {}'.format(len(train_data)))
    print('Validation images: {}'.format(len(val_data)))

    ### LOAD FROM CHECKPOINT ###
    if resume_from_epoch is not None:
        model.load_checkpoint(resume_from_epoch)

    ### TRAIN AND VALIDATE ###
    visualizer = Visualizer()
    epoch_avg_train_loss = {}
    epoch_avg_val_loss = {}
    best_total_val_loss = 1e12
    batches_processed = 0
    random_crop = config.random_crop
    epochs = range(1, config.train_epochs + 1)

    for epoch in epochs:
        epoch_start = time.perf_counter()

        # TRAIN
        epoch_train_loss = {}
        model.set_mode('train')
        config.random_crop = random_crop

        for source, target in train_loader:
            model.set_data(source, target)
            model.optimize()

            if len(epoch_train_loss) == 0:
                epoch_train_loss = model.get_errors()
            else:
                epoch_train_loss = utils.concatenate_dicts(
                    epoch_train_loss, model.get_errors())

            batches_processed += 1
            if batches_processed % config.plot_interval:
                visualizer.show_images(model.get_images())

        this_epoch_avg_train_loss = utils.compute_dict_avg(epoch_train_loss)
        if len(epoch_avg_train_loss) == 0:
            epoch_avg_train_loss = this_epoch_avg_train_loss
        else:
            epoch_avg_train_loss = utils.concatenate_dicts(
                epoch_avg_train_loss, this_epoch_avg_train_loss)

        # VALIDATE
        epoch_val_loss = {}
        model.set_mode('eval')
        config.random_crop = False

        for source, target in val_loader:
            model.set_data(source, target)
            model.test()
            if len(epoch_val_loss) == 0:
                epoch_val_loss = model.get_errors()
            else:
                epoch_val_loss = utils.concatenate_dicts(
                    epoch_val_loss, model.get_errors())

        this_epoch_avg_val_loss = utils.compute_dict_avg(epoch_val_loss)
        if len(epoch_avg_val_loss) == 0:
            epoch_avg_val_loss = this_epoch_avg_val_loss
        else:
            epoch_avg_val_loss = utils.concatenate_dicts(
                epoch_avg_val_loss, this_epoch_avg_val_loss)

        epoch_end = time.perf_counter()

        print('End of epoch {}/{} | time: {:.3f} s'.format(
            epoch, config.train_epochs, epoch_end - epoch_start))

        # VISUALIZE
        errors = utils.tag_dict_keys(this_epoch_avg_train_loss, 'avg_train')
        errors.update(utils.tag_dict_keys(this_epoch_avg_val_loss, 'avg_val'))
        visualizer.print_errors(errors)
        visualizer.plot_errors(epoch, errors)

        # SAVE
        model.save_checkpoint('latest')

        if epoch % config.save_interval == 0:
            model.save_checkpoint(epoch)

        curr_total_val_loss = 0
        for key, val in epoch_avg_val_loss.items():
            try:
                curr_total_val_loss += val[-1]
            except IndexError:
                curr_total_val_loss += val

        if epoch == 1 or curr_total_val_loss < best_total_val_loss:
            model.save_checkpoint('best')
            best_total_val_loss = curr_total_val_loss

    # SAVE TRAIN/VAL ERRORS
    errors = utils.tag_dict_keys(epoch_avg_train_loss, 'avg_train')
    errors.update(utils.tag_dict_keys(epoch_avg_val_loss, 'avg_val'))
    save_loss_csv(epochs, errors)


def test(model, test_data, which_epoch='best', label=None):
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=config.dataloader_workers,
                             pin_memory=config.use_cuda)

    model.load_checkpoint(which_epoch)
    model.set_mode('eval')
    config.random_crop = False

    if label is None:
        output_dir = os.path.join(config.results_dir,
                                  config.experiment_name,
                                  'test_{}'.format(which_epoch))
    else:
        output_dir = os.path.join(config.results_dir,
                                  config.experiment_name,
                                  '{}_test_{}'.format(label, which_epoch))
    os.makedirs(output_dir, exist_ok=True)

    for idx, (source, target) in enumerate(test_loader):
        model.set_data(source, target)
        model.test()
        output = model.get_images()

        for img_label, img in output.items():
            this_output_dir = os.path.join(output_dir, img_label)
            os.makedirs(this_output_dir, exist_ok=True)

            output_file = os.path.join(
                this_output_dir, '{:05}.png'.format(idx))
            print("Saving to {}".format(output_file))
            img.save(output_file)


def save_loss_csv(epochs, errors):
    save_dir = os.path.join(config.results_dir, config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'loss.csv')

    header = ['epoch'] + [key for key in errors]
    entries = [errors[key] for key in errors]
    entries = np.array(entries).T.tolist()

    print("Saving errors to {}".format(save_file))
    with open(save_file, 'wt') as file:
        file.write(','.join(header) + '\n')
        for epoch, entry in zip(epochs, entries):
            line = ','.join([str(int(epoch))] +
                            [str(val)for val in entry]) + '\n'
            file.write(line)
