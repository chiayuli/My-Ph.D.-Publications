"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training audios = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        total_DA1_loss=0
        total_DA2_loss=0
        total_DA3_loss=0
        total_DB_loss=0
        total_GA_loss=0
        total_GB_loss=0
        total_cycleA_loss=0
        total_cycleB_loss=0
        total_idtA_loss=0
        total_idtB_loss=0
        total_loss=0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            losses = model.get_current_losses()
            num_samples=data['A'].size()[0]
            total_DA1_loss+=(losses['D_A1']/float(num_samples))
            total_DA2_loss+=(losses['D_A2']/float(num_samples))
            total_DA3_loss+=(losses['D_A3']/float(num_samples))
            total_DB_loss+=(losses['D_B']/float(num_samples))
            total_GA_loss+=(losses['G_A']/float(num_samples))
            total_GB_loss+=(losses['G_B']/float(num_samples))
            total_cycleA_loss+=(losses['cycle_A']/float(num_samples))
            total_cycleB_loss+=(losses['cycle_B']/float(num_samples))
            total_idtA_loss+=(losses['idt_A']/float(num_samples))
            total_idtB_loss+=(losses['idt_B']/float(num_samples))
            cyclegan_loss=total_DA1_loss+total_DA2_loss+total_DA3_loss+total_DB_loss+\
                total_GA_loss+total_GB_loss+\
                total_cycleA_loss+ total_cycleB_loss+total_idtA_loss+total_idtB_loss
            total_loss+=(cyclegan_loss/float(num_samples))
        print("### real_B ###")
        print(model.real_B)
        print("### fake_B ###")
        print(model.fake_B)
        print("Epoch %d total loss: %.5f, DA1: %.5f, DA2: %.5f, DA3: %.5f, DB: %.5f, GA: %.5f, GB: %.5f " % (epoch, total_loss, total_DA1_loss, total_DA2_loss, total_DA3_loss, total_DB_loss, total_GA_loss, total_GB_loss))
        print("    cycle_A: %.5f, cycle_B: %.5f, idt_A: %.5f, idt_B: %.5f " % (total_cycleA_loss, total_cycleB_loss, total_idtA_loss, total_idtB_loss))

        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
