"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from datetime import datetime
from collections import defaultdict
import kaldiio
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    now=datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start creating dataset...", current_time)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
    A_prev_ID=""
    output={}
    now=datetime.now()
    current_time = now.strftime("%H:%M:%S")
    currdir=os.getcwd()
    print("Start generating fake_B Time =", current_time, currdir)
    END=len(dataset)
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        A_uttID="".join(data['A_uttid'])
        if (A_uttID != A_prev_ID):
            if (i > 0):
                mat=A_all.cpu().numpy().reshape((-1,40))
                output.update({A_prev_ID:mat})
                print("Write %s features %s to output dictionary" % (A_prev_ID, A_all.view(-1,40)))
            A_prev_ID=A_uttID
            A_all=model.fake_B.squeeze(0).squeeze(0)[5]
            print(i, model.fake_A)
        elif (i == END-1):
            mat=A_all.cpu().numpy().reshape((-1,40))
            output.update({A_uttID:mat})
            print("Write %s features %s to output dictionary" % (A_uttID, A_all.view(-1,40)))
        else:
            curr_tensor = model.fake_B.squeeze(0).squeeze(0)[5]
            A_all=torch.cat((A_all, curr_tensor), 0)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    feats_ark_path=os.path.join(currdir,'dt05_real_noisy_from_v12_res6b_v13','feats.ark')
    feats_scp_path=os.path.join(currdir,'dt05_real_noisy_from_v12_res6b_v13','feats.scp')
    kaldiio.save_ark(feats_ark_path, output, feats_scp_path)
    print("Write features to kaldi format %s %s" % (feats_ark_path, feats_scp_path))
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)
