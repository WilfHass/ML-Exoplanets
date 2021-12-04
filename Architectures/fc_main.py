import sys, os
import sys
import torch
import argparse
import numpy as np
sys.path.append('src') 
from load_data import *
from parameter import Parameter
from fc_net import FCNet
from helper_gen import make_parser, performance, optimize
from heatmap import *
from torch.utils.tensorboard import SummaryWriter

pwd = os.getcwd()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FC Main')
    parser.add_argument('--input', default ='src/torch_data',type=str, help="location for input file")
    parser.add_argument('--view', default='local', type=str,help="view")
    parser.add_argument('--param', default='param/fc_local.json',type=str,help="location of parameter file")
    parser.add_argument('--result', default='result/',type=str,help="location of results file")

   
    args = parser.parse_args()

    param_file = args.param

    params = Parameter(param_file, pwd)
    input_folder = args.input
    view = args.view
    result_file = args.result
    epoch_num = params.epoch
    lr = params.lr

    # Create directory and files for TensorBoard
    # res_path = "./runs/"
    run_ID = 'FC_test_local'  # Update the run_ID to see comparison of different runs
    tb_ID = "./runs/" + run_ID
    for i in range(1, 500):
        cur_name = ("./runs/" + run_ID + '_{}'.format(str(i)))
        if not os.path.isdir(cur_name):
            tb_ID = cur_name
            break

    print(tb_ID)
    writer = SummaryWriter(tb_ID)



    train_loader, test_loader = dataPrep(input_folder, params.trainbs,params.testbs)
    train_set = list(train_loader)
    test_set = list(test_loader)
    fc_net = FCNet(view)

    beta_l = 0.9
    beta_h = 0.999
    amsgrad = 0
    optim = torch.optim.Adam(fc_net.parameters(), lr=lr, betas=(beta_l, beta_h), amsgrad=amsgrad)

    # optim = torch.optim.SGD(model.parameters(),lr=lr, momentum=params.mom)

    loss_fn = torch.nn.BCELoss()

    for e in range(epoch_num):
        
        # Train the model
        fc_net.train()
        train_loss_val = 0
        for batch_idx, (data, label) in enumerate(train_loader):

            optim.zero_grad()
            outputs = fc_net(data)
            label = torch.reshape(label, (len(label), -1))
            loss = loss_fn(outputs, label)
            loss.backward()
            optim.step()
            loss_f_train = float(loss.item())
            train_loss_val += loss_f_train

        # Obtain performance metrics for training
        perf_list_train = performance(fc_net, train_set)

        # Test the model
        fc_net.eval()
        test_loss_val = 0
        with torch.no_grad():
            for batch_idx_test, (data_test, label_test) in enumerate(test_loader):
                outputs_test = fc_net(data_test)
                label_test = torch.reshape(label_test, (len(label_test), -1))
                loss_test = loss_fn(outputs_test, label_test)
                loss_f_test = float(loss_test.item())
                test_loss_val += loss_f_test

        # Obtain performance metrics for testing
        perf_list_test = performance(fc_net, test_set)

        train_loss_avg = train_loss_val/len(train_set)
        test_loss_avg = test_loss_val / len(test_set)

        if e % 1 == 0:
            print("Epoch [{}/{}] \t Loss: {}".format(e+1, epoch_num, train_loss_avg))

        # Add all metrics to TensorBoard
        writer.add_scalar('Training loss', train_loss_avg, float(e))
        writer.add_scalar('Training accuracy', float(perf_list_train[0]), float(e))
        writer.add_scalar('Training precision', float(perf_list_train[1]), float(e))
        writer.add_scalar('Training recall', float(perf_list_train[2]), float(e))
        writer.add_scalar('Training AUC', float(perf_list_train[3]), float(e))

        writer.add_scalar('Test loss', test_loss_avg, float(e))
        writer.add_scalar('Test accuracy', float(perf_list_test[0]), float(e))
        writer.add_scalar('Test precision', float(perf_list_test[1]), float(e))
        writer.add_scalar('Test recall', float(perf_list_test[2]), float(e))
        writer.add_scalar('Test AUC', float(perf_list_test[3]), float(e))



    # Important training/testing parameters
    tf_params = {
        "epochs": float(epoch_num),
        "lr": float(lr),
        "beta lower": beta_l,
        "beta upper": beta_h,
        "amsgrad": amsgrad,
        "epsilon": 0,
        "training bs": float(params.trainbs),
        "testing bs": float(params.testbs)
    }

    # Add last performance metrics to TensorBoard
    tf_metric = {
        "Training loss": train_loss_avg,
        "Training accuracy": float(perf_list_train[0]),
        "Training precision": float(perf_list_train[1]),
        "Training recall": float(perf_list_train[2]),
        "Training AUC": float(perf_list_train[3]),

        "Test loss": test_loss_avg,
        "Test accuracy": float(perf_list_test[0]),
        "Test precision": float(perf_list_test[1]),
        "Test recall": float(perf_list_test[2]),
        "Test AUC": float(perf_list_test[3])
    }

    writer.add_hparams(tf_params, tf_metric)

    writer.close()
    #optimize(fc_net, train_batchlists, params)

    # create_heatmap(fc_net, input_folder, view) # commented because of an error
    # perf_list = performance(fc_net, test_set) # Don't need this at the end anymore

    # To open tensorboard after a run:
    # tensorboard --logdir=runs