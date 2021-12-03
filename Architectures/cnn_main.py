import sys, os, json
import torch
import argparse
from datetime import datetime

sys.path.append('src') 
from load_data import *
#from parameter import Parameter
from cnn_net import CNNNet
from helper_gen import performance
from heatmap import *

from torch.utils.tensorboard import SummaryWriter

pwd = os.getcwd()
        
if __name__ == '__main__':
    # Print start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    
    parser = argparse.ArgumentParser(description='CNN Main')
    parser.add_argument('--input', default ='src/torch_data',type=str, help="location for input file")
    parser.add_argument('--view', default='local',type=str,help="view")
    parser.add_argument('--param', default='cnn_local.json',type=str,help="location of parameter file")
    #parser.add_argument('--result', default='result/',type=str,help="location of results")
    parser.add_argument('-v', default=1 , help="Verbosity (0 = no command line output, 1 = print training loss")
    args = parser.parse_args()

    # Check if view is valid
    if (args.view != 'local' and args.view != 'global' and args.view != 'both'):
        print("Error: view needs to be local, global or both")
        sys.exit(1)
    else:
        view = args.view

    input_folder = args.input
    #result_file = args.result

    # Get parameters from json parameter file
    #params = Parameter(param_file, pwd)
    param_file = args.param
    with open(os.path.join('param', param_file)) as paramfile:
        params = json.load(paramfile)

    lr = params['optim']['learning rate']
    beta_l = params['optim']['beta 1']
    beta_h = params['optim']['beta 2']
    epsilon = params['optim']['epsilon']
    amsgrad = params['optim']['amsgrad']
    wd = params['optim']['weight decay']

    num_epochs = params['training']['num epochs']
    trainbs = params['training']['batch size']
    testbs = params['testing']['batch size']

    res_path = params['output']['results path']
    run_ID = params['output']['results name']


    # Create directory and files for TensorBoard
    #res_path = "./runs/"
    #run_ID = 'cnn_local_test'  # Update the run_ID to see comparison of different runs
    tb_ID = res_path + run_ID
    for i in range(1, 500):
        cur_name = (res_path + run_ID + '_{}'.format(str(i)))
        if not os.path.isdir(cur_name):
            tb_ID = cur_name
            break

    print(tb_ID)
    writer = SummaryWriter(tb_ID)

    # Training and Testing data
    #train_loader, test_loader = dataPrep(input_folder, params.trainbs, params.testbs)
    train_loader, test_loader = dataPrep(input_folder, trainbs, testbs)
    train_set = list(train_loader)
    test_set = list(test_loader)

    # Model, optimizer and loss function
    cnn_net = CNNNet(view)
    optim = torch.optim.Adam(cnn_net.parameters(), lr=lr, betas=(beta_l, beta_h), eps=epsilon, weight_decay=wd, amsgrad=amsgrad)
    loss_fn = torch.nn.BCELoss()


    for e in range(num_epochs):
        
        # Train the model
        cnn_net.train()
        train_loss_val = 0
        for batch_idx_train, (data_train, label_train) in enumerate(train_loader):
            optim.zero_grad()
            outputs_train = cnn_net(data_train)
            label_train = torch.reshape(label_train,(len(label_train),-1))
            loss_train = loss_fn(outputs_train, label_train)
            loss_train.backward()
            optim.step()
            loss_f_train = float(loss_train.item())
            train_loss_val += loss_f_train

        # Obtain performance metrics for training
        perf_list_train = performance(cnn_net, train_set)


        # Test the model
        cnn_net.eval()
        test_loss_val = 0
        with torch.no_grad():
            for batch_idx_test, (data_test, label_test) in enumerate(test_loader):
                outputs_test = cnn_net(data_test)
                label_test = torch.reshape(label_test,(len(label_test),-1))
                loss_test = loss_fn(outputs_test, label_test)
                loss_f_test = float(loss_test.item())
                test_loss_val += loss_f_test
        
        # Obtain performance metrics for testing
        perf_list_test = performance(cnn_net, test_set)

        # Print training loss
        if e % 1 == 0:
            print("Epoch [{}/{}] \t Train Loss: {}".format(e+1, num_epochs, train_loss_val/len(train_set)))

        # Add all metrics to TensorBoard
        writer.add_scalar('Training loss', train_loss_val, float(e))
        writer.add_scalar('Training accuracy', float(perf_list_train[0]), float(e))
        writer.add_scalar('Training precision', float(perf_list_train[1]), float(e))
        writer.add_scalar('Training recall', float(perf_list_train[2]), float(e))
        writer.add_scalar('Training AUC', float(perf_list_train[3]), float(e))

        writer.add_scalar('Test loss', test_loss_val, float(e))
        writer.add_scalar('Test accuracy', float(perf_list_test[0]), float(e))
        writer.add_scalar('Test precision', float(perf_list_test[1]), float(e))
        writer.add_scalar('Test recall', float(perf_list_test[2]), float(e))
        writer.add_scalar('Test AUC', float(perf_list_test[3]), float(e))


    # Important training/testing parameters
    tf_params = {
        "epochs": float(num_epochs),
        "lr": float(lr),
        "beta lower": beta_l,
        "beta upper": beta_h,
        "amsgrad": amsgrad,
        "epsilon": epsilon,
        "training bs": float(trainbs),
        "testing bs": float(testbs)
    }

    # Add last performance metrics to TensorBoard
    tf_metric = {
        "Training loss": train_loss_val,
        "Training accuracy": float(perf_list_train[0]),
        "Training precision": float(perf_list_train[1]),
        "Training recall": float(perf_list_train[2]),
        "Training AUC": float(perf_list_train[3]),
        "Test loss": train_loss_val,
        "Test accuracy": float(perf_list_test[0]),
        "Test precision": float(perf_list_test[1]),
        "Test recall": float(perf_list_test[2]),
        "Test AUC": float(perf_list_test[3])
    }

    print()
    print("TensorBoard Params:")
    print(tf_params)
    print()

    writer.add_hparams(tf_params, tf_metric)
    writer.close()
    # To open tensorboard, run in terminal:
    # tensorboard --logdir=runs
    #optimize(fc_net, train_batchlists, params)

    # Print final performance metrics
    # [acc, prec, rec, AUC]
    print("Training:")
    print("acc \t prec \t rec \t AUC")
    print(perf_list_train)
    print()
    print("Testing:")
    print("acc \t prec \t rec \t AUC")
    print(perf_list_test)
    print()

    # Print end time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

    print("Finished")

    # Create heatmap
    #optimize(cnn_net, train_batchlists, params)
    #test_set = list(test_loader)
    create_heatmap(cnn_net, input_folder, view)