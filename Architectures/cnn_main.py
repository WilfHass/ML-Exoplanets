import sys, os
import torch
import argparse
sys.path.append('src') 
from load_data import *
from parameter import Parameter
from cnn_net import CNNNet
from helper_gen import make_parser, performance, optimize
from heatmap import *

from torch.utils.tensorboard import SummaryWriter

pwd = os.getcwd()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Main')
    parser.add_argument('--input', default ='src/torch_data',type=str, help="location for input file")
    parser.add_argument('--view', default='local',type=str,help="view")
    parser.add_argument('--param', default='param/cnn_local.json',type=str,help="location of parameter file")
    parser.add_argument('--result', default='result/',type=str,help="location of parameter file")

   
    args = parser.parse_args()

    param_file = args.param

    params = Parameter(param_file, pwd)
    input_folder = args.input
    view = args.view
    result_file = args.result
    epoch_num = params.epoch

    train_loader, test_loader = dataPrep(input_folder, params.trainbs, params.testbs)
    cnn_net = CNNNet(view)
    beta_l = 0.9
    beta_h = 0.999
    amsgrad = 0
    optim = torch.optim.Adam(cnn_net.parameters(), lr=params.lr, betas=(beta_l, beta_h), amsgrad=amsgrad)
    #optim = torch.optim.Adam(cnn_net.parameters(), lr=params.lr, betas=(0.9, 0.999), amsgrad=False)

    # optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)

    test_set = list(test_loader)

    loss_fn = torch.nn.BCELoss()

    res_path = "./runs/"
    run_ID = 'cnn_local_test'  # Update the run_ID to see comparison of different runs
    tb_ID = res_path + run_ID
    for i in range(1, 500):
        cur_name = (res_path + run_ID + '_{}'.format(str(i)))
        if not os.path.isdir(cur_name):
            tb_ID = cur_name
            break

    print(tb_ID)

    
    writer = SummaryWriter(tb_ID)

    for e in range(epoch_num):
        
        # Train the model
        cnn_net.train()
        for batch_idx, (data, label) in enumerate(train_loader):

            optim.zero_grad()
            outputs = cnn_net(data)
            label = torch.reshape(label,(len(label),-1))
            loss = loss_fn(outputs, label)
            loss.backward()
            optim.step()
            loss_f = float(loss.item())
            

        if e % 1 == 0:
            print("Epoch [{}/{}] \t Loss: {}".format(e+1, epoch_num, loss.item()))

        perf_list = performance(cnn_net, test_set)

        writer.add_scalar('Training loss', loss_f, float(e))
        writer.add_scalar('Training accuracy', float(perf_list[0]), float(e))
        writer.add_scalar('Training precision', float(perf_list[1]), float(e))
        writer.add_scalar('Training recall', float(perf_list[2]), float(e))
        writer.add_scalar('Training AUC', float(perf_list[3]), float(e))

    tf_params = {
        "epochs": float(params.epoch),
        "lr": float(params.lr),
        "momentum": float(params.mom),
        "training bs": float(params.trainbs),
        "beta lower": beta_l,
        "beta upper": beta_h,
        "amsgrad": amsgrad
    }

    tf_metric = {
        "Training loss": loss_f,
        "Training accuracy": float(perf_list[0]),
        "Training precision": float(perf_list[1]),
        "Training recall": float(perf_list[2]),
        "Training AUC": float(perf_list[3])
    }
    print(tf_params)

    writer.add_hparams(tf_params, tf_metric)

    writer.close()
    print("Finished")
    # To open tensorboard, run in terminal:
    # tensorboard --logdir=runs
    #optimize(fc_net, train_batchlists, params)

    # perf_list = performance(fc_net, test_set)
    # [acc, prec, rec, AUC]
    print(perf_list)

    
    #optimize(cnn_net, train_batchlists, params)
    #test_set = list(test_loader)
    #create_heatmap(cnn_net, input_folder,view)
    #perf_list = performance(cnn_net, test_set)