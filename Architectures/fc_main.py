import sys, os
import sys
import torch

sys.path.append('src') 
from load_data import *
from parameter import Parameter
from fc_net import FCNet
from helper_gen import make_parser, performance, optimize
# torch -- tensorboard interface
from torch.utils.tensorboard import SummaryWriter

pwd = os.getcwd()
        
if __name__ == '__main__':

    args = make_parser()

    param_file = args.param

    params = Parameter(param_file, pwd)
    input_folder = args.input
    view = args.view
    result_file = args.result
    epoch_num = params.epoch

    train_loader, test_loader = dataPrep(input_folder, params.trainbs)
    fc_net = FCNet(view)
    beta_l = 0.9
    beta_h = 0.999
    amsgrad = 0
    optim = torch.optim.Adam(fc_net.parameters(), lr=params.lr, betas=(beta_l, beta_h), amsgrad=amsgrad)

    test_set = list(test_loader)

    # optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)

    loss_fn = torch.nn.BCELoss()

    res_path = "./runs/"
    run_ID = 'fc_both_test'  # Update the run_ID to see comparison of different runs
    tb_ID = res_path + run_ID
    for i in range(1, 500):
        cur_name = (res_path + run_ID + '_{}'.format(str(i)))
        if not os.path.isdir(cur_name):
            tb_ID = cur_name
            break

    print(tb_ID)

    
    writer = SummaryWriter(tb_ID)

    for e in range(1, epoch_num+1):
        
        # Train the model
        fc_net.train()
        for batch_idx, (data, label) in enumerate(train_loader):

            optim.zero_grad()
            outputs = fc_net(data)
            label = torch.reshape(label, (len(label), -1))
            loss = loss_fn(outputs, label)
            loss.backward()
            optim.step()
            loss_f = float(loss.item())

        if (e % 5 == 0) or (e == 1):
            print("Epoch [{}/{}] \t Loss: {}".format(e, epoch_num, loss_f))

        perf_list = performance(fc_net, test_set)

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
