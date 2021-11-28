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
res_path = "./runs/"
run_ID = 'fc_test_4_both' # Update the run_ID to see comparison of different runs
writer = SummaryWriter(res_path + run_ID)

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
    optim = torch.optim.Adam(fc_net.parameters(), lr=params.lr, betas=(0.9, 0.99), amsgrad=False)

    test_set = list(test_loader)

    # optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)

    loss_fn = torch.nn.BCELoss()

    for e in range(epoch_num):
        
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

        if e % 10 == 0:
            print("Epoch [{}/{}] \t Loss: {}".format(e, epoch_num, loss_f))

        perf_list = performance(fc_net, test_set)
        writer.add_scalar('Training loss', loss_f, float(e))
        writer.add_scalar('Training accuracy', float(perf_list[0]), float(e))
        writer.add_scalar('Training precision', float(perf_list[1]), float(e))
        writer.add_scalar('Training recall', float(perf_list[2]), float(e))
        writer.add_scalar('Training AUC', float(perf_list[3]), float(e))

    print("Finished")
    # To open tensorboard, run in terminal:
    # tensorboard --logdir=runs
    #optimize(fc_net, train_batchlists, params)

    # perf_list = performance(fc_net, test_set)
    # [acc, prec, rec, AUC]
    print(perf_list)