import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self, view, device):
        '''
        Convolutional Neural Network
        size : size of data
        view : view of TCE data -> 'global' | 'local' | 'both'
        '''
        super(CNNNet, self).__init__()

        self.view = view
        self.device = device

        self.local_length = 201
        self.global_length = 2001


        ## Convolutional Columns

        # Only local view use these layers
        self.conv_in_local = nn.Conv1d(1, 16, 5, padding='same')  # (in channels, out channels, kernel size)
        self.conv_1_local = nn.Conv1d(16, 16, 5, padding='same')
        self.conv_2_local = nn.Conv1d(16, 32, 5, padding='same')
        self.conv_3_local = nn.Conv1d(32, 32, 5, padding='same')

        # Only global view uses these layers
        self.conv_in_global = nn.Conv1d(1, 16, 5, padding='same')
        self.conv_1_global = nn.Conv1d(16, 16, 5, padding='same')
        self.conv_2_global = nn.Conv1d(16, 32, 5, padding='same')
        self.conv_3_global = nn.Conv1d(32, 32, 5, padding='same')
        self.conv_4_global = nn.Conv1d(32, 64, 5, padding='same')
        self.conv_5_global = nn.Conv1d(64, 64, 5, padding='same')
        self.conv_6_global = nn.Conv1d(64, 128, 5, padding='same')
        self.conv_7_global = nn.Conv1d(128, 128, 5, padding='same')
        self.conv_8_global = nn.Conv1d(128, 256, 5, padding='same')
        self.conv_9_global = nn.Conv1d(256, 256, 5, padding='same')


        ## Maxpool layers
        # Local view
        self.maxpool_local = nn.MaxPool1d(7, stride=2)

        # Global view
        self.maxpool_global = nn.MaxPool1d(5, stride=2)

        ## Fully connected layers
        # Local view
        #self.fc_in_local = nn.Linear(32 * 5, 512, bias=True)  # output of conv1d is 32*5 ? --> only *5 once because 1d not 2d convolution?
        #self.fc_in_local = nn.Linear(1280, 512, bias=True)  # Where the heck does 1280 come from????
        self.fc_in_local = nn.Linear(1472, 512, bias=True)  # Where the heck does 1280 come from????

        # Global view
        #self.fc_in_global = nn.Linear(256 * 5, 512, bias=True)  # output of conv1d is 256*5 ? --> only *5 once because 1d not 2d convolution?
        #self.fc_in_global = nn.Linear(13056, 512, bias=True)    # Where in the heck does 13056 come from!?!?!?!?!
        self.fc_in_global = nn.Linear(15104, 512, bias=True)    # Where in the heck does 13056 come from!?!?!?!?!

        # Both global and local view
        #self.fc_in_both = nn.Linear(1280 + 13056, 512, bias=True)  # output of conv1d is 32*5 + 256*5 ? --> only *5 once because 1d not 2d convolution?
        self.fc_in_both = nn.Linear(1472 + 15104, 512, bias=True)

        # FC layers for all views
        self.fc_1 = nn.Linear(512, 512, bias=True)
        self.fc_2 = nn.Linear(512, 512 , bias=True)
        #self.fc_3 = nn.Linear(512, 512, bias=True)
        self.fc_out = nn.Linear(512, 1, bias=True)


    # Feed forward
    def forward(self, x):
        '''
        Feed forward propogation
        x : data input of batch size by 2201
        '''
        x = x.to(self.device)
        x_local = x[:, :201]
        x_global = x[:, 201:]

        x_local = torch.unsqueeze(x_local, dim=1)
        x_global = torch.unsqueeze(x_global, dim=1)

        #x = torch.unsqueeze(x, dim=1)
        #print(x.shape)

        if (self.view == "local"):
            #x = torch.unsqueeze(local_data, dim=1)
            #x = x.view(-1, 1, self.local_length)

            conv_output = self.forward_conv_local(x_local)
            #print(len(conv_output))    #???
            conv_output = torch.flatten(conv_output, 1)
            #print(conv_output.shape)
            h1 = F.relu(self.fc_in_local(conv_output))
            #print(h1.shape)

        elif (self.view == "global"):
            #x = torch.unsqueeze(global_data, dim=1)

            conv_output = self.forward_conv_global(x_global)
            conv_output = torch.flatten(conv_output, 1)
            h1 = F.relu(self.fc_in_global(conv_output))

        elif (self.view == "both"):
            local_conv_output = self.forward_conv_local(x_local)
            global_conv_output = self.forward_conv_global(x_global)

            local_conv_output = torch.flatten(local_conv_output, 1)
            global_conv_output = torch.flatten(global_conv_output, 1)

            # Concatenate tensors from outputs of two different convolutional columns?
            ## What is format of local and global view tensors???
            conv_output = torch.cat((local_conv_output, global_conv_output), dim=1)

            h1 = F.relu(self.fc_in_both(conv_output))

        h2 = F.relu(self.fc_1(h1))
        h3 = F.relu(self.fc_2(h2))
        #h4 = F.relu(self.fc_3(h3))
        y = torch.sigmoid(self.fc_out(h3))

        return y


    # Feed Forward for convolutional column for local view
    def forward_conv_local(self, x):
        x = F.relu(self.conv_in_local(x))
        x = F.relu(self.conv_1_local(x))
        x = self.maxpool_local(x)             
        x = F.relu(self.conv_2_local(x))
        x = F.relu(self.conv_3_local(x))
        x = self.maxpool_local(x)           
        
        return x

    # Feed Forward for convolutional column for global view
    def forward_conv_global(self, x):
        x = F.relu(self.conv_in_global(x))
        x = F.relu(self.conv_1_global(x))
        x = self.maxpool_global(x)             
        x = F.relu(self.conv_2_global(x))
        x = F.relu(self.conv_3_global(x))
        x = self.maxpool_global(x) 
        x = F.relu(self.conv_4_global(x))
        x = F.relu(self.conv_5_global(x))
        x = self.maxpool_global(x)
        x = F.relu(self.conv_6_global(x))
        x = F.relu(self.conv_7_global(x))
        x = self.maxpool_global(x)
        x = F.relu(self.conv_8_global(x))
        x = F.relu(self.conv_9_global(x))
        x = self.maxpool_global(x)        
        
        return x

    # Reset parameters if using net more than once
    def reset(self):
        self.conv_in_local.reset_parameters()
        self.conv_1_local.reset_parameters()
        self.conv_2_local.reset_parameters()
        self.conv_3_local.reset_parameters()

        self.conv_in_global.reset_parameters()
        self.conv_1_global.reset_parameters()
        self.conv_2_global.reset_parameters()
        self.conv_3_global.reset_parameters()
        self.conv_4_global.reset_parameters()
        self.conv_5_global.reset_parameters()
        self.conv_6_global.reset_parameters()
        self.conv_7_global.reset_parameters()
        self.conv_8_global.reset_parameters()
        self.conv_9_global.reset_parameters()

        self.fc_in_local.reset_parameters()
        self.fc_in_global.reset_parameters()
        self.fc_in_both.reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        #self.fc_3.reset_parameters()
        self.fc_out.reset_parameters()