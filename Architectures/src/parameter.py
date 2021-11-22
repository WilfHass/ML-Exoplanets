import json

class Parameter():
    
    def __init__(self, param_file, pwd):

        with open(pwd + "/" + param_file) as json_file:
            data = json.load(json_file)

            p = data['data']
            self.trainbs =  int(p[0]['train_batchsize'])
            self.testbs = int(p[0]['test_batchsize'])
            self.epoch = int(p[0]['epoch'])

            p = data['optim']
            self.lr = float(p[0]['learning_rate'])
            self.tri = int(p[0]['training_iterations'])
            self.mom = float(p[0]['momentum'])
        