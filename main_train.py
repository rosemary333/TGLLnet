import torch, time, os
import models, utils
from torch import nn, optim
from dataset import load_datasets
from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(best_auc, model, optimizer, epoch):
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # torch.save({
    #     'model_state_dict': model_state_dict,
    #     'global_epoch': epoch,
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'best_auc': best_auc,
    # }, os.path.join('checkpoints', config.model_name + '_' + config.experiment + '_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, train_dataloader):
    model.train()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    adj = 0
    for inputs, target in train_dataloader:

        inputs = inputs + torch.randn_like(inputs) * 0.1
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forwards
        if 'Mymodel' in config.model_name:
            output,adj = model(inputs)
        else:
            output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1

        output = torch.sigmoid(output)
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    auc = roc_auc_score(targets, outputs)
    TPR = utils.compute_TPR(targets, outputs)
    acc = utils.compute_ACC(targets, outputs)
    f1 = utils.compute_F1(targets, outputs)
    print('train_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f,    acc: %.4f, f1: %.4f' % (loss_meter / it_count, auc, TPR, acc, f1))
    return loss_meter / it_count, auc, TPR, acc, f1, adj


# val and test
def test_epoch(model, criterion, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    adj = 0
    with torch.no_grad():
        for inputs, target in val_dataloader:

            # inputs = inputs + torch.randn_like(inputs) * 0.1
            inputs = inputs.to(device)
            target = target.to(device)
            if 'Mymodel' in config.model_name:
                output,adj = model(inputs)
            else:
                output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1

            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        auc = roc_auc_score(targets, outputs)
        TPR = utils.compute_TPR(targets, outputs)
        acc = utils.compute_ACC(targets, outputs)
        f1 = utils.compute_F1(targets, outputs)

    print('test_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f, acc: %.4f, f1: %.4f' % (loss_meter / it_count, auc, TPR, acc, f1))
    return loss_meter / it_count, auc, TPR, acc, f1


def train(config=config):
    # seed
    setup_seed(config.seed)
    # print('torch.cuda.is_available:', torch.cuda.is_available())

    # datasets
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
    )
    
    # mode
    model = getattr(models, config.model_name)(num_classes=num_classes)
    print("*"*100)
    print('model_name:{}, dataset:{}, experiment:{}, num_classes={}'.format(config.model_name, config.datafolder.split("/")[-2], config.experiment, num_classes))
    model = model.to(device)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # =========>train<=========
    bauc, btpr, bacc, bf1, badj = 0, 0, 0, 0, 0
    for epoch in range(1, config.max_epoch + 1):
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size,
                                                                             config.lr))

        since = time.time()
        train_loss, train_auc, train_TPR, train_acc, train_f1, adj = train_epoch(model, optimizer, criterion,
                                                       train_dataloader)

        val_loss, val_auc, val_TPR, val_acc, val_f1 = test_epoch(model, criterion, val_dataloader)

        loss, auc, TPR, acc, f1 = test_epoch(model, criterion, test_dataloader)
        
        # print(adj)
        if (bauc + bf1) < (auc + f1):
            bauc, btpr, bacc, bf1 = auc, TPR, acc, f1
            save_checkpoint(bauc, model, optimizer, epoch)
            badj = adj
        print()
        if epoch == config.max_epoch:
            print(badj)
            print('btest_macro_auc: %.2f,  TPR: %.2f, acc: %.2f, f1: %.2f\n' % (bauc*100, btpr*100, bacc*100, bf1*100))

            # result_list = [[bauc, btpr, bacc, bf1]]
            # columns = ['', '', '', '', '']

            # dt = pd.DataFrame(result_list, columns=columns)

            # if config.model_name == 'gcn_ablation':
            #     dt.to_csv('result/csv/'+config.model_name +'_' +ablation + '_' + config.experiment + '_' +'result.csv', mode='a')
            # else:
            #     dt.to_csv('result/csv/'+config.model_name +'_' + config.experiment + '_' +'result.csv', mode='a')
            
        if epoch==1:
            print('time:%s\n' % (utils.print_time_cost(since)))


if __name__ == '__main__':
    # train()
    models_list = ['Mymodel'] # MiniRocket  'fcn_wang', 'resnet1d_wang',  'lstm',  'lstm_bidir', 'inceptiontime','xresnet1d101', 'vit', 'mobilenetv3_small','dccacb', 'ATI_CNN','MVMnet','MRMnet','Mymodel','Mymodel_ablation'
    for m in models_list:
        config.model_name=m
        for index in range(1):
            # ptb-xl
            config.datafolder = '/data/yuanxiaoyan/ecg_process/data/ptbxl/'
            for exp in ['exp0','exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']: #
                if exp == 'exp0':
                    config.seed = 10
                elif exp == 'exp1':
                    config.seed = 20
                elif exp == 'exp1.1':
                    config.seed = 20
                elif exp == 'exp1.1.1':
                    config.seed = 20
                elif exp == 'exp2':
                    config.seed = 7
                elif exp == 'exp3':
                    config.seed = 10
                config.experiment = exp
                train(config)
            # cpsc2018
            config.datafolder = '/data/yuanxiaoyan/ecg_process/data/ICBEB/'
            config.experiment = 'cpsc'
            config.seed = 7
            train(config)

    # config.datafolder = '../data/hf/'
    # config.experiment = 'hf'
    # config.seed = 9
    # train(config)
