# coding=utf-8
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pyriemann

from emg_FE_classify_sampler import EMG_FE_Classify_Sampler

# from prototypical_loss import prototypical_loss as loss_fn
from fenet_loss import fe_loss as loss_fn

from EMG_FE_dataset import EMG_dataset
from protonet import ProtoNet
from fenet import FeNet
from parser_util import get_parser
from utils import core
from pyriemann.utils import mean
from pyriemann.utils import covariance
from pyriemann.utils import tangentspace
from tqdm import tqdm
import itertools



def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt):

    dataset = EMG_dataset(option=opt)

    n_classes = len(np.unique(dataset.t))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset



def init_sampler(opt, dataset, sTest):
    index = {}
    index['t'] = dataset.t
    index['s'] = dataset.s
    index['d'] = dataset.d

    return EMG_FE_Classify_Sampler(option=opt, index = index, sTest = sTest)


def init_dataloader(opt, sTest):
    dataset = init_dataset(opt)
    sampler = init_sampler(opt, dataset, sTest)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    if opt.dataset_type == 'EMG_dataset':
        model = FeNet().to(device)
    else:
        model = ProtoNet().to(device)

    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, model, optim, lr_scheduler):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    best_state = True
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    nSub = 10
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    dataset = EMG_dataset(option=opt)
    index = {}
    index['t'] = dataset.t
    index['s'] = dataset.s
    index['d'] = dataset.d

    for sTest in range(nSub):
        print('=== sTest: {} ==='.format(sTest))
        # training dataset loader
        trainValDataloader = torch.utils.data.DataLoader\
            (dataset, batch_sampler= EMG_FE_Classify_Sampler(option=opt, index=index, sExclude=sTest))

        # test dataset loader
        testDataloader = torch.utils.data.DataLoader \
            (dataset, batch_sampler=EMG_FE_Classify_Sampler(option=opt, index=index, sExtract=sTest))
        test_iter = iter(testDataloader)
        batch = next(test_iter)
        batch_x_test, batch_y_test = batch

        tr_iter = iter(trainValDataloader)


        model.train()
        for epoch in range(opt.epochs):
            print('=== Epoch: {} ==='.format(epoch))
            for batch in tqdm(tr_iter):
                print('\n')
                optim.zero_grad()
                batch_x, batch_y = batch

                # Todo: feature extraction and compute Loss and accuacy
                loss, acc = featExt_and_compLossAcc\
                    (opt, model,batch_x[0:int(batch_x.shape[0]/2)],batch_y[0:int(batch_x.shape[0]/2)])
                print('Train Loss: {}, Train Acc: {}'.format(loss, acc))

                # Todo: Prepare gradients, Update Parameters, and Evaluation
                loss.backward()
                optim.step()

                # Todo: Save results
                train_loss.append(loss.item())
                train_acc.append(acc.item())

                # Todo: feature extraction and compute Loss and Accuracy in Test dataset
                loss, acc = featExt_and_compLossAcc\
                    (opt, model,batch_x[int(batch_x.shape[0] / 2):],batch_y[int(batch_x.shape[0] / 2):])
                print('Val Loss: {}, Val Acc: {}'.format(loss, acc))

                # Todo: Save results
                val_loss.append(loss.item())
                val_acc.append(acc.item())

                # Todo: Get Test Accuracy
                loss, acc = featExt_and_compLossAcc(opt, model, batch_x_test, batch_y_test)
                print('Test Loss: {}, Test Acc: {}'.format(loss, acc))
            print('=== Epoch: {}  Finished ==='.format(epoch))
            lr_scheduler.step()
            print('=== Leaning Rate Update ==='.format(epoch))

            avg_loss = np.mean(val_loss[-opt.iterations:])
            avg_acc = np.mean(val_acc[-opt.iterations:])
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
            print('Test Loss: {}, Test Acc: {}{}'.format(
                loss, acc))

            if avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.state_dict()

        torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

def featExt_and_compLossAcc(opt, model, x_train, y_train):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    # Todo: Riemannian Feature Extraction
    cov = covariance.covariances(np.swapaxes(x_train[:].cpu().numpy(),1, 2),estimator='cov')
    Cref = mean.mean_riemann(cov[:opt.classes_per_it_tr * opt.num_support_tr])
    x_feat_train = torch.FloatTensor(tangentspace.tangent_space(cov, Cref))
    # Todo: Forward and Caculate Loss
    x, y = x_feat_train.to(device), y_train.to(device)
    model_output = model(torch.unsqueeze(x,1))
    loss, acc = loss_fn(model_output, target=y,
                        n_support=opt.num_support_tr)
    return loss, acc

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    # tr_dataloader = tr_dataloader,

    # tr_dataloader = init_dataloader(options, 'train')
    # test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)

    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()
