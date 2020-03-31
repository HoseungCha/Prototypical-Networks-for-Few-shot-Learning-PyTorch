# coding=utf-8
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pyriemann

from emg_sampler import EMG_sampler
from emgnet_loss import emg_loss as loss_fn
from emg_dataset import EMG_dataset
from emgnet import EMGnet
from parser_util import get_parser
from utils import core
from pyriemann.utils import mean
from pyriemann.utils import covariance
from pyriemann.utils import tangentspace
from tqdm import tqdm
from parser_util import get_parser
import pyriemann_torch

import itertools
import time


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


def init_emgnet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    return EMGnet().to(device)


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


def train(opt):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    best_state = True
    nSub = 10
    # sTest = int(opt.test_subject)

    # Load EMG dataset
    dataset = EMG_dataset(option=opt)

    # load EMG dataset labeling
    index = {}
    index['t'] = dataset.t
    index['s'] = dataset.s
    index['d'] = dataset.d

    # for each test subject, the validation scheme was conducted
    for sTest in range(2, nSub):
        print('=== sTest: {} ==='.format(sTest))
        # 모델 초기화
        model = init_emgnet(opt)
        optim = init_optim(opt, model)
        lr_scheduler = init_lr_scheduler(opt, optim)

        # Performance parameters initialization
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        test_loss = []
        test_acc = []
        bestTestAcc = []
        bestTestLoss = []
        best_acc = 0

        # test dataset loader and prepare Riemannian Feature
        testDataloader = torch.utils.data.DataLoader \
            (dataset, batch_sampler=EMG_sampler(option=opt, index=index, sExtract=sTest), shuffle=False)
        test_iter = iter(testDataloader)
        batch = next(test_iter)
        batch_x_test, batch_y_test = batch

        test_x_support, test_y_support = reimannian_feat_ext(opt,
                                   batch_x_test[:opt.classes_per_it_tr * opt.num_support_tr],
                                   batch_y_test[:opt.classes_per_it_tr * opt.num_support_tr])
        test_x, test_y = reimannian_feat_ext(opt,batch_x_test, batch_y_test)
        test_x_support, test_y_support = test_x_support.to(device), test_y_support.to(device)
        test_x, test_y = test_x.to(device), test_y.to(device)
        del batch_x_test, batch_y_test, batch, test_iter, testDataloader
        torch.cuda.empty_cache()

        # do epoch; for each epoch, the best model from validation data is going to be determined for the test data
        for epoch in range(opt.epochs):
            # set model name to save
            best_model_path = os.path.join(opt.experiment_root, 'best_model_sTest_{}_epoch_{}.pth'.format(sTest, epoch))
            last_model_path = os.path.join(opt.experiment_root, 'last_model_sTest_{}_epoch_{}.pth'.format(sTest, epoch))
            print('=== Epoch: {} ==='.format(epoch))
            time.sleep(0.01)

            # Load randomly selected subject Training/Validation Dataset except test subject
            trainValDataloader = torch.utils.data.DataLoader \
                (dataset, batch_sampler = EMG_sampler(option=opt, index=index, sExclude=sTest))
            tr_iter = iter(trainValDataloader)
            # do iteration
            for batch in tqdm(tr_iter):
                model.train()
                print('\n')
                batch_x, batch_y = batch

                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)

                # Monitoring gradients sine Riemannian Feature was extracted
                optim.zero_grad()

                # feature extraction
                x, y = reimannian_feat_ext(opt, batch_x[0:int(batch_x.shape[0]/2)], batch_y[0:int(batch_x.shape[0]/2)])
                # torch.unsqueeze(test_x_support, 1)
                x = torch.cat((x.to(device), test_x_support), 0)
                # x, y = x.to(device), y.to(device)

                model_output = model(torch.unsqueeze(x,1), )
                # model()


                # compute Loss and accuracies; please note that test_x_support data was included as query data
                # loss, acc, y_hat = loss_fn(torch.cat((model_output, model(torch.unsqueeze(test_x_support,1))), 0),
                #                     target=torch.cat((y, test_y_support), 0), n_support=opt.num_support_tr)
                loss, acc, y_hat = loss_fn(model_output,target=torch.cat((y, test_y_support), 0), n_support=opt.num_support_tr)
                print('Train Loss: {}, Train Acc: {}'.format(loss.item(), acc.item()))

                # Compute gradients and update model parameters
                loss.backward()
                optim.step()

                # Save results
                train_loss.append(loss.item())
                train_acc.append(acc.item())

                # Validation Dataset Evaluation
                model.eval()
                # validation features extraction
                x, y = reimannian_feat_ext(opt,
                                           batch_x[int(batch_x.shape[0] / 2):],
                                           batch_y[int(batch_x.shape[0] / 2):])
                x = torch.cat((x.to(device), test_x_support), 0)
                # x, y = x.to(device), y.to(device)

                # predict the validation data with test_x_support to find out the best model for the test subject
                model_output = model(torch.unsqueeze(x, 1))
                loss, acc, y_hat = loss_fn(torch.cat((model_output, model(torch.unsqueeze(test_x_support,1))), 0),
                                    target=torch.cat((y, test_y_support), 0), n_support=opt.num_support_tr)
                print('Val Loss: {}, Val Acc: {}'.format(loss.item(), acc.item()))

                # Save results
                val_loss.append(loss.item())
                val_acc.append(acc.item())

                # fixme: CHECK MEMORY OKAY
                # continue

                # Predict the Test subject data
                model.eval()
                model_output = model(torch.unsqueeze(test_x, 1))
                loss, acc, y_hat = loss_fn(model_output, target=test_y,
                                    n_support=opt.num_support_tr)

                print('Test Loss: {}, Test Acc: {}\n'.format(loss.item(), acc.item()))
                time.sleep(0.01)

                # Save results
                test_loss.append(loss.item())
                test_acc.append(acc.item())

            print('Epoch: {}  Finished ==='.format(epoch))


            print('Leaning Rate Update ==='.format(epoch))
            lr_scheduler.step() # for every epoch, learning rate is decreased by the value of gamma

            # average loss and accuracy was computed for this epoch ( average along the iterations)
            avg_loss = np.mean(val_loss[-opt.iterations:])
            avg_acc = np.mean(val_acc[-opt.iterations:])

            # if the validation performance is better than the previous one, update the best accuracy and save the model
            postfix = ' (Best with Test Acc: {})'.format(acc.item()) if avg_acc >= best_acc else '(Best Val Acc: {} with Test Acc: {})'.format(
                best_acc, bestTestAcc[-1])
            print('Avg Val Loss: {}, Avg Val Acc: {}\n{}'.format(
                avg_loss, avg_acc, postfix))

            if avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_y_hat = list(itertools.chain.from_iterable(y_hat.tolist()))
                best_state = model.state_dict()
                # model.eval()
                # model.load_state_dict(best_state)
                # model_output = model(torch.unsqueeze(test_x, 1))
                # loss, acc = loss_fn(model_output, target=test_y,
                #                     n_support=opt.num_support_tr)
                bestTestLoss.append(loss.item())
                bestTestAcc.append(acc.item())
                # # loss, acc = featExt_and_compLossAcc(opt, model, batch_x_test, batch_y_test)
                # print('bestTestLoss: {}, bestTestAcc: {}{}\n'.format(loss, acc, postfix))
                # time.sleep(0.01)
                # del best_state
                # torch.cuda.empty_cache()
        torch.save(model.state_dict(), last_model_path)
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                     'test_loss', 'test_acc', 'bestTestLoss', 'bestTestAcc','best_y_hat']:
            save_list_to_file(os.path.join(opt.experiment_root,
                                           name + '_sTest_{}.txt'.format(sTest)), locals()[name])
        # delete memory
        del x, y, batch, batch_x, batch_y, trainValDataloader, model_output, loss, acc, tr_iter, model, optim, lr_scheduler
        torch.cuda.empty_cache()








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
    model = init_fenet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

# def featExt_and_compLossAcc(opt, model, x_train, y_train):
def reimannian_feat_ext(opt, x_train, y_train):

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    # Todo: Riemannian Feature Extraction
    covMat = pyriemann_torch.covariances(x_train[:])
    # cov = covariance.covariances(np.swapaxes(x_train[:].cpu().numpy(),1, 2),estimator='cov')


    Cref = pyriemann_torch.mean_riemann(covMat[:opt.classes_per_it_tr * opt.num_support_tr])
    x_feat_train = torch.FloatTensor(tangentspace.tangent_space(cov, Cref))
    # Todo: Forward and Caculate Loss
    x, y = x_feat_train.to(device), y_train.to(device)
    # model_output = model(torch.unsqueeze(x,1))
    # loss, acc = loss_fn(model_output, target=y,
    #                     n_support=opt.num_support_tr)
    # return loss, acc

    return x, y

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


    res = train(opt=options)

    # best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    # print('Testing with last model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)
    #
    # model.load_state_dict(best_state)
    # print('Testing with best model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

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
