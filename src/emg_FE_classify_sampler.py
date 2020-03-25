# coding=utf-8
import numpy as np
import torch
from utils import core

class EMG_FE_Classify_Sampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''
    nSub = 10
    nFE = 11
    nSes = 25
    iSesTrain = range(0, 5)
    nWin = 41


    def __init__(self, option, index= None):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(EMG_FE_Classify_Sampler, self).__init__()
        # Todo: Possible query and support indexes
        # Todo: Confirm test subject
        self.index = index
        self.index_test_subject = option.test_subject_index
        self.sample_per_class = option.num_support_tr
        self.iterations = option.iterations
        self.nDomain = 5


    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        index = self.index
        iterations = self.iterations
        # cpi = self.classes_per_it



        # Todo: 다른 피험자로부터 train test 나눔
        sVal = core.getIdxExclude_of_inputIndex(range(0, self.nSub), [self.index_test_subject])
        sVal = np.random.permutation(sVal)[0]
        sTrain = core.getIdxExclude_of_inputIndex(range(0, self.nSub), [self.index_test_subject, sVal])
        sTrain = np.random.permutation(sTrain)[0]



        for it in range(iterations):
            # batch 초기화
            batch = torch.LongTensor(spc*self.nFE*2*2)

            # Todo: domain 랜덤으로 두개 선택 하기나누기
            temp = np.random.permutation(list(range(self.nDomain)))
            dSupport = temp[0]
            dQuery = temp[1]

            # Todo: training data
            query = []
            support = []
            for t in np.random.permutation(range(self.nFE)):
                a1 = core.ismember(index['s'], [sTrain])
                a2 = core.ismember(index['t'], [t])
                a3 = core.ismember(index['d'], [dQuery])
                a = [a1[k] and a2[k] and a3[k] for k in range(a1.__len__())]
                found = core.find(a, None)
                query.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])

                a1 = core.ismember(index['s'], [sTrain])
                a2 = core.ismember(index['t'], [t])
                a3 = core.ismember(index['d'], [dSupport])
                a = [a1[k] and a2[k] and a3[k] for k in range(a1.__len__())]
                found = core.find(a, None)
                support.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])

            # batch에 저장
            for t, c in enumerate(list(range(self.nFE))):
                s = slice(c * spc, (c + 1) * spc)
                batch[s] = query[t]

            for t, c in enumerate(list(range(self.nFE, self.nFE*2))):
                s = slice(c * spc, (c + 1) * spc)
                batch[s] = support[t]


            # Todo: validation data
            temp = np.random.permutation(list(range(self.nDomain)))
            dSupport = temp[0]
            dQuery = temp[1]
            query = []
            support = []
            for t in np.random.permutation(range(self.nFE)):
                a1 = core.ismember(index['s'], [sVal])
                a2 = core.ismember(index['t'], [t])
                a3 = core.ismember(index['d'], [dQuery])
                a = [a1[k] and a2[k] and a3[k] for k in range(a1.__len__())]
                found = core.find(a, None)
                query.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])

                a1 = core.ismember(index['s'], [sVal])
                a2 = core.ismember(index['t'], [t])
                a3 = core.ismember(index['d'], [dSupport])
                a = [a1[k] and a2[k] and a3[k] for k in range(a1.__len__())]
                found = core.find(a, None)
                support.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])

            # batch에 저장
            for t, c in enumerate(list(range(2*self.nFE, 3 * self.nFE))):
                s = slice(c * spc, (c + 1) * spc)
                batch[s] = query[t]

            for t, c in enumerate(list(range(3* self.nFE, 4* self.nFE))):
                s = slice(c * spc, (c + 1) * spc)
                batch[s] = support[t]

            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
