# coding=utf-8
import numpy as np
import torch
import matplotlib
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
        # index[0] = dataset.t
        # index[1] = dataset.s
        # index[2] = dataset.d

        # Todo: Possible query and support indexes
        query = []
        support = []

        # Todo: Confirm test subject
        index_test_subject = option.test_subject_index
        # Todo: Get the validation subjects (cross-validation -> one batch)
        # index_val_subject = core.getIdxExclude_of_inputIndex(range(0,self.nSub), [index_test_subject])
        # Todo: prepare leave-subject-out cross-validation Loop
        for sVal in core.getIdxExclude_of_inputIndex(range(0,self.nSub), [index_test_subject]):
            for s in core.getIdxExclude_of_inputIndex(range(0,self.nSub), [index_test_subject, sVal]):
                for t in range(0, self.nFE):
                    for dSupport in range(0, 5):
                        for d in core.getIdxExclude_of_inputIndex(range(0,5), [dSupport]):
                            a1 = core.ismember(index['s'], [s])
                            a2 = core.ismember(index['t'], [t])
                            a3 = core.ismember(index['d'], [d])
                            a = [a1[k] and a2[k] and a3[k] for k in range(1, a1.__len__())]
                            found = core.find(a,None)
                            query.append(torch.randperm(found.__len__())[:50])

                    a1 = core.ismember(index['s'], [s])
                    a2 = core.ismember(index['t'], [t])
                    a3 = core.ismember(index['d'], [dSupport])
                    a = [a1[k] and a2[k] and a3[k] for k in range(1, a1.__len__())]
                    found = core.find(a, None)
                    support.append(torch.randperm(found.__len__())[:200])

        a = 1;

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                # label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                label_idx = int(c)
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
