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

    def __init__(self, option, dataset = None):
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

        index_sub_test = 9

        dataset.dataset


        self.labels = torch.tensor(labels)
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.ses = torch.LongTensor(ses)
        self.win = torch.LongTensor(win)
        self.sub = torch.LongTensor(sub)
        test_subject_index = option.test_subject_index

        # core.ismember(range(0, self.nSub), [iSubTrain, option.test_subject_index]), None)
        val_subject_index = core.getIdxExclude_of_inputIndex(range(0,self.nSub), [test_subject_index])

        self.sub.eq(test_subject_index).nonzero()

        query_idxs = torch.stack(list(map(lambda c: self.ses.eq(c).nonzero(), torch.arange(0,5)))).view(-1)
        query_idxs = core.find_multiple_index(self.ses.eq, torch.arange(0,5))

        core.find(torch.arange(0,5), 5)
        self.ses.eq([1,2,3,4,5]).nonzero()


        for i in range(self.nSub):
            for j in range(self.nFE):
                for k in range(self.nSes):
                    for l in range(self.nWin):
                        index = i * (self.nFE * self.nSes * self.nWin) +\
                                j * (self.nSes * self.nWin)\
                                + k * (self.nWin) + l
                        print(index)


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
