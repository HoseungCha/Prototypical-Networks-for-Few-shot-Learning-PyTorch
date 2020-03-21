import matplotlib
import matplotlib.pyplot as plt
import torch



def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]

def find(listVar,sample2find):
    # listVar = [1, 2, 3, 2]
    # sample2find = 2
    return [i for i, x in enumerate(listVar) if x == sample2find]

def find_multiple_index(a,b):
    return torch.stack(list(map(lambda c: a.eq(c).nonzero(), b))).view(-1)

def getIdxExclude_of_inputIndex(x,y):
    return find(ismember(x, y), None)

def getIdx_intersection(t1,t2): #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
    # t1 = torch.tensor([1, 9, 12, 5, 24])
    # t2 = torch.tensor([1, 24])

    # Create a tensor to compare all values at once
    compareview = t2.repeat(t1.shape[0], 1).T

    # Intersection
    # print(t1[(compareview == t1).T.sum(1) == 1])
    # # Non Intersection
    # print(t1[(compareview != t1).T.prod(1) == 1])
    return t1[(compareview == t1).T.sum(1) == 1], t1[(compareview != t1).T.prod(1) == 1]

