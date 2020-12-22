import math
import random
import torch
import torch.nn.functional as F

class MemoryItem():

    def __init__(self, img, target, filepath, scanner, current_grammatrix=None):
        self.img = img.detach().cpu()
        self.target = target.detach().cpu()
        self.filepath = filepath
        self.scanner = scanner
        self.traincounter = 0
        self.current_grammatrix = current_grammatrix

class DynamicMemory():

    def __init__(self, memorymaximum=256, balance_memory=True, gram_weights=None):
        self.memoryfull = False
        self.memorylist = []
        self.memorymaximum = memorymaximum
        self.gram_weights = gram_weights

        #this is for balancing in the binary classification case...
        self.balance_memory = balance_memory
        self.classcounter = {0: 0, 1: 0}

    def insert_element(self, item):
        if not self.memoryfull:
            self.memorylist.append(item)
            if len(self.memorylist) == self.memorymaximum:
                self.memoryfull = True
        else:
            assert(item.current_grammatrix is not None)
            insertidx = -1
            mingramloss = 1000
            for j, ci in enumerate(self.memorylist):
                l_sum = 0.0
                for i in range(len(item.current_grammatrix)):
                    l_sum += self.gram_weights[i] * F.mse_loss(
                        item.current_grammatrix[i], ci.current_grammatrix[i], reduction='mean')

                if l_sum < mingramloss:
                    mingramloss = l_sum
                    insertidx = j
            self.memorylist[insertidx] = item

    #forceditems are in the batch, the others are chosen randomly
    def get_training_batch(self, batchsize, randombatch=False, forceditems=None):
        batchsize = min(batchsize, len(self.memorylist))

        imgshape = self.memorylist[0].img.shape

        x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2]))
        y = torch.empty(size=(batchsize, 1))
        j = 0

        if forceditems is not None:
            for ci in forceditems:
                x[j] = ci.img
                y[j] = ci.label
                ci.traincounter += 1
                j += 1

            batchsize -= j

        if randombatch:
            random.shuffle(self.cachelist)
        else:
            self.cachelist.sort()

        if batchsize>0:
            for ci in self.cachelist[-batchsize:]:
                x[j] = ci.img
                y[j] = ci.label
                ci.traincounter += 1
                j += 1

        return x, y

    def __iter__(self):
        return self.cachelist.__iter__()