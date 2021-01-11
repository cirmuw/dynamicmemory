import math
import random
import torch
import torch.nn.functional as F

class MemoryItem():

    def __init__(self, img, target, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.target = target
        self.filepath = filepath
        self.scanner = scanner
        self.traincounter = 0
        self.outlier_counter = 0
        self.current_grammatrix = current_grammatrix
        self.pseudo_domain = pseudo_domain

class DynamicMemory():

    def __init__(self, memorymaximum=256, balance_memory=True, gram_weights=None, base_transformer=None, base_if=None):
        self.memoryfull = False
        self.memorylist = []
        self.memorymaximum = memorymaximum
        self.gram_weights = gram_weights

        #this is for balancing in the binary classification case...
        self.balance_memory = balance_memory

        if base_if is not None:
            self.transformer = base_transformer
            self.isoforest = {0: base_if}
            self.pseudodomaincounter = {0: 0, 1: 0}
            self.outlier_memory = []
            self.outlier_epochs = 10
        else:
            self.transformer = None

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

        y = list()

        j = 0

        if forceditems is not None:
            for ci in forceditems:
                x[j] = ci.img
                y.append(ci.target)
                ci.traincounter += 1
                j += 1

            batchsize -= j

        if randombatch:
            random.shuffle(self.memorylist)
        else:
            self.memorylist.sort()

        if batchsize>0:
            for ci in self.memorylist[-batchsize:]:
                x[j] = ci.img
                y.append(ci.target)
                ci.traincounter += 1
                j += 1

        return x, y

    def counter_outlier_memory(self):
        for item in self.outlier_memory:
            item.outlier_counter += 1
            if item.outlier_counter > self.outlier_epochs:
                self.outlier_memory.remove(item)

    def __iter__(self):
        return self.memorylist.__iter__()