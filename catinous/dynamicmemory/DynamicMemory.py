import math
import random
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection
import collections
import numpy as np

class MemoryItem():

    def __init__(self, img, target, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.target = target.detach().cpu()
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
            self.isoforests = {0: base_if}
            self.outlier_memory = []
            self.outlier_epochs = 10
            self.max_per_domain = memorymaximum
            self.pseudo_detection = True
        else:
            self.transformer = None

    def insert_element(self, item):
        domain = -1
        if self.transformer is not None:
            item.current_grammatrix = self.transformer.transform(item.current_grammatrix.reshape(1, -1))
            item.current_grammatrix = item.current_grammatrix[0] #TODO: debug this!

            domain = self.check_pseudodomain(item.current_grammatrix)
            item.pseudo_domain = domain

        if domain == -1:
            # insert into outlier memory
            # check outlier memory for new clusters
            self.outlier_memory.append(item)
        else:
            if not self.memoryfull:
                self.memorylist.append(item)
                if len(self.memorylist) == self.memorymaximum:
                    self.memoryfull = True
            else:
                assert(item.current_grammatrix is not None)
                insertidx = -1
                mingramloss = 1000
                for j, ci in enumerate(self.memorylist):
                    if self.pseudo_detection and ci.pseudo_domain==domain:
                        l_sum = 0.0
                        for i in range(len(item.current_grammatrix)):
                            l_sum += self.gram_weights[i] * F.mse_loss(
                                item.current_grammatrix[i], ci.current_grammatrix[i], reduction='mean')

                        if l_sum < mingramloss:
                            mingramloss = l_sum
                            insertidx = j
                self.memorylist[insertidx] = item

    def find_insert_position(self):
        for idx, item in enumerate(self.memorylist):
            if item.deleteflag:
                return idx
        return -1

    def flag_items_for_deletion(self):
        for k, v in self.domaincomplete.items():
            domain_count = len(self.get_domainitems(k))
            if domain_count > self.max_per_domain:
                todelete = domain_count - self.max_per_domain
                for item in self.memorylist:
                    if todelete > 0:
                        if item.pseudo_domain == k:
                            if not item.deleteflag:
                                item.deleteflag = True

                            todelete -= 1

    def check_outlier_memory(self, model):
        if len(self.outlier_memory)>5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]

            distances = squareform(pdist(outlier_grams))
            print('check outlier memory distances', distances, sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5])
            if sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5]<0.20: #TODO: this is an arbritary threshold

                clf = IsolationForest(n_estimators=5, random_state=self.seed, warm_start=True, contamination=0.10).fit(
                    outlier_grams)

                new_domain_label = len(self.isoforests)
                self.max_per_domain = int(self.memorymaximum/(new_domain_label+1))

                self.flag_items_for_deletion()

                to_delete = []
                for k, p in enumerate(clf.predict(outlier_grams)):
                        if p == 1:
                            idx = self.find_insert_position()
                            if idx != -1:
                                elem = self.outlier_memory[k]
                                elem.pseudo_domain = new_domain_label
                                self.memorylist[idx] = elem
                                self.domaincounter[new_domain_label] += 1
                                to_delete.append(self.outlier_memory[k])
                                self.labeling_counter += 1
                                detection, wrong_detection = model.check_detection(elem.img, elem.target)
                                self.domainPerf[new_domain_label].append(detection)
                                self.domainPerf_wrong[new_domain_label].append(wrong_detection)
                for elem in to_delete:
                    self.outlier_memory.remove(elem)

                self.isoforests[new_domain_label] = clf

                for elem in self.get_domainitems(new_domain_label):
                    print('found new domain', new_domain_label, elem.scanner)


    def check_pseudodomain(self, grammatrix):
        max_pred = 0
        current_domain = -1

        for j, clf in self.isoforests.items():
            current_pred = clf.decision_function(grammatrix.reshape(1, -1))
            if current_pred > max_pred:
                current_domain = j
                max_pred = current_pred

        return current_domain

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