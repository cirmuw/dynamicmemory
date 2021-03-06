import math
import random
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest
import numpy as np


class MemoryItem():
    """
        Memory Item holding all attributes needed for a dynamic memory
    """

    def __init__(self, img, target, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        """
                        Initialization of a memory item.

                        :param img (tensor): image tensor stored in the item
                        :param target (tensor): target for the task
                        :param filepath (str): path to the file from which the image was read
                        :param scanner (str): scanner with which the image was acquired
                        :param current_grammatrix (np.array): calculated gram matrix for the image
                        :param pseudo_domain (int): pseudo domain the item was assigned to
        """

        self.img = img
        self.target = target
        self.filepath = filepath
        self.scanner = scanner
        self.traincounter = 0
        self.outlier_counter = 0
        self.current_grammatrix = current_grammatrix
        self.pseudo_domain = pseudo_domain
        self.deleteflag = False


class DynamicMemory():
    """
        Dynamic Memory, handling insertion of items based on style of the images,
        optionally balances the memory by using a pseudo domain detection approach.
    """

    def __init__(self, memorymaximum=256, balance_memory=True, base_transformer=None, base_if=None,
                 seed=None):
        """
            Initialization of the dynamic memory.

            :param memorymaximum (int): number of elements that can be stored to memory
            :param balance_memory (bool): whether or not to use pseudo domains to balance the memory
            :param base_transformer: optional transformer applied to gram matrices of memory items
            :param base_if: Isolation forest to find outliers for the base style
            :param seed (int): random seed to ensure reproducibility
        """


        self.memoryfull = False
        self.memorylist = []
        self.memorymaximum = memorymaximum
        self.balance_memory = balance_memory

        if base_if is not None:
            self.transformer = base_transformer
            self.isoforests = {0: base_if}
            self.outlier_memory = []
            self.outlier_epochs = 10
            self.max_per_domain = memorymaximum
            self.pseudo_detection = True
            self.domaincounter = {0: 0}
        else:
            self.transformer = None
            self.pseudo_detection = False

        self.seed = seed

    def insert_element(self, item):
        """
            Function to insert an element into the memory.
            Takes care of transforming the gram matrix, pseudo domain detection, and finding the image that should be replaced

            :param item (MemoryItem): item to be inserted
        """
        domain = -1
        if self.transformer is not None:
            item.current_grammatrix = np.hstack([gm.flatten() for gm in item.current_grammatrix])
            item.current_grammatrix = self.transformer.transform(item.current_grammatrix.reshape(1, -1))
            item.current_grammatrix = item.current_grammatrix[0]

            domain = self.check_pseudodomain(item.current_grammatrix)
            item.pseudo_domain = domain

        if self.pseudo_detection and domain == -1:
            # insert into outlier memory
            # check outlier memory for new clusters
            self.outlier_memory.append(item)
        elif not self.memoryfull:
            self.memorylist.append(item)
            if len(self.memorylist) == self.memorymaximum:
                self.memoryfull = True
        else:
            assert (item.current_grammatrix is not None)
            insertidx = -1
            if self.pseudo_detection:
                insertidx = self.find_insert_position()

            if insertidx == -1:
                mingramloss = 1000
                for j, ci in enumerate(self.memorylist):
                    l_sum = 10000
                    if self.pseudo_detection and ci.pseudo_domain == domain:
                        l_sum = F.mse_loss(torch.tensor(item.current_grammatrix), torch.tensor(ci.current_grammatrix),
                                           reduction='sum')
                    elif not self.pseudo_detection:
                        l_sum = 0.0
                        for i in range(len(item.current_grammatrix)):
                            l_sum += F.mse_loss(
                                item.current_grammatrix[i], ci.current_grammatrix[i], reduction='mean')

                    if l_sum < mingramloss:
                        mingramloss = l_sum.item()
                        insertidx = j

            if insertidx != -1:
                self.memorylist[insertidx] = item
            else:
                print('insertidx still -1')

    def find_insert_position(self):
        for idx, item in enumerate(self.memorylist):
            if item.deleteflag:
                return idx
        return -1

    def flag_items_for_deletion(self):
        for k, v in self.isoforests.items():
            domain_count = len(self.get_domainitems(k))
            print('domain', k, domain_count)
            if domain_count > self.max_per_domain:
                todelete = domain_count - self.max_per_domain
                for item in self.memorylist:
                    if todelete > 0:
                        if item.pseudo_domain == k:
                            if not item.deleteflag:
                                item.deleteflag = True

                            todelete -= 1

    def get_domainitems(self, domain):
        items = []
        for mi in self.memorylist:
            if mi.pseudo_domain == domain:
                items.append(mi)
        return items

    def check_outlier_memory(self, model):
        """
            Checks the outlier memory if there is a new pseudo domain detection
        """
        if len(self.outlier_memory) > 5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]

            distances = squareform(pdist(outlier_grams))
            if sorted([np.array(sorted(d)[:6]).sum() for d in distances])[5] < 0.20:

                clf = IsolationForest(n_estimators=5, random_state=self.seed, warm_start=True, contamination=0.10).fit(
                    outlier_grams)

                new_domain_label = len(self.isoforests)
                self.max_per_domain = int(self.memorymaximum / (new_domain_label + 1))
                self.domaincounter = 0

                self.flag_items_for_deletion()

                to_delete = []
                for k, p in enumerate(clf.predict(outlier_grams)):
                    if p == 1:
                        idx = self.find_insert_position()
                        if idx != -1:
                            elem = self.outlier_memory[k]
                            elem.pseudo_domain = new_domain_label
                            self.memorylist[idx] = elem
                            self.domaincounter += 1
                            to_delete.append(self.outlier_memory[k])
                for elem in to_delete:
                    self.outlier_memory.remove(elem)

                if self.domaincounter > 0:
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

    def get_training_batch(self, batchsize, forceditems=None):
        """
            Samples a training batch of batchsize elements consisting of forced items and randomly samples items

            :param forceditems (list MemoryItem): items that are forced to be in the batch
        """

        batchsize = min(batchsize, len(self.memorylist))

        imgshape = self.memorylist[0].img.shape
        x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2]))
        y = list()
        j = 0

        if forceditems is not None:
            for ci in forceditems:
                if j < batchsize:
                    x[j] = ci.img
                    y.append(ci.target)
                    ci.traincounter += 1
                    j += 1

            batchsize -= j

        if self.balance_memory and self.pseudo_detection and len(self.isoforests) > 1:
            items_per_domain = math.ceil(batchsize / len(self.isoforests))

            for i in range(len(self.isoforests)):
                domain_items = self.get_domainitems(i)
                random.shuffle(domain_items)
                for k in range(items_per_domain):
                    if batchsize > 0 and len(domain_items) > k:
                        x[j] = domain_items[k].img
                        y.append(domain_items[k].target)
                        j += 1
                        batchsize -= 1
                        domain_items[k].traincounter += 1

        if batchsize > 0:
            random.shuffle(self.memorylist)
            for ci in self.memorylist[-batchsize:]:
                x[j] = ci.img
                y.append(ci.target)
                ci.traincounter += 1
                j += 1

        return x, y

    def get_training_batches(self, batchsize, batches=2, randombatch=False, forceditems=None):
        """
            Samples multiple batches for training
        """

        batchsize = min(batchsize, len(self.memorylist))

        xs = []
        ys = []

        if forceditems is not None:
            force_per_batch = int(len(forceditems) / batches)

        for b in range(batches):
            bs = batchsize
            if forceditems is not None:
                forcedbatchitems = forceditems[b * force_per_batch:(b + 1) * force_per_batch]
            else:
                forcedbatchitems = None

            imgshape = self.memorylist[0].img.shape

            x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2]))

            y = list()

            j = 0

            if forcedbatchitems is not None:
                for ci in forcedbatchitems:
                    x[j] = ci.img
                    y.append(ci.target)
                    ci.traincounter += 1
                    j += 1

                bs -= j

            if randombatch:
                random.shuffle(self.memorylist)

            if bs > 0:
                for ci in self.memorylist[-bs:]:
                    x[j] = ci.img
                    y.append(ci.target)
                    ci.traincounter += 1
                    j += 1

            xs.append(x)
            ys.append(y)

        return xs, ys

    def counter_outlier_memory(self):
        """
            Increments the counter for each outlier item, eventually element are deleted if they exceed the outlier_epochs
        """

        for item in self.outlier_memory:
            item.outlier_counter += 1
            if item.outlier_counter > self.outlier_epochs:
                self.outlier_memory.remove(item)

    def __iter__(self):
        return self.memorylist.__iter__()
