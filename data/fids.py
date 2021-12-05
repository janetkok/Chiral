import pandas as pd
import torch
import numpy as np
class FormulatedImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples formulated amount of instances from minority label combinations on top of the original imbalanced dataset
    Arguments:
        dataset: label combination imbalanced dataset
        indices: a list of indices
        callback_get_label: a callback-like function which takes two arguments - dataset and index
        generator:  If not ``None``, this RNG will be used to generate random indexes and multiprocessing to generate
            `base_seed` for workers. (default: ``None``)
        minPerct: percentage threshold to be considered as minority label combination
        addPerct: potential percentage of minority label combination to be added to dataset
        maxI: maximum images that can be added per instance
    """

    def __init__(self, dataset, indices: list = None,
                 generator=None, minPerct=0, addPerct=0, maxI=0):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        self.generator = generator

        # distribution of label combinations in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()

        # minority label combinations
        M = label_to_count[label_to_count/len(dataset) < minPerct]
        MInd = M.index.values

        # adding a formulated amount of instances from minority label combination on top of original dataset
        NPI = [round(len(dataset)*addPerct/i) for i in (M.values.tolist())] #number of images that can be sampled per instance
        RNPI = np.clip(NPI, 1, maxI) #regulated number of images that can be sampled per instance
        for m, mRNPI in zip(MInd, RNPI):
            mInd = df[df["label"] == m].index.values.tolist() #all indices of the instances of the minority label combination
            self.indices = self.indices + (mInd*mRNPI)

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

    def _get_labels(self, dataset):
        if isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator).tolist())

    def __len__(self):
        return self.num_samples