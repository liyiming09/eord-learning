### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from data.confounder_dataset import SegmentationDataset

class CcityBoxDataset(SegmentationDataset):
    def initialize(self, opt):
        super(CcityBoxDataset, self).initialize(opt)
        self.class_of_interest = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33] # will define it in child
        self.class_of_background = None # will define it in child

    def name(self):
        return 'CitiscapesBoxDataset'
