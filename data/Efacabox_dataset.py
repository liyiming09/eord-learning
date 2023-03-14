### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from data.learneord_dataset import SegmentationDataset

class EFacaBoxDataset(SegmentationDataset):
    def initialize(self, opt):
        super(EFacaBoxDataset, self).initialize(opt)
        self.class_of_interest = [2, 3, 4, 6, 7, 8, 9, 10, 11] # will define it in child
        self.class_of_background = None # will define it in child

    def name(self):
        return 'CitiscapesBoxDataset'
