from resources import *
from enum import Enum
from functools import partial

class AdversarialGenerationMethod(Enum):

    pixel_attack = partial(pixel_attack)
    shadow_attack = partial(shadow_attack)
    patch_attack = partial(patch_attack)
    JSM_attack = partial(JSM_attack)

    def __call__(self, *args):
        return self.value(*args)

def pixel_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass

def shadow_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass

def patch_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass

def JSM_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass


def get_candidates(dataset : ReiformICDataSet) -> ReiformICDataSet:

    # Select images that are in prime locations for evaluation to have adversarial pairs created and evaluated

    pass


def generate_pairs(dataset : ReiformICDataSet, 
                   methods : Dict[AdversarialGenerationMethod, Callable]) -> Dict[AdversarialGenerationMethod,
                                                                                  ReiformICDataSet]:
    
    resulting_datasets : Dict[AdversarialGenerationMethod, ReiformICDataSet] = {}

    for method in methods:
        resulting_datasets[method] = method(dataset)

    return resulting_datasets

