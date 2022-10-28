from losses.reward_criterion import RewardCriterion

__factory = {
    'RewardCriterion': RewardCriterion,
}

def names():
    return sorted(__factory.keys())

def create(name):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()