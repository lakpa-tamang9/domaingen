# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.ERMDPP import ERMDPP
from alg.algs.MMD import MMD
from alg.algs.MMDDPP import MMDDPP
from alg.algs.CORAL import CORAL
from alg.algs.CORAL_DPP import CORAL_DPP
from alg.algs.DANN import DANN
from alg.algs.DANN_DPP import DANN_DPP
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup
from alg.algs.Mixup_DPP import Mixup_DPP
from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.GroupDRO_DPP import GroupDRO_DPP
from alg.algs.ANDMask import ANDMask
from alg.algs.VREx import VREx
from alg.algs.DIFEX import DIFEX
from alg.algs.myalg import AAE

ALGORITHMS = [
    "ERM",
    "ERMDPP",
    "Mixup",
    "Mixup_DPP",
    "CORAL",
    "CORAL_DPP",
    "MMD",
    "MMDDPP",
    "DANN",
    "DANN_DPP",
    "MLDG",
    "GroupDRO",
    "GroupDRO_DPP",
    "RSC",
    "ANDMask",
    "VREx",
    "DIFEX",
    "AAE",
    "MBDG",
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
