'''
 * @ Date: 2020-09-27 22:50:23
 * @ Author: Qing Shuai
 * @ LastEditors: Qing Shuai
 * @ LastEditTime: 2020-09-27 22:50:24
 * @ FilePath: /iMoCap/script/config/default.py
'''
import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.DATA_DIR = 'data'

_C.MAX_ITER = 5
_C.USE_VIEWS = -1
_C.VIEWS = [0, 1, 2, 3]
_C.DEBUG = CN()
_C.DEBUG.RENDER_VIEW = False
_C.DEBUG.RENDER_SCENE = False
_C.DEBUG.VERBOSE = False

# for SMPL model
_C.MODEL = CN()

_C.MODEL.match_method = 'ALS'
_C.MODEL.rank_rate = 1.
_C.MODEL.model_type = 'smpl'
_C.MODEL.gender = 'male'
_C.MODEL.batch_size = 1
_C.MODEL.model_folder = 'data/smplx_data'
_C.MODEL.kpts_type = 'body25'
_C.MODEL.smpl_method = 'hmmr'
_C.MODEL.pose_method = 'cpn'

_C.FIT = CN(new_allowed=True)
_C.FIT.ROBUST = False
_C.FIT.SCALE = 1
_C.FIT.RHO2D = 1
# for finetune stage
_C.FINETUNE = CN()
_C.FINETUNE.WEIGHT = CN()
_C.FINETUNE.WEIGHT.repro = 1.
_C.FINETUNE.WEIGHT.kpts_huber_a = 0.1
_C.FINETUNE.WEIGHT.init = 0.
_C.FINETUNE.WEIGHT.init_l1 = 0.
_C.FINETUNE.WEIGHT.reg_pose_l1 = 0.
_C.FINETUNE.WEIGHT.smooth = 0.1
_C.FINETUNE.WEIGHT.smooth_pose = 0.
_C.FINETUNE.WEIGHT.smooth_pose_l1 = 0.
_C.FINETUNE.WEIGHT.smooth_rh = 0.
_C.FINETUNE.WEIGHT.smooth_th = 0.

_C.FINETUNE.ROBUST_2D = False
_C.FINETUNE.VISUALIZE = False

# for optimize stage
_C.OPTIMIZE = CN()
_C.OPTIMIZE.RANK = 1
_C.OPTIMIZE.ROBUST = True

_C.OPTIMIZE.WEIGHT = CN()
_C.OPTIMIZE.WEIGHT.repro = 1.
_C.OPTIMIZE.WEIGHT.kpts_huber_a = 0.
_C.OPTIMIZE.WEIGHT.reg_pose_l1 = 0.
_C.OPTIMIZE.WEIGHT.init = 0.
_C.OPTIMIZE.WEIGHT.init_l1 = 0.
_C.OPTIMIZE.WEIGHT.smooth = 0.
_C.OPTIMIZE.WEIGHT.smooth_pose = 0.
_C.OPTIMIZE.WEIGHT.smooth_pose_l1 = 0.
_C.OPTIMIZE.WEIGHT.smooth_rh = 0.
_C.OPTIMIZE.WEIGHT.smooth_th = 0.
_C.OPTIMIZE.WEIGHT.rank = 1.
_C.OPTIMIZE.WEIGHT.rank_pose = 0.
_C.OPTIMIZE.WEIGHT.rank_pose_l1 = 0.
_C.OPTIMIZE.WEIGHT.rank_rh = 0.
_C.OPTIMIZE.WEIGHT.rank_th = 0.

# for matching
_C.MATCH = CN()
_C.MATCH.USE_SVT = True
_C.MATCH.USE_DUAL = True
_C.MATCH.VISUALIZE = False
_C.MATCH.alpha = 0.1
_C.MATCH.tol = 1e-3
_C.MATCH.maxIter = 50
_C.MATCH.verbose = False
_C.MATCH.REF_ID = -1

cfg = _C

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

def check_config(cfg):
    pass
