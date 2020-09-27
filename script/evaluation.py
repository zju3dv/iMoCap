import numpy as np
from os.path import join
import os
import joblib
import json
from collections import OrderedDict
USE_IDX = [3,  # R ankle
    2,  # R knee
    1,  # R hip
    4,  # L hip
    5,  # L knee
    6,  # L ankle
    16,  # R Wrist
    15,  # R Elbow
    14,  # R shoulder
    11,  # L shoulder
    12,  # L Elbow
    13,  # L Wrist
]

def evaluate_metric(smpl_render, joints3d, body_params, cam_params=None, joints_est=None, type='h36m'):
    device = smpl_render.device
    if 'Rh' in body_params.keys():
        Rh, Th = body_params['Rh'], body_params['Th']
        Rh = torch.Tensor(Rh).to(device)
        Th = torch.Tensor(Th[:, None, :]).to(device)
    else:
        Rh = None
        Th = None
    poses, shapes = body_params['poses'], body_params['shapes']
    if not torch.is_tensor(poses):
        poses = torch.Tensor(poses).to(device)
        shapes = torch.Tensor(shapes).to(device)
        if cam_params is not None:
            for key in cam_params.keys():
                cam_params[key] = torch.Tensor(cam_params[key]).to(device)
    if Rh is None:
        rot = None
    else:
        rot = batch_rodrigues(Rh)
    if joints_est is None:
        if type == 'h36m':
            keypoints = smpl_render.GetKeypoints(
                inp_pose=poses,
                inp_betas=shapes,
                inp_rot=rot,
                inp_trans=Th,
                kpts_type='h36m',
            )
        elif type == 'cocoplus':
            keypoints = smpl_render.GetKeypoints(inp_pose=poses, 
                    inp_betas=shapes, 
                    inp_rot=rot,
                    kpts_type='cocoplus')
    else:
        keypoints = torch.Tensor(joints_est).to(smpl_render.device)

    if cam_params is not None:
        Rc, Tc = cam_params['Rc'], cam_params['Tc']
        keypoints = keypoints @ Rc.t()
    if type == 'h36m':
        errors_joints, errors_joints_pa = compute_error_3d(
            gt3ds=joints3d[:, USE_IDX],
            preds=keypoints.detach().cpu().numpy()[:, USE_IDX],
            vis=None,
        )
    elif type == 'cocoplus':
        joints_pred = keypoints.detach().cpu().numpy()[:, :len(USE_IDX)]
        errors_joints, errors_joints_pa = compute_error_3d(
            gt3ds=joints3d[:, USE_IDX],
            preds=joints_pred,
            vis=None,
        )
    errors_joints = np.array(errors_joints).mean()
    errors_joints_pa = np.array(errors_joints_pa).mean()
    report = {}
    report['MPJPE-PA'] = '%.2f'%(errors_joints_pa*1000)
    report['MPJPE'] = '%.2f'%(errors_joints*1000)
    return report

def compute_error_3d(gt3ds, preds, vis=None):
    """
    Returns MPJPE after pelvis alignment and MPJPE after Procrustes. Should
    evaluate only on the 14 common joints.

    Args:
        gt3ds (Nx14x3).
        preds (Nx14x3).
        vis (N).

    Returns:
        MPJPE, PA-MPJPE
    """
    # import open3d
    # import smpl
    # import matplotlib.pyplot as plt
    min_frames = min(gt3ds.shape[0], preds.shape[0])
    gt3ds = gt3ds[:min_frames]
    preds = preds[:min_frames]
    assert len(gt3ds) == len(preds)
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        if vis is None or vis[i]:
            gt3d = gt3d.reshape(-1, 3)
            # Root align.
            gt3d = align_by_pelvis(gt3d)
            pred3d = align_by_pelvis(pred)
            # ax = smpl.plotSkel3D(gt3d, config=smpl.CONFIG['lsp'])
            # ax = smpl.plotSkel3D(pred3d, config=smpl.CONFIG['lsp'], ax=ax)
            # plt.show()
            # 假装增加一个骨长
            scale = np.sqrt(np.sum(gt3d**2))/np.sqrt(np.sum(pred3d**2))
            pred3d = pred3d*scale
            # 增加骨长结束
            joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
            errors.append(np.mean(joint_error))
            # Get PA error.
            pred3d_sym = compute_similarity_transform(pred3d, gt3d)
            pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
            errors_pa.append(np.mean(pa_error))

    return errors, errors_pa

def align_by_pelvis(joints, get_pelvis=False):
    """
    Aligns joints by pelvis to be at origin. Assumes hips are index 3 and 2 of
    joints (14x3) in LSP order. Pelvis is midpoint of hips.

    Args:
        joints (14x3).
        get_pelvis (bool).
    """
    left_id = 3
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)
        
def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points
    S1 (3 x N) closest to a set of 3D points S2, where R is an 3x3 rotation
    matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrustes problem.

    Args:
        S1 (3xN): Original 3D points.
        S2 (3xN'): Target 3D points.

    Returns:
        S1_hat: S1 after applying optimal alignment.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1
    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat
    
def compute_match_error(match, match_data, ref_id=0):
    match_gt = match_data['match']
    match_idx = match_data['index']
    match_0 = match[:, match_gt[ref_id]]
    res = []
    mapiv = cfg.VIEWS
    for iv in range(match.shape[0]):
        res.append(np.array(match_idx[mapiv[iv]])[match[iv]])
    res = np.array(res)
    error = np.abs(res[1:] - res[:1]).mean()/max(match_idx[ref_id])
    match_err = '%.2f%%'%(error*100)
    return match_err

def evaluate(data):
    results = {'MPJPE':0, 'MPJPE-PA':0}
    videos = list(data.keys())
    action = videos[0].split('.')[0]
    match_gt = join('../dataset/h36m', action, 'match_gt.json')
    with open(match_gt) as f:
        match_data = json.load(f)
    if 'match' in data.keys():
        match = data.pop('match')
        ref_id = data.pop('ref_id')
        match_err = compute_match_error(match, match_data, 0)
        results['Match'] = match_err
    
    pose_gt = join('./data/joints3d/', 'S9_{}.mat'.format(action.replace('1', ' 1')))
    from scipy.io import loadmat, savemat
    annot = loadmat(pose_gt)
    videonames = sorted(list(data.keys()))
    if args.eval0:
        videonames = [videonames[0]]
    videonames_all = ['{}.{}'.format(action, v) for v in ['54138969', '55011271', '58860488', '60457274']]
    for videoname in videonames:
        test_view = videonames_all.index(videoname)
        idx = match_data['index'][test_view]
        used_frames = match_data['match'][test_view]
        idx = [idx[i] for i in used_frames]
        if args.fps25:
            idx = [i*2 for i in idx]
        joints3d = annot['joints3d'][idx, :, :]
        data_ = data[videoname]
        joints = GetVertices(smpl_render, data_, True)[used_frames, :, :]
        if 'camera' in data_.keys():
            camera = data_.pop('camera')
        else:
            camera = None
        joints3d_ = joints3d @ annot['R'][test_view].T + annot['T'][test_view:test_view+1]
        res = evaluate_metric(smpl_render, joints3d_, data_, camera, joints, 'h36m')
        for key in res.keys():
            results[key] += float(res[key])/len(videonames)
    results['MPJPE'] = '{:.2f}'.format(results['MPJPE'])
    results['MPJPE-PA'] = '{:.2f}'.format(results['MPJPE-PA'])
    return results

def GetVertices(smpl_render, params, return_joints=False):
    if 'Th' in params.keys():
        Th = torch.Tensor(params['Th'][:, None, :]).to(device)
        Rh = torch.Tensor(params['Rh']).to(device)
        rot = batch_rodrigues(Rh)
    else:
        Th = None 
        rot = None
    inpdict = {
        'inp_pose': torch.Tensor(params['poses']).to(device),
        'inp_betas': torch.Tensor(params['shapes']).to(device),
        'inp_trans': Th,
        'inp_rot': rot,
        'kpts_type': smpl_render.valid_type
    }
    if return_joints:
        vertices = smpl_render.GetKeypoints(**inpdict)
    else:
        vertices = smpl_render.GetVertices(**inpdict)
    if True:
        return vertices.detach().cpu().numpy()
    else:
        return vertices
        
def to_table(all_report, savename=None):
    # 输出
    headers = [args.path, 'MPJPE', 'MPJPE-PA', 'Match']
    report_ref = all_report
    table = []
    for act in report_ref.keys():
        tabs = [act]
        for key in headers[1:]:
            if key in report_ref[act].keys():
                tabs.append(report_ref[act][key])
            else:
                tabs.append('*')
        table.append(tabs)

    from tabulate import tabulate
    # tab_name = join(output_dir, 'table_fancy.txt')
    print(tabulate(table, headers, tablefmt='fancy_grid'))
    if savename is not None:
        print(tabulate(table, headers, tablefmt='plain'), file=open(savename, 'w'))
    
if __name__ == "__main__":
    from config.cmdline import parse_args
    from config.default import update_config, cfg
    args = parse_args()
    update_config(cfg, args)
    if args.usecpp:
        pass
    else:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        from smplmodel.smplx_utils import SMPLX
        from smplmodel.geometry import batch_rodrigues
        smpl_render = SMPLX(model_type=cfg.MODEL.model_type,
            gender=cfg.MODEL.gender, 
            batch_size=cfg.MODEL.batch_size, device=device,
            model_folder=cfg.MODEL.model_folder)
        smpl_render.kpts_type = cfg.MODEL.kpts_type
        smpl_render.valid_type = 'h36m'
    
    if args.tcc:
        match_gt = join(args.path, 'match_gt.json')
        with open(match_gt) as f:
            match_data = json.load(f)
        predicted_match_path = join(args.path, 'matching_results.pkl')
        predicted_match = np.array(joblib.load(predicted_match_path))
        match_err = compute_match_error(predicted_match, match_data, 0)
        print('The matching error of TCC on {} is {}.'.format(os.path.basename(args.path), match_err))
    else:
        checkpoint_path = join(args.path, 'state')
        from glob import glob
        checkpoints = sorted(glob(join(checkpoint_path, '*.pkl')))
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('.')[0]) if 'final' not in x else 999)
        print('>>> detected checkpoints: ', ' '.join(checkpoints))
        all_report = OrderedDict()
        # preprocess:
        pre_path = os.path.abspath(join(args.path, '..', 'preprocess_{}_{}.pkl'.format(cfg.MODEL.smpl_method, cfg.MODEL.pose_method)))
        if os.path.exists(pre_path):
            data = joblib.load(pre_path)
            reports = evaluate(data)
            all_report['CNN'] = reports
        fine_paths = sorted(glob(os.path.abspath(join(args.path, '..', 'finetune*_{}_{}.pkl'.format(cfg.MODEL.smpl_method, cfg.MODEL.pose_method)))))
        if len(fine_paths) > 0:
            for fine_path in fine_paths:
                data = joblib.load(fine_path)
                reports = evaluate(data)
                basename = os.path.basename(fine_path)
                all_report[basename] = reports

        for check in checkpoints:
            data = joblib.load(check)
            reports = evaluate(data)
            basename = os.path.basename(check)
            all_report[basename] = reports
        to_table(all_report, savename=join(args.path, 'report.txt'))