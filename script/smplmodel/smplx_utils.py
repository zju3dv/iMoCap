'''
    @ some codes are from smplify-x: https://github.com/vchoutas/smplify-x
'''
import numpy as np
import torch
from os.path import join
import os
import smplx

def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))


class JointMapper(torch.nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)

class SMPLX:
    def __init__(self, model_folder=None, model_type='smplx', ext='npz',
        gender='neutral', 
        plot_joints=False,
        plotting_module='pyrender',
        use_face=True,
        use_hands=True,
        flat_hand_mean=False,
        use_face_contour=False, 
        batch_size=1,
        device=torch.device('cpu')):
        if model_folder is None:
            model_folder = join(smplx.__path__[0], 'models')
        self.use_hands = use_hands
        self.use_face = use_face
        self.use_face_contour = use_face_contour
        self.device = device
        self.batch_size = batch_size
        joint_mapper = JointMapper(smpl_to_openpose(
                    model_type, use_hands=use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format='coco25'))
        if model_type == 'smplh' and gender == 'neutral':
            import warnings
            warnings.warn("smplh does not have neutral model, \
                use male instead.")
            gender  = 'male'
        model_params = dict(
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True, 
                        batch_size=batch_size,
                        model_type=model_type,
                        gender=gender,
                        flat_hand_mean=flat_hand_mean)
        model = smplx.create(model_folder, 
                        **model_params)
        self.model = model.to(device)
        for par in model.parameters():
            par.requires_grad = False
        self.faces_tensor = self.model.faces_tensor
        self.faces = self.model.faces
        self.coco17inbody25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
        self.model_type = model_type
        # load regressor
        reg_path = join('data', 'smpl_regressor', 'J_regressor_body25.npy')
        if os.path.exists(reg_path):
            self.J_body25 = np.load(reg_path).T
            self.J_body25 = torch.Tensor(self.J_body25)[None, :, :].to(self.device)
        reg_path = join('data', 'smpl_regressor', 'J_regressor_h36m.npy')
        if os.path.exists(reg_path):
            # TODO:I forget where is the J_regressor from
            self.J_h36m = np.load(reg_path)
            # exchange left foot and right foot
            self.J_h36m[[1,2,3,4,5,6]] = self.J_h36m[[4,5,6,1,2,3]]
            self.J_h36m = torch.Tensor(self.J_h36m)[None, :, :].to(self.device)
        coco_hmmr_path = join(os.path.dirname(__file__), 'model', 'coco_hmmr.npy')
        print(self.model)
    
    def GetVerticesAll(self, inp_rot=None, inp_trans=None,
        global_orient=None, body_pose=None, jaw_pose=None,
        left_hand_pose=None, right_hand_pose=None,
        betas=None, expression=None):
        vertices = self.model(betas=betas,
            global_orient=global_orient,
            body_pose=body_pose, 
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose, 
            right_hand_pose=right_hand_pose, 
            expression=expression, 
            return_verts=True).vertices
        if inp_rot is None:
            inp_rot = torch.eye(3, 3, device=self.device).unsqueeze(0).repeat(vertices.shape[0], 1, 1)
        if inp_trans is None:
            inp_trans = torch.zeros(vertices.shape[0], 1, 3, device=self.device)
        transform = torch.matmul(vertices, inp_rot.transpose(1, 2)) + inp_trans
        return transform
    
    def splitKpts(self, keypoints):
        if keypoints.shape[1] > 67:
            return keypoints[:, :25], keypoints[:, 25:25+21], \
                keypoints[:, 25+21:25+21+21], keypoints[:, 25+21+21:]
        elif keypoints.shape[1] > 25:
            return keypoints[:, :25], keypoints[:, 25:25+21], \
                keypoints[:, 25+21:25+21+21]
        else:
            return keypoints

    def GetKeypointsAll(self, inp_rot=None, inp_trans=None,
        global_orient=None, body_pose=None, jaw_pose=None,
        left_hand_pose=None, right_hand_pose=None,
        betas=None, expression=None,
        split=True, return_verts=False):
        # Rot: (bn, 3, 3), trans:(bn, 1, 3)
        keypoints = self.model(betas=betas,
            global_orient=global_orient,
            body_pose=body_pose, 
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose, 
            right_hand_pose=right_hand_pose, 
            expression=expression,
            return_verts=return_verts)
        if return_verts:
            verts = keypoints.vertices
        keypoints = keypoints.joints
        # bn, 118, 3
        if inp_rot is None:
            inp_rot = torch.eye(3, 3, device=self.device).unsqueeze(0).repeat(keypoints.shape[0], 1, 1)
        if inp_trans is None:
            inp_trans = torch.zeros(keypoints.shape[0], 1, 3, device=self.device)
        transform = torch.matmul(keypoints, inp_rot.transpose(1, 2)) + inp_trans
        if not return_verts:
            if split == False:
                return transform
            else:
                return self.splitKpts(transform)
        else:
            verts = torch.matmul(verts, inp_rot.transpose(1, 2)) + inp_trans
            if split == False:
                return transform, verts
            else:
                return self.splitKpts(transform), verts
                
    def _GetVertices(self, inp_pose=None, inp_betas=None, inp_trans=None, inp_rot=None,
        jaw_pose=None, left_hand_pose=None, right_hand_pose=None,
        expression=None):
        global_orient = inp_pose[:, :3]
        if self.model_type != 'smpl': # for smplh and smplx
            body_pose = inp_pose[:, 3:66]
        else:
            body_pose = inp_pose[:, 3:]
        if inp_rot is not None:
            # pass
            global_orient = torch.zeros_like(global_orient, device=body_pose.device)
        keypoints = self.GetVerticesAll(
            global_orient=global_orient,
            inp_rot=inp_rot, inp_trans=inp_trans,
            betas=inp_betas, expression=expression,
            body_pose=body_pose, jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        return keypoints

    def GetVertices(self, inp_pose=None, inp_betas=None, inp_trans=None, inp_rot=None, kpts_type='body25',
        jaw_pose=None, left_hand_pose=None, right_hand_pose=None, 
        expression=None):
        params = {'inp_pose':inp_pose,
                  'inp_betas':inp_betas,
                  'inp_trans':inp_trans,
                  'inp_rot':inp_rot,
                  'jaw_pose':jaw_pose,
                  'left_hand_pose':left_hand_pose,
                  'right_hand_pose':right_hand_pose,
                  'expression':expression}
        nBatch = inp_pose.shape[0]
        for key in params.keys():
            if params[key] is not None:
                assert params[key].shape[0] == nBatch or params[key].shape[0] == 1, \
                    print(params[key].shape, ' is not a proper shape')
                if params[key].shape[0] == 1:
                    tgt_shape = [nBatch] + [-1]*(len(params[key].shape) - 1) 
                    params[key] = params[key].expand(tgt_shape)
        # padding or repeating
        keypoints = []
        begin = 0
        while True:
            if begin >= inp_pose.shape[0]:
                break
            params_batch = {
                key:params[key][begin:begin+self.batch_size] if params[key] is not None else None for key in params.keys()
            }
            batch_indeed = params_batch['inp_pose'].shape[0]
            for key in params_batch.keys():
                if params_batch[key] is not None:
                    if params_batch[key].shape[0] < self.batch_size:
                        # padding zero
                        zeros = torch.zeros((self.batch_size-params_batch[key].shape[0], *params_batch[key].shape[1:]), device=self.device)
                        params_batch[key] = torch.cat((params_batch[key], zeros), dim=0)
            _keypoints = self._GetVertices(**params_batch)
            keypoints.append(_keypoints[:batch_indeed])
            begin = begin + self.batch_size
        keypoints = torch.cat(keypoints, dim=0)

        return keypoints

    def _GetKeypoints(self, inp_pose=None, inp_betas=None, inp_trans=None, inp_rot=None,
        jaw_pose=None, left_hand_pose=None, right_hand_pose=None,
        expression=None, kpts_type='body25'):
        if self.model_type == 'smpl':
            # vertices: (bn, 6890, 3)
            vertices = self.GetVertices(inp_pose, inp_betas, inp_trans, inp_rot)
            if kpts_type == 'body25' or kpts_type == 'all':
                keypoints = torch.matmul(self.J_body25, vertices)
            elif kpts_type == 'coco17':
                keypoints = torch.matmul(self.J_body25, vertices)
                keypoints = keypoints[:, self.coco17inbody25]
            elif kpts_type == 'cocoplus':
                keypoints = torch.matmul(self.J_coco_hmmr, vertices)
            elif kpts_type == 'h36m':
                keypoints = torch.matmul(self.J_h36m, vertices)
            else:
                raise NotImplementedError
            return keypoints
        elif self.model_type == 'smplh':
            global_orient = inp_pose[:, :3]
            body_pose = inp_pose[:, 3:66]
            if inp_rot is not None:
                global_orient = torch.zeros_like(global_orient, device=body_pose.device)
            
            keypoints, vertices = self.GetKeypointsAll(
                global_orient=global_orient,
                inp_rot=inp_rot, inp_trans=inp_trans,
                betas=inp_betas,
                body_pose=body_pose, jaw_pose=jaw_pose,
                left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                split=False, return_verts=True)
            body25 = torch.matmul(self.J_body25, vertices)
            if kpts_type == 'body25':
                return body25
            elif kpts_type == 'coco17':
                return body25[:,self.coco17inbody25, :]
            elif kpts_type == 'all':
                return torch.cat([body25, keypoints[:, 25:]], dim=1)
            else:
                raise NotImplementedError

        elif self.model_type == 'smplx':
            global_orient = inp_pose[:, :3]
            body_pose = inp_pose[:, 3:66]
            if inp_rot is not None:
                global_orient = torch.zeros_like(global_orient, device=body_pose.device)
                
            keypoints = self.GetKeypointsAll(
                global_orient=global_orient,
                inp_rot=inp_rot, inp_trans=inp_trans,
                betas=inp_betas, expression=expression,
                body_pose=body_pose, jaw_pose=jaw_pose,
                left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                split=False)
            if kpts_type == 'body25':
                return keypoints[:, :25]
            elif kpts_type == 'coco17':
                return keypoints[:,self.coco17inbody25, :]
            elif kpts_type == 'all':
                return keypoints
            else:
                raise NotImplementedError
    
    def GetKeypoints(self, inp_pose=None, inp_betas=None, 
        inp_trans=None, inp_rot=None, kpts_type='body25',
        jaw_pose=None, left_hand_pose=None, right_hand_pose=None, 
        expression=None):
        params = {'inp_pose':inp_pose,
                  'inp_betas':inp_betas,
                  'inp_trans':inp_trans,
                  'inp_rot':inp_rot,
                  'jaw_pose':jaw_pose,
                  'left_hand_pose':left_hand_pose,
                  'right_hand_pose':right_hand_pose,
                  'expression':expression}
        nBatch = inp_pose.shape[0]
        keypoints = self._GetKeypoints(kpts_type=kpts_type, **params)
        return keypoints

    def GetBODY25(self, inp_pose, inp_betas=None, inp_trans=None, inp_rot=None):
        return self.GetKeypoints(inp_pose, 
            inp_betas=inp_betas,
            inp_trans=inp_trans,
            inp_rot=inp_rot,
            kpts_type='body25')
    
    def GetCOCO17(self, inp_pose, inp_betas=None, inp_trans=None, inp_rot=None):
        return self.GetKeypoints(inp_pose, 
            inp_betas=inp_betas,
            inp_trans=inp_trans,
            inp_rot=inp_rot,
            kpts_type='coco17')