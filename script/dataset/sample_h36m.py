import os 
from glob import glob
from os.path import join
import numpy as np
import subprocess
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='')
parser.add_argument('--out_path', type=str, default='../dataset/h36m')
parser.add_argument('--num', type=int, default=7)
parser.add_argument('--all', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def extract_video(video, out_dir):
    mkdir(out_dir)
    cmd = [
        'ffmpeg',
        '-i', '{}'.format(video),
        '-start_number', '0',
        # '-r', '25', # h36m sample to 25fps
        '{temp_dir}/frame%08d.png'.format(temp_dir=out_dir),
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)

def compress_video(path, savename):
    cmd = ['ffmpeg', '-y', '-i', '{}/frame%08d.png'.format(
        path), '{}'.format(join(path, '..', savename+'.mp4'))]
    subprocess.call(cmd)
    os.system('rm {}/frame*.png'.format(path))
    

def sample_action(video_path, out_path, seq,
    max_frame=2000, rate=0.075, rate_rest=0.1, rate_pad=0.1,
    num_point_pad=50):
    # initialize RNG with seeds from sequence id
    import hashlib
    s = "{}:{}:{}:{}:{}:{}".format(seq, max_frame, rate, rate_rest, rate_pad, num_point_pad)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    print("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)

    if os.path.exists(join(out_path, seq, 'match_gt.json')):
        print('file already exists!')
        return False
    videos = sorted(glob(join(video_path, seq, '*.mp4')))
    basenames = [os.path.basename(video).replace('.mp4', '') for video in videos]
    for video in videos:
        basename = os.path.basename(video).replace('.mp4', '')
        out_dir = join(out_path, seq, basename, 'tmp')
        extract_video(video, out_dir)
    imglists = []
    for video in videos:
        basename = os.path.basename(video).replace('.mp4', '')
        imglist = sorted(glob(join(out_path, seq, basename, 'tmp', '*.png')))
        imglists.append(imglist[:max_frame])

    out_path = join(out_path, seq)
    max_frame = min(max_frame, len(imglists[0]))
    choice = np.sort(np.linspace(0, max_frame - 1, int(max_frame*rate)).astype(int))
    # generate the reference video
    for ii, idx in enumerate(choice):
        os.system('cp {} {}'.format(imglists[0][idx], join(out_path, basenames[0], 'frame%08d.png'%(ii))))
    compress_video(join(out_path, basenames[0]), basenames[0])
    # generate the other view
    results = [choice.tolist()]
    density_index = np.array([0, 1, num_point_pad * 0.1,
                              num_point_pad * 0.2, num_point_pad]).astype(int)
    for i in range(1, 4):
        choice_section = np.random.choice(choice[3:-3], num_point_pad, replace=False)
        choice_all = []
        for density_i in range(len(density_index)-1):
            density = choice_section[density_index[density_i]:density_index[density_i+1]]
            for item in density:
                index = np.argwhere(choice==item)[0, 0]
                choice_item = np.arange(choice[index]+1, choice[index+1])
                if density_i == 0:
                    choice_item = np.array([])
                    for pad in range(3):
                        choice_item_p = np.arange(choice[index + pad] + 1, choice[index + 1 + pad]+1)
                        choice_item_n = np.arange(choice[index - 1 - pad] + 1, choice[index - pad]+1)
                        choice_item = np.append(choice_item, choice_item_p)
                        choice_item = np.append(choice_item, choice_item_n)
                    choice_item = choice_item.astype(int)
                elif density_i == 1:
                    choice_item = np.random.choice(choice_item, int(choice_item.shape[0] / 3), replace=False)
                elif density_i == 2:
                    choice_item = np.random.choice(choice_item, int(choice_item.shape[0] / 6), replace=False)
                else:
                    choice_item = np.random.choice(choice_item, 1, replace=False)
                choice_all = choice_all + list(choice_item)
        choice_all = list(choice_all) + list(choice)
        choice_all.sort()
        for ii, idx in enumerate(choice_all):
            os.system('cp {} {}'.format(imglists[i][idx], join(out_path, basenames[i], 'frame%08d.png'%(ii))))
        compress_video(join(out_path, basenames[i]), basenames[i])
        results.append(choice_all)
    match_gt = np.zeros((len(results), len(choice)), dtype=np.int)
    for ii, idx in enumerate(choice):
        for iv in range(len(results)):
            match_gt[iv, ii] = results[iv].index(idx)
    import json
    output = {}
    for i in range(len(results)):
        results[i] = [int(var) for var in results[i]]
    output['index'] = results
    output['match'] = match_gt.tolist()
    with open(join(out_path, 'match_gt.json'), 'w') as f:
        json.dump(output, f, indent=4)
    import shutil
    for basename in basenames:
        shutil.rmtree(join(out_path, basename))
        
if __name__ == "__main__":
    # we select the most challenge actions from h36m
    # according to https://arxiv.org/pdf/1805.04095.pdf
    mpjpes = {'Directions':48.5, 'Discussion1': 54.4, 'Eating':54.4,
        'Greeting':52.0, 'Phoning':59.4, 'Photo':65.3, 'Posing':49.9,
        'Purchases':52.9, 'Sitting':65.8, 'SittingDown':71.1, 'Smoking':56.6,
        'Waiting':52.9, 'WalkDog':60.9, 'WalkTogether':47.8, 'Walking':44.7}
    sort = sorted(mpjpes.items(), key=lambda x:-x[1])
    seqnames = [s[0] for s in sort]
    if not args.all:
        seqnames = seqnames[:args.num]
        print('We sample {} videos: '.format(args.num), seqnames)
    mkdir(args.out_path)
    if args.debug:
        for seq in ['Sitting']:
            sample_action(args.video_path, args.out_path, seq)
    else:
        for seq in seqnames:
            sample_action(args.video_path, args.out_path, seq)