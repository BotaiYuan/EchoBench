import pandas as pd
import os.path as osp
import os
import multiprocessing as mp
import io
import base64
from PIL import Image
import csv

# LOAD & DUMP
def dump(data, f, **kwargs):

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    return dump_tsv(data, f, **kwargs)

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)

def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

def decode_img_omni(tup):
    root, im, p = tup
    images = toliststr(im)
    paths = toliststr(p)
    if len(images) > 1 and len(paths) == 1:
        paths = [osp.splitext(p)[0] + f'_{i}' + osp.splitext(p)[1] for i in range(len(images))]

    assert len(images) == len(paths)
    paths = [osp.join(root, p) for p in paths]
    for p, im in zip(paths, images):
        if osp.exists(p):
            continue
        if isinstance(im, str) and len(im) > 64:
            decode_base64_to_image_file(im, p)
    return paths

def localize_df(data, dname, nproc=32):
    assert 'image' in data
    indices = list(data['index'])
    indices_str = [str(x) for x in indices]
    images = list(data['image'])
    image_map = {x: y for x, y in zip(indices_str, images)}

    root = osp.join('images', dname)
    os.makedirs(root, exist_ok=True)

    if 'image_path' in data:
        img_paths = list(data['image_path'])
    else:
        img_paths = []
        for i in indices_str:
            if len(image_map[i]) <= 64 and isinstance(image_map[i], str):
                idx = image_map[i]
                assert idx in image_map and len(image_map[idx]) > 64
                img_paths.append(f'{idx}.jpg')
            else:
                img_paths.append(f'{i}.jpg')

    tups = [(root, im, p) for p, im in zip(img_paths, images)]

    paths=[]
    # decode_img_omni(tups)
    for tup in tups:
        path=decode_img_omni(tup)
        paths.append(path)

    data.pop('image')
    if 'image_path' not in data:
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    return data

def LOCALIZE(fname, new_fname=None):
    if new_fname is None:
        new_fname = fname.replace('.tsv', '_local.tsv')

    base_name = osp.basename(fname)
    dname = osp.splitext(base_name)[0]

    # data = pd.read_csv(fname, sep='\t',skiprows=0, nrows=1000,header=0)
    data = pd.read_csv(fname, sep='\t')
    data_new = localize_df(data, dname)
    dump(data_new, new_fname)
    print(f'The localized version of data file is {new_fname}')
    return new_fname

fname='./EchoBench.tsv'
local_path = './EchoBench_local.tsv'

LOCALIZE(fname, local_path)

