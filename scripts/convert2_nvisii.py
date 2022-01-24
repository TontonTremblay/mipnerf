import json
import os
from os import path

from absl import app
from absl import flags
import jax
import numpy as np
from PIL import Image
import glob 

FLAGS = flags.FLAGS

flags.DEFINE_string('blenderdir', None,
                    'Base directory for all Blender data.')
flags.DEFINE_string('outdir', "mip",
                    'Where to save multiscale data.')
flags.DEFINE_integer('n_down', 4,
                     'How many levels of downscaling to use.')

jax.config.parse_flags_with_absl()


def load_renderings(data_dir, split):
  """Load images and metadata from disk."""
  f = 'transforms_{}.json'.format(split)
  with open(path.join(data_dir, f), 'r') as fp:
    meta = json.load(fp)
  images = []
  cams = []
  # raise()
  # print('Loading imgs')
  # print(meta['frames'])
  for frame in meta['frames']:
    fname = os.path.join(data_dir, frame['file_path'] + '.png')
    with open(fname, 'rb') as imgin:
      image = np.array(Image.open(imgin), dtype=np.float32) / 255.
    cams.append(frame['transform_matrix'])
    images.append(image)
  print(len(images))
  # raise()
  ret = {}
  ret['images'] = np.stack(images, axis=0)
  print('Loaded all images, shape is', ret['images'].shape)
  ret['camtoworlds'] = np.stack(cams, axis=0)
  w = ret['images'].shape[2]
  camera_angle_x = float(meta['camera_angle_x'])
  print()
  ret['focal'] = .5 * w / np.tan(.5 * camera_angle_x)
  return ret


def down2(img):
  sh = img.shape
  return np.mean(np.reshape(img, [sh[0] // 2, 2, sh[1] // 2, 2, -1]), (1, 3))


def convert_to_nerfdata(basedir, newdir, n_down, near_far = [2.,6.]):
  """Convert Blender data to multiscale."""
  if not os.path.exists(newdir):
    os.makedirs(newdir)
  # splits = ['train', 'val', 'test']
  splits = ['test']
  bigmeta = {}
  # Foreach split in the dataset
  for split in splits:
    print('Split', split)
    # Load everything
    data = load_renderings(basedir, split)

    # Save out all the images
    imgdir = 'images_{}'.format(split)
    os.makedirs(os.path.join(newdir, imgdir), exist_ok=True)
    fnames = []
    widths = []
    heights = []
    focals = []
    cam2worlds = []
    lossmults = []
    labels = []
    nears, fars = [], []
    f = data['focal']
    print('Saving images')
    for i, img in enumerate(data['images']):
      for j in range(n_down):
        fname = '{}/{:03d}_d{}.png'.format(imgdir, i, j)
        fnames.append(fname)
        fname = os.path.join(newdir, fname)
        with open(fname, 'wb') as imgout:
          img8 = Image.fromarray(np.uint8(img * 255))
          img8.save(imgout)
        widths.append(img.shape[1])
        heights.append(img.shape[0])
        focals.append(f / 2**j)
        cam2worlds.append(data['camtoworlds'][i].tolist())
        lossmults.append(4.**j)
        labels.append(j)
        nears.append(near_far[0])
        fars.append(near_far[1])
        img = down2(img)

    # Create metadata
    meta = {}
    meta['file_path'] = fnames
    meta['cam2world'] = cam2worlds
    meta['width'] = widths
    meta['height'] = heights
    meta['focal'] = focals
    meta['label'] = labels
    meta['near'] = nears
    meta['far'] = fars
    meta['lossmult'] = lossmults

    fx = np.array(focals)
    fy = np.array(focals)
    cx = np.array(meta['width']) * .5
    cy = np.array(meta['height']) * .5
    arr0 = np.zeros_like(cx)
    arr1 = np.ones_like(cx)
    k_inv = np.array([
        [arr1 / fx, arr0, -cx / fx],
        [arr0, -arr1 / fy, cy / fy],
        [arr0, arr0, -arr1],
    ])
    k_inv = np.moveaxis(k_inv, -1, 0)
    meta['pix2cam'] = k_inv.tolist()

    bigmeta[split] = meta

  for k in bigmeta:
    for j in bigmeta[k]:
      print(k, j, type(bigmeta[k][j]), np.array(bigmeta[k][j]).shape)

  jsonfile = os.path.join(newdir, 'metadata.json')
  with open(jsonfile, 'w') as f:
    json.dump(bigmeta, f, ensure_ascii=False, indent=4)


def main(unused_argv):

  blenderdir = FLAGS.blenderdir
  outdir = FLAGS.outdir
  n_down = FLAGS.n_down
  # outdir = blenderdir + '/' + outdir 
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  # structure folders
  # images_XXX, XXX 

  # files
  # metadata.json
  # transform_XXX.json
  folders = ['train','test','val']
  for folder in folders:
    if not os.path.exists(outdir + '/'+folder):
      os.makedirs(outdir + '/'+folder)

  #generete the json files
  out_train = {}
  out_val = {}
  out_test = {}

  out_train['frames']=[]
  out_val['frames']=[]
  out_test['frames']=[]

  scale = 1
  if "lego" in blenderdir:
    with open('poses_lego.npy', 'rb') as f:
      poses = np.load(f)  
  if "google" in blenderdir:
    with open('poses_google.npy', 'rb') as f:
      poses = np.load(f)  

  if "amazon" in blenderdir:
    with open('poses_amazon.npy', 'rb') as f:
      poses = np.load(f)  

  if "abc" in blenderdir:
    with open('poses_abc.npy', 'rb') as f:
      poses = np.load(f)  
  


  # poses = [[[-9.32621241e-01, -1.68306619e-01,  3.19203198e-01,  3.35528678e-01*5],
  #           [ 3.60857040e-01, -4.34981942e-01,  8.24968517e-01,  7.08714938e-01*5],
  #           [-4.47034836e-08,  8.84569824e-01,  4.66407895e-01,  4.44145864e-01*5],
  #           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.10000000e+00],]]
  # poses = [poses[0]]
  out = out_test
  folder_name = 'test'
  
  file_name = sorted(glob.glob(blenderdir+"/*.json"))[0]

  # for i_file_name, file_name in enumerate(sorted(glob.glob(blenderdir+"/*.json"))):
  # print(len(poses))
  # raise()
  for i_file_name, pose in enumerate(poses):
    i_file_name = 0
    # if i_file_name>=100 and i_file_name<105:
    #   folder_name = 'val'

    #   out = out_val
    # elif i_file_name>=105:
    #   out = out_test
    #   folder_name = 'test'



    c2w=[]
    print(i_file_name,file_name)
    with open(file_name) as json_file:
        data = json.load(json_file)
        c2w = data["camera_data"]["cam2world"]        
    fr={}
    fr["file_path"]=folder_name+"/"+"{num:05d}".format(num=i_file_name)
    pose = np.array(pose).T
    fr["transform_matrix"]=[[pose[0][0],pose[1][0],pose[2][0],pose[3][0]*scale],
                            [pose[0][1],pose[1][1],pose[2][1],pose[3][1]*scale],
                            [pose[0][2],pose[1][2],pose[2][2],pose[3][2]*scale],
                            [pose[0][3],pose[1][3],pose[2][3],pose[3][3]]] 
    # fr["transform_matrix"] = pose 
    out['frames'].append(fr)
    # break
  out_train['aabb'] = [
    [
      data['camera_data']['scene_min_3d_box'][0],
      data['camera_data']['scene_min_3d_box'][1],
      data['camera_data']['scene_min_3d_box'][2],
    ],
    [
      data['camera_data']['scene_max_3d_box'][0],
      data['camera_data']['scene_max_3d_box'][1],
      data['camera_data']['scene_max_3d_box'][2],
    ],
  ]
  out_test['aabb'] = out_train['aabb']
  out_val['aabb'] = out_train['aabb']

  cx = data['camera_data']['intrinsics']['cx']
  fx = data['camera_data']['intrinsics']['fx']

  import math
  # camang = math.atan((cx/4)/(fx/4))*2
  print('for 1600')
  print('fx',fx)
  print("fov",math.atan((1600/(2*fx)))*2)

  print('for 800')
  fx/=2 
  print('fx',fx)
  print("fov",math.atan((800/(2*fx)))*2)


  # camang = math.atan((800/(2*fx)))*2

  # print('fx',fx,0.5*1600/np.tan(0.5*camang),.5 * 800 / np.tan(.5 * camang))
  # print(0.785398)
  # print(camang)
  # raise()

  out_train['camera_angle_x']= math.atan((800/(2*fx)))*2 
  out_test['camera_angle_x'] = out_train['camera_angle_x']
  out_val['camera_angle_x'] = out_train['camera_angle_x']


  with open(f'{outdir}/transforms_train.json', 'w') as outfile:
      json.dump(out_train, outfile, indent=2)
  with open(f'{outdir}/transforms_val.json', 'w') as outfile:
      json.dump(out_val, outfile, indent=2)
  with open(f'{outdir}/transforms_test.json', 'w') as outfile:
      json.dump(out_test, outfile, indent=2)

  # make the png files
  # raise()
  def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    img[img > 1] = 1
    img[img < 0] = 0
    return img

  def load_rgb_exr(img_path, resize=-1):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # img[:, :, :3] = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    if resize > 0:
      img = cv2.resize(img, (resize, resize))
    img = linear_to_srgb(img)
    img = np.int32(img * 255)
    return img
  
  import cv2 

  folder_name = "test"
  i_file_name = -1
  for image in sorted(glob.glob(os.path.join(blenderdir+"/*.exr"))):
    if 'depth' in image or 'seg' in image:
      continue
    png_file_name = image.split('/')[-1].replace(".exr",'')

    # if os.path.isfile(f"{outdir}/{folder_name}/{png_file_name}.png"):
    #   continue
    # raise()
    i_file_name += 1
    # if i_file_name>=100 and i_file_name<105:
    #   folder_name = 'val'
    # elif i_file_name>=105:
    #   folder_name = 'test'
    im = load_rgb_exr(image,resize=800)
    cv2.imwrite(f"{outdir}/{folder_name}/{png_file_name}.png",im)
    print(im.shape,im.min(),im.max())
    break
    
  near = 1000000000
  far = -1000000000
  for image in sorted(glob.glob(os.path.join(blenderdir+"/*.depth.exr"))):
    depth = cv2.imread(image,  
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
    depth[depth>3.4028235e+37] = 0
    depth[depth<-3.4028235e+37] = 0 
    if depth[depth>0].min() < near:
      near = depth[depth>0].min()
    if depth[depth>0].max() > far:
      far = depth[depth>0].max()
  near = near - near*.2
  far = far + far*.2
  # print(near,far)
  # near = 0.01 
  # far  = 3
  # near, far = 0.3694779872894287, 1.2352559566497803

  # raise() 
  # blenderdir = blenderdir + "/mip"
  # dirs = [os.path.join(blenderdir, f) for f in os.listdir(blenderdir)]
  # dirs = [d for d in dirs if os.path.isdir(d)]
  
  dirs = [blenderdir+"/mip"]

  print(dirs)
  
  for basedir in dirs:
    print()
    # newdir = os.path.join(outdir, os.path.basename(basedir))
    newdir = basedir
    print('Converting from', outdir, 'to', outdir)
    convert_to_nerfdata(outdir, outdir, n_down, near_far = [near,far])


if __name__ == '__main__':
  app.run(main)
