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
  print('Loading imgs')
  for frame in meta['frames']:
    fname = os.path.join(data_dir, frame['file_path'] + '.png')
    with open(fname, 'rb') as imgin:
      image = np.array(Image.open(imgin), dtype=np.float32) / 255.
    cams.append(frame['transform_matrix'])
    images.append(image)
  ret = {}
  ret['images'] = np.stack(images, axis=0)
  print('Loaded all images, shape is', ret['images'].shape)
  ret['camtoworlds'] = np.stack(cams, axis=0)
  w = ret['images'].shape[2]
  camera_angle_x = float(meta['camera_angle_x'])
  ret['focal'] = .5 * w / np.tan(.5 * camera_angle_x)
  return ret


def down2(img):
  sh = img.shape
  return np.mean(np.reshape(img, [sh[0] // 2, 2, sh[1] // 2, 2, -1]), (1, 3))


def convert_to_nerfdata(basedir, newdir, n_down, near_far = [2.,6.]):
  """Convert Blender data to multiscale."""
  if not os.path.exists(newdir):
    os.makedirs(newdir)
  splits = ['train', 'val', 'test']
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
  outdir = blenderdir + '/' + outdir 
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

  out = out_train
  folder_name = 'train'
  for i_file_name, file_name in enumerate(sorted(glob.glob(blenderdir+"/*.json"))):
    if i_file_name>=100 and i_file_name<105:
      folder_name = 'val'

      out = out_val
    elif i_file_name>=105:
      out = out_test
      folder_name = 'test'



    c2w=[]
    print(i_file_name,file_name)
    with open(file_name) as json_file:
        data = json.load(json_file)
        c2w = data["camera_data"]["cam2world"]        
    fr={}
    fr["file_path"]=folder_name+"/"+"{num:05d}".format(num=i_file_name)
    fr["transform_matrix"]=[[c2w[0][0],c2w[1][0],c2w[2][0],c2w[3][0]*scale],
                            [c2w[0][1],c2w[1][1],c2w[2][1],c2w[3][1]*scale],
                            [c2w[0][2],c2w[1][2],c2w[2][2],c2w[3][2]*scale],
                            [c2w[0][3],c2w[1][3],c2w[2][3],c2w[3][3]]]  
    out['frames'].append(fr)

  out_train['camera_angle_x']=0.785398
  out_test['camera_angle_x'] = out_train['camera_angle_x']
  out_val['camera_angle_x'] = out_train['camera_angle_x']


  with open(f'{outdir}/transforms_train.json', 'w') as outfile:
      json.dump(out_train, outfile, indent=2)
  with open(f'{outdir}/transforms_val.json', 'w') as outfile:
      json.dump(out_val, outfile, indent=2)
  with open(f'{outdir}/transforms_test.json', 'w') as outfile:
      json.dump(out_test, outfile, indent=2)

  # make the png files

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

  folder_name = "train"
  i_file_name = -1
  for image in sorted(glob.glob(os.path.join(blenderdir+"/*.exr"))):
    if 'depth' in image or 'seg' in image:
      continue
    png_file_name = image.split('/')[-1].replace(".exr",'')

    if os.path.isfile(f"{outdir}/{folder_name}/{png_file_name}.png"):
      continue
    # raise()
    i_file_name += 1
    if i_file_name>=100 and i_file_name<105:
      folder_name = 'val'
    elif i_file_name>=105:
      folder_name = 'test'
    im = load_rgb_exr(image,resize=800)
    cv2.imwrite(f"{outdir}/{folder_name}/{png_file_name}.png",im)
    # print(im.shape,im.min(),im.max())

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
  print(near,far)
  # near = 0.3694779872894287 
  # far  = 1.2352559566497803

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
    print('Converting from', basedir, 'to', newdir)
    convert_to_nerfdata(basedir, newdir, n_down, near_far = [near,far])


if __name__ == '__main__':
  app.run(main)
