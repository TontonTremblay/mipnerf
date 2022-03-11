import os
import sys
import shutil
import math
from pathlib import Path, PurePath, PurePosixPath
import numpy as np
import pyexr as exr
import csv
from scipy.ndimage.filters import convolve1d
import glob
from PIL import Image

# import flip
# import flip.utils

PAPER_FOLDER = Path(__file__).resolve().parent.parent
SUPPL_FOLDER = PAPER_FOLDER/'supplemental'
SCRIPTS_FOLDER = PAPER_FOLDER/'scripts'
TEMPLATE_FOLDER = SCRIPTS_FOLDER/'template'
DATA_FOLDER = SCRIPTS_FOLDER/'data'

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
NERF_DATA_FOLDER = os.environ.get('NERF_DATA_FOLDER') or "E:\\Projects\\nerf\\data"

# Search for neural_graphics_primitives in the build folder.
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build", "**/*.pyd"), recursive=True)]
# TODO why do I get a .so not a .pyd on my linux build? hmm
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build", "**/*.so"), recursive=True)]


def generate_crops_for_latex(img, path, crops, exposure=0, gamma=2.2):
	basepath, ext = os.path.splitext(path)

	for crop in crops:
		cropped_img = img[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2],...]
		crop_basepath = f"{basepath}-{ext[1:]}-{crop[2]}x{crop[3]}+{crop[0]}+{crop[1]}-crop"
		save_as_ldr(cropped_img, crop_basepath + ext, exposure=exposure, gamma=gamma)


def figure_from_template(template_file, variables):
	env = jinja2.Environment(
		block_start_string = '\BLOCK{',
		block_end_string = '}',
		variable_start_string = '\VAR{',
		variable_end_string = '}',
		comment_start_string = '\#{',
		comment_end_string = '}',
		line_statement_prefix = '%%',
		line_comment_prefix = '%#',
		trim_blocks = True,
		autoescape = False,
		loader = jinja2.FileSystemLoader(str(TEMPLATE_FOLDER))
	)
	template = env.get_template(template_file)
	preamble = \
f'''%!TEX root = paper.tex
% DO NOT EDIT---This file is generated using {sys.argv[0]}
'''
	return preamble + template.render(variables)


def sanitize_path(path):
	return str(PurePosixPath(path.relative_to(PAPER_FOLDER)))

# from https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
def trapez(y,y0,w):
	return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)
def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
	# The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
	# If either of these cases are violated, do some switches.
	if abs(c1-c0) < abs(r1-r0):
		# Switch x and y, and switch again when returning.
		xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
		return (yy, xx, val)

	# At this point we know that the distance in columns (x) is greater
	# than that in rows (y). Possibly one more switch if c0 > c1.
	if c0 > c1:
		return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

	# The following is now always < 1 in abs
	slope = (r1-r0) / (c1-c0)

	# Adjust weight by the slope
	w *= np.sqrt(1+np.abs(slope)) / 2

	# We write y as a function of x, because the slope is always <= 1
	# (in absolute value)
	x = np.arange(c0, c1+1, dtype=float)
	y = x * slope + (c1*r0-c0*r1) / (c1-c0)

	# Now instead of 2 values for y, we have 2*np.ceil(w/2).
	# All values are 1 except the upmost and bottommost.
	thickness = np.ceil(w/2)
	yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
	xx = np.repeat(x, yy.shape[1])
	vals = trapez(yy, y.reshape(-1,1), w).flatten()

	yy = yy.flatten()

	# Exclude useless parts and those outside of the interval
	# to avoid parts outside of the picture
	mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

	return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def diagonally_truncated_mask(shape, x_threshold, angle):
	result = np.zeros(shape, dtype=bool)
	for x in range(shape[1]):
		for y in range(shape[0]):
			thres = x_threshold * shape[1] - (angle * shape[0] / 2) + y * angle
			result[y, x, ...] = x < thres
	return result


def diagonally_combine_two_images(img1, img2, x_threshold, angle, gap=0, color=1):
	if img2.shape != img1.shape:
		raise ValueError(f"img1 and img2 must have the same shape; {img1.shape} vs {img2.shape}")
	mask = diagonally_truncated_mask(img1.shape, x_threshold, angle)
	result = img2.copy()
	result[mask] = img1[mask]
	if gap > 0:
		rr, cc, val = weighted_line(0, int(x_threshold * img1.shape[1] - (angle * img1.shape[0] / 2)), img1.shape[0]-1, int(x_threshold * img1.shape[1] + (angle * img1.shape[0] / 2)), gap)
		result[rr, cc, :] = result[rr, cc, :] * (1 - val[...,np.newaxis]) + val[...,np.newaxis] * color
	return result

def diagonally_combine_images(images, x_thresholds, angle, gap=0, color=1):
	result = images[0]
	for img, thres in zip(images[1:], x_thresholds):
		result = diagonally_combine_two_images(result, img, thres, angle, gap, color)
	return result

def prepare_folder(folder, backup=True):
	if backup:
		bup_folder = Path(folder.parent/(folder.name + "_backup"))
		if bup_folder.exists():
			shutil.rmtree(bup_folder, ignore_errors=False)
		if folder.exists():
			folder.rename(bup_folder)
	else:
		shutil.rmtree(folder, ignore_errors=False)
	folder.mkdir(parents=True)

def downsample(img, downsampling=1):
	return resize(img, (img.shape[0] // downsampling, img.shape[1] // downsampling))

def apply_color_map(img, exposure=-5.0, gamma=2.2):
	tonemapped = (2.0**exposure * img)**(1.0 / gamma)

	colormap = ScalarMappable(cmap="viridis")
	return colormap.to_rgba(tonemapped, norm=False)[:,:,:3]

def tonemap_scalar_raw(img, jpg_file, downsampling=1, crop=None, quality=95):
	img = np.clip(img, 0.0, None)
	if crop is not None:
		img = img[crop[0][0]:crop[1][0],crop[0][1]:crop[1][1],:]
	if downsampling != 1:
		img = downsample(img, downsampling)
	write_image_gamma(jpg_file, apply_color_map(img[:,:]), gamma=1.0, quality=quality)

def tonemap_scalar(exr_file, jpg_file, downsampling=1, crop=None, quality=95):
	img = read_image(exr_file).astype(np.float32)
	img = np.clip(img, 0.0, None)
	if crop is not None:
		img = img[crop[0][0]:crop[1][0],crop[0][1]:crop[1][1],:]
	if downsampling != 1:
		img = downsample(img, downsampling)
	write_image_gamma(jpg_file, apply_color_map(img[:,:,0]), gamma=1.0, quality=quality)

def tonemap_colored(exr_file, jpg_file, gamma=2.2, downsampling=1, crop=None, quality=95):
	img = read_image(exr_file).astype(np.float32)
	img = np.clip(img, 0.0, 1.0)
	if crop is not None:
		img = img[crop[0][0]:crop[1][0],crop[0][1]:crop[1][1],:]
	if downsampling != 1:
		img = downsample(img, downsampling)
	write_image_gamma(jpg_file, img, gamma=gamma, quality=quality)

def write_image_pillow(img_file, img, quality):
	img_array = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
	im = Image.fromarray(img_array)
	if os.path.splitext(img_file)[1] == ".jpg":
		im = im.convert('RGB') # Bake the alpha channel
	im.save(img_file, quality=quality, subsampling=0)

def read_image_pillow(img_file):
	img = Image.open(img_file, 'r').convert('RGB')
	img = np.asarray(img).astype(np.float32)
	return img / 255.0

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)



def read_image(file):
	if os.path.splitext(file)[1] == ".exr":
		img = exr.read(file).astype(np.float32)
	else:
		img = read_image_pillow(file)
		if img.shape[2] == 4:
			img[...,0:3] = srgb_to_linear(img[...,0:3])
			# Premultiply alpha
			img[...,0:3] *= img[...,3:4]
		else:
			img = srgb_to_linear(img)
	return img

def write_image(file, img, quality=95):
	if os.path.splitext(file)[1] == ".exr":
		img = exr.write(file, img)
	else:
		if img.shape[2] == 4:
			img = np.copy(img)
			img[...,0:3] = linear_to_srgb(img[...,0:3])
			# Unmultiply alpha
			img[...,0:3] = np.where(img[...,3:4] > 0, img[...,0:3] / img[...,3:4], 0)
		else:
			img = linear_to_srgb(img)
		write_image_pillow(file, img, quality)

def write_image_gamma(file, img, gamma, quality=95):
	if os.path.splitext(file)[1] == ".exr":
		img = exr.write(file, img)
	else:
		img = img**(1.0/gamma) # this will break alpha channels
		write_image_pillow(file, img, quality)

def trim(error, skip=0.000001):
	error = np.sort(error.flatten())
	size = error.size
	skip = int(skip * size)
	return error[skip:size-skip].mean()

def luminance(a):
	a = np.maximum(0, a)**0.4545454545
	return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

def SSIM(a, b):
	def blur(a):
		k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
		x = convolve1d(a, k, axis=0)
		return convolve1d(x, k, axis=1)
	a = luminance(a)
	b = luminance(b)
	mA = blur(a)
	mB = blur(b)
	sA = blur(a*a) - mA**2
	sB = blur(b*b) - mB**2
	sAB = blur(a*b) - mA*mB
	c1 = 0.01**2
	c2 = 0.03**2
	p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
	p2 = (2.0*sAB + c2)/(sA + sB + c2)
	error = p1 * p2
	return error

def L1(img, ref):
	return np.abs(img - ref)

def APE(img, ref):
	return L1(img, ref) / (1e-2 + ref)

def SAPE(img, ref):
	return L1(img, ref) / (1e-2 + (ref + img) / 2.)

def L2(img, ref):
	return (img - ref)**2

def RSE(img, ref):
	return L2(img, ref) / (1e-2 + ref**2)

def rgb_mean(img):
	return np.mean(img, axis=2)

def compute_error_img(metric, img, ref):
	img[np.logical_not(np.isfinite(img))] = 0
	img = np.maximum(img, 0.)
	if metric == "MAE":
		return L1(img, ref)
	elif metric == "MAPE":
		return APE(img, ref)
		# return APE(img, ref)
	elif metric == "SMAPE":
		return SAPE(img, ref)
	elif metric == "MSE":
		return L2(img, ref)
	elif metric == "MScE":
		return L2(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric == "MRSE":
		return RSE(img, ref)
		# return RSE(np.clip(img, 0, 100), np.clip(ref, 0, 100))
	elif metric == "MtRSE":
		return trim(RSE(img, ref))
	elif metric == "MRScE":
		return RSE(np.clip(img, 0, 100), np.clip(ref, 0, 100))
	elif metric == "SSIM":
		return SSIM(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric in ["FLIP", "\FLIP"]:
		# Set viewing conditions
		monitor_distance = 0.7
		monitor_width = 0.7
		monitor_resolution_x = 3840
		# Compute number of pixels per degree of visual angle
		pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

		ref_srgb = np.clip(flip.color_space_transform(ref, "linrgb2srgb"), 0, 1)
		img_srgb = np.clip(flip.color_space_transform(img, "linrgb2srgb"), 0, 1)
		result = flip.compute_flip(flip.utils.HWCtoCHW(ref_srgb), flip.utils.HWCtoCHW(img_srgb), pixels_per_degree)
		assert np.isfinite(result).all()
		return flip.utils.CHWtoHWC(result)

	raise ValueError(f"Unknown metric: {metric}.")

def compute_error(metric, img, ref, metric_map_filename=None,greater_zero=False,ignore_alpha=False):
	if greater_zero:
		img = img[ref>0]
		ref = ref[ref>0]
	if ignore_alpha: 
		img = img[ref[:,:,3]>0]
		ref = ref[ref[:,:,3]>0]

	metric_map = compute_error_img(metric, img, ref)
	metric_map[np.logical_not(np.isfinite(metric_map))] = 0
	if len(metric_map.shape) == 3:
		metric_map = np.mean(metric_map, axis=2)
	mean = np.mean(metric_map)
	if metric_map_filename:
		if not metric_map_filename.suffix.lower() == '.exr':
			index_map = np.clip(255.0 * np.squeeze(metric_map), 0, 255)
			metric_map = flip.utils.CHWtoHWC(flip.utils.index2color(index_map, flip.utils.get_magma_map()))
		exr.write(data=metric_map, filename=str(metric_map_filename))
	return mean


def to_precision(x,p):
	"""
	returns a string representation of x formatted with a precision of p

	Based on the webkit javascript implementation taken from here:
	https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
	"""

	x = float(x)

	if x == 0.:
		return "0." + "0"*(p-1)

	out = []

	if x < 0:
		out.append("-")
		x = -x

	e = int(math.log10(x))
	tens = math.pow(10, e - p + 1)
	n = math.floor(x/tens)

	if n < math.pow(10, p - 1):
		e = e -1
		tens = math.pow(10, e - p+1)
		n = math.floor(x / tens)

	if abs((n + 1.) * tens - x) <= abs(n * tens -x):
		n = n + 1

	if n >= math.pow(10,p):
		n = n / 10.
		e = e + 1

	m = "%.*g" % (p, n)

	if e < -2 or e >= p:
		out.append(m[0])
		if p > 1:
			out.append(".")
			out.extend(m[1:p])
		out.append('e')
		if e > 0:
			out.append("+")
		out.append(str(e))
	elif e == (p -1):
		out.append(m)
	elif e >= 0:
		out.append(m[:e+1])
		if e+1 < len(m):
			out.append(".")
			out.extend(m[e+1:])
	else:
		out.append("0.")
		out.extend(["0"]*-(e+1))
		out.append(m)

	return "".join(out)


class TempCWD:
	"""Temporarily change the current working directory.

	This class can be used to change the CWD, and restore it upon exiting the `with` clause, or
	explicitly calling close().

	Usage:
	------

	>>> import os
	>>> os.makedirs('temp', exist_ok=True)
	>>> print(os.getcwd())
	>>> with TempCWD('temp'):
	>>>     print(os.getcwd())
	>>> print(os.getcwd())

	Terminal output:
	C:/Users/frousselle/source/repos/falcor
	C:/Users/frousselle/source/repos/falcor/temp
	C:/Users/frousselle/source/repos/falcor

	"""
	def __init__(self, folder):
		self.cwd = os.getcwd()
		os.chdir(folder)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.close()

	def close(self):
		if self.cwd is not None:
			os.chdir(self.cwd)
		self.cwd = None
