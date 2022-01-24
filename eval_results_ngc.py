import subprocess
import glob 
import os

# path_data = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/falling_google_scenes/"
# path_checkpoints = "/home/jtremblay/code/ngc/results/google/"

# # print('hello')
# for i_folder,folder in enumerate(sorted(glob.glob(path_checkpoints + "*/"))):
# 	checkpoint = glob.glob(folder+"*/")[0]
# 	num = folder.split("/")[-2]
# 	subprocess.call(['mkdir',f'{checkpoint}/500k'])
# 	print(f'{checkpoint}/500k')
# 	subprocess.call(['cp',f'{checkpoint}/checkpoint_500000',f'{checkpoint}/500k/'])
# 	raise()
# 	if not os.path.exists(f"{path_data}/{str(num).zfill(5)}/mip/"):
# 		subprocess.call(['python',"scripts/convert_ndds_data.py",
# 			"--blenderdir",
# 			f"{path_data}/{str(num).zfill(5)}",
# 			'--outdir',
# 			f"{path_data}/{str(num).zfill(5)}/mip/"
# 			])

# 	subprocess.call([
# 		"python","eval.py",'--save_output',
# 		'--data_dir',f"{path_data}/{str(num).zfill(5)}/mip/",
# 		'--train_dir',checkpoint
# 		])
# 	raise()
# 	# if i_folder == 0 :
# 	# 	break

# # raise()

# path_data = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/nvisii_mvs_10/abc/"
# path_checkpoints = "/home/jtremblay/code/ngc/results/abc/"

# print('hello')
# for i_folder,folder in enumerate(sorted(glob.glob(path_checkpoints + "*/"))):
# 	checkpoint = glob.glob(folder+"*/")[0]
# 	num = folder.split("/")[-2]
# 	if not os.path.exists(f"{path_data}/{str(num).zfill(5)}/mip/"):
# 		subprocess.call(['python',"scripts/convert_ndds_data.py",
# 			"--blenderdir",
# 			f"{path_data}/{str(num).zfill(5)}",
# 			'--outdir',
# 			f"{path_data}/{str(num).zfill(5)}/mip/"
# 			])

# 	subprocess.call([
# 		"python","eval.py",'--save_output',
# 		'--data_dir',f"{path_data}/{str(num).zfill(5)}/mip/",
# 		'--train_dir',checkpoint
# 		])


# path_data = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/nvisii_mvs_10/amazon_berkeley/"
# path_checkpoints = "/home/jtremblay/code/ngc/results/amazon/"

# print('hello')
# # for i_folder,folder in enumerate(sorted(glob.glob(path_checkpoints + "*/"))):
# for i_folder,folder in [[0,f"{path_checkpoints}/0/"],[2,f"{path_checkpoints}/2/"]]:
# 	checkpoint = glob.glob(folder+"*/")[0]
# 	num = folder.split("/")[-2]
# 	if not os.path.exists(f"{path_data}/{str(num).zfill(5)}/mip/"):
# 		subprocess.call(['python',"scripts/convert_ndds_data.py",
# 			"--blenderdir",
# 			f"{path_data}/{str(num).zfill(5)}",
# 			'--outdir',
# 			f"{path_data}/{str(num).zfill(5)}/mip/"
# 			])
# 	print(f"{path_data}/{str(num).zfill(5)}/mip/")
# 	print(checkpoint)
# 	# raise()
# 	subprocess.call([
# 		"python","eval.py",'--save_output',
# 		'--data_dir',f"{path_data}/{str(num).zfill(5)}/mip/",
# 		'--train_dir',checkpoint
# 		])

# raise()

# path_data = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/nvisii_mvs_10/lego/"
# path_checkpoints = "/home/jtremblay/code/ngc/results/lego/"

# print('hello')
# folders_lego = sorted(glob.glob(path_data + "*/"))
# for i_folder,folder in enumerate(sorted(glob.glob(path_checkpoints + "*/"))):
# 	checkpoint = glob.glob(folder+"*/")[0]
# 	# num = folder.split("/")[-2]
# 	if not os.path.exists(f"{folders_lego[i_folder]}/mip/"):
# 		subprocess.call(['python',"scripts/convert_ndds_data.py",
# 			"--blenderdir",
# 			f"{folders_lego[i_folder]}",
# 			'--outdir',
# 			f"{folders_lego[i_folder]}/mip/"
# 			])

# 	subprocess.call([
# 		"python","eval.py",'--save_output',
# 		'--data_dir',f"{folders_lego[i_folder]}/mip/",
# 		'--train_dir',checkpoint
# 		])


# XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.4/ 
# python eval.py --save_output 
# --train_dir ../ngc/results/google/0/2372503/ 
# --data_dir /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/falling_google_scenes/00000/mip/



# path_data = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/nvisii_mvs_10/lego/"
path_data = '/media/jtremblay/data_large/brick_zoom_levels/'
path_checkpoints = "/home/jtremblay/code/ngc/results/bricks_zoomed/"

print('hello')
folders_lego = sorted(glob.glob(path_data + "*/"))
for i_folder,folder in enumerate(sorted(glob.glob(path_checkpoints + "*/"))):
	checkpoint = glob.glob(folder+"*/")[0]
	num = folder.split("/")[-2]
	if not os.path.exists(f"{folders_lego[i_folder]}/mip/"):
		subprocess.call(['python',"scripts/convert_ndds_data_bricks_zoom.py",
			"--blenderdir",
			f"{folders_lego[i_folder]}",
			'--outdir',
			f"{folders_lego[i_folder]}/mip/"
			])

	subprocess.call([
		"python","eval.py",'--save_output',
		'--data_dir',f"{folders_lego[i_folder]}/mip/",
		'--train_dir',checkpoint
		])
	raise()