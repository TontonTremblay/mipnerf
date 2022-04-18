import subprocess

path_data = '/media/jtremblay/data_large/brick_zoom_levels/'
path_checkpoints = "/home/jtremblay/code/ngc/results/bricks_zoomed/"

# print('hello')

# checkpoint = glob.glob(folder+"*/")[0]
# num = folder.split("/")[-2]
# if not os.path.exists(f"{folders_lego[i_folder]}/mip/"):
# 	subprocess.call(['python',"scripts/convert_ndds_data_bricks_zoom.py",
# 		"--blenderdir",
# 		f"{folders_lego[i_folder]}",
# 		'--outdir',
# 		f"{folders_lego[i_folder]}/mip/"
# 		])

subprocess.call([
	"python","eval.py",'--save_output',
	'--data_dir',f"/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/nvisii_mvs_10/lego/Night_Fury_Dragon_-_Lego_Elves_Style/mip/",
	# '--train_dir',"nightfury_16/"
	'--train_dir',"/home/jtremblay/code/ngc/results/lego/Night/2374376/"
	])
# raise()