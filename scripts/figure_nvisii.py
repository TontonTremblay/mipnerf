import nvisii

import numpy as np 
import open3d as o3d


opt = lambda: None
opt.spp = 4000 
opt.width = 1000
opt.height = 1000 
opt.noise = False
opt.out = '16_create_mesh_from_data_alex2.png'
opt.path_obj = 'testJ.ply'

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless = True, verbose = True,max_entities=1000000,max_transforms=1000000)

if not opt.noise is True: 
    nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create(
        name = "camera",  
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (150,150,100),
)
nvisii.set_camera_entity(camera)

nvisii.set_dome_light_intensity(1)
nvisii.set_dome_light_color([1,1,1])

# # # # # # # # # # # # # # # # # # # # # # # # #

# Although NVISII has official support for stl files through mesh.create_from_file,
# let's open the STL from another library, and use the create_from_data interface. 

# let load the object using open3d
mesh = o3d.io.read_triangle_mesh(opt.path_obj)

vertices = np.array(mesh.vertices)

mesh = nvisii.mesh.create_box(name='cube')
mat = nvisii.material.create(name='cube')
mat.set_base_color([0.9,0.1,0.9])
print(np.min(vertices[:,2]))
print(np.max(vertices[:,2]))
vertices = vertices[vertices[:,2]<170]
for i_vert, vert in enumerate(vertices):
    # if vert[2]>170:
    #     continue
    # print(vert)

    nvisii.entity.create(
        name = str(i_vert),
        mesh = mesh, 
        material = mat,
        transform = nvisii.transform.create(str(i_vert),
            position =[vert[0],vert[1],vert[2]],
            scale = [0.45,0.45,0.45]
        )
    )

# # # # # # # # # # # # # # # # # # # # # # # # #

print(nvisii.get_scene_aabb_center())
print(nvisii.get_scene_min_aabb_corner())
print(nvisii.get_scene_max_aabb_corner())

# # # # # # # # # # # # # # # # # # # # # # # # #
pos = nvisii.get_scene_max_aabb_corner()
camera.get_transform().look_at(
    at = nvisii.get_scene_aabb_center(),
    up = (0,0,1),
    eye = (pos[0]*1.2,pos[1]*1.2,pos[2]*1.15),
)

print("at",nvisii.get_scene_aabb_center())
print('eye',(pos[0]*1.2,pos[1]*1.2,pos[2]*1.15))

nvisii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.out}"
)

# let's clean up the GPU
nvisii.deinitialize()