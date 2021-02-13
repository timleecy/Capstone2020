import os
import sys
import numpy as np

import test_vis as vis
import test_stats as stats

#ROHIT_BASE_PATH = '/playpen/ammirato/Data/RohitData/'
#ROHIT_BASE_PATH = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
ROHIT_BASE_PATH = 'C:/Capstone2020/ActiveVisionDataset/'


instance_ids = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,
                21,22,23,24,25,26,27,28]


all_scenes_list = [ 
                      'Home_001_1',
                      'Home_001_2',
                      'Home_002_1',
                      'Home_003_1',
                      'Home_003_2',
                      'Home_004_1',
                      'Home_004_2',
                      'Home_005_1',
                      'Home_005_2',
                      'Home_006_1',
                      'Home_008_1',
                      'Home_014_1',
                      'Home_014_2',
                      'Office_001_1'
]


function_name = sys.argv[1]
if len(sys.argv) > 2:
    scene_name = sys.argv[2]
    scene_path = os.path.join(ROHIT_BASE_PATH,scene_name)
    requested_instance = sys.argv[3]

instance_found = vis.check_present_instance(scene_path, requested_instance)
if not instance_found:
    print("Instance requested is not in the selected scene.")
    
else:
    #VISUALIZATIONS
    if function_name == 'vis_boxes_and_move':
        vis.vis_boxes_and_move(ROHIT_BASE_PATH, scene_path, requested_instance)
    elif function_name == 'vis_camera_pos_dirs':
        vis.vis_camera_pos_dirs(scene_path)
    
    #STATS
    elif function_name == 'get_instance_locations':
        stats.get_two_set_covers(ROHIT_BASE_PATH, instance_ids,
                                     scene_list=all_scenes_list)







