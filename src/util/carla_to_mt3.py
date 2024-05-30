import json
import torch
import numpy as np
import glob
import os
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

def extract_json(timestamp=str, n_batches=int, resolution: int=1, time_steps: int = 20):
     
    path_pattern = os.getcwd() + "\\" + timestamp
    matches = glob.glob(path_pattern)
    if matches:
        jsonl_file_path = matches[0]  
        file_pattern = f"{jsonl_file_path}\\*.jsonl"
        #json_file = glob.glob(file_pattern)[0]
        
        #For chosing random file in data_set folder
        json_files = glob.glob(file_pattern)
        json_file = glob.glob(file_pattern)[np.random.randint(len(json_files))]
        #json_file = 'C:\\Users\\David\\GIT\\MT3EOT\\test\\RSP9_2024-05-17_152503.jsonl'
    else:
        print("No matching .jsonl files found.")
    print('##### RADAR SPAWN POINT #####')
    print(json_file)
    infile = open(json_file, 'r')
    all_data = [json.loads(line) for line in infile]
    infile.close() 
    
    batch_detections = []
    batch_gt = []
    while len(batch_detections) !=2:
    

        rand_idx_start = np.random.randint(200, len(all_data) - time_steps + 1)
        #rand_idx_start = 4426
        if any(len(sublist) == 0 for i, sublist in enumerate(all_data[rand_idx_start: rand_idx_start+time_steps*resolution])): # If data contains empty timesteps
            continue
        else:
            basetime = all_data[rand_idx_start][1].get('t')
            data_set = all_data[rand_idx_start: rand_idx_start+time_steps*resolution]
        print(f'json index: {rand_idx_start}')
        id_map = {-1: -1}
        id_generator = count(0)

        tset_detections = []
        tset_gt = []
        for time_step_data in data_set:
            tstep_detections = []
            tstep_gt = []
            for detection in time_step_data:
                if detection == time_step_data[0]:
                    ids_gt = time_step_data[0].get('ids_truth')
                    bboxes_truth = time_step_data[0].get('bboxes_truth')
                    tstep_gt = [ids_gt, bboxes_truth]
                else:
                    # Extract the data directly, assuming these keys always exist
                    r, vr, phi = detection.get("r"), detection.get("vr"), detection.get("phi")
                    x, y, vx, vy = detection.get('pointcloudx'), detection.get('pointcloudy'), detection.get('vx'), detection.get('vy')
                    t = round(detection.get("t") - basetime, 3)
                    
                    id = detection.get('id', -1)
                    tag = detection.get('tag')

                    if tag != 14 and id > 0: # some props get positive IDs, can be more easily fixed below 
                        id = -id

                    if id > -1 and id not in id_map:
                        id_map[id] = next(id_generator)
                    mapped_id = id_map.get(id, -1) # gets -1 if not in id_map

                    tstep_detections.append([id, mapped_id, r, vr, phi, x, y, vx, vy, t])

            np.random.shuffle(tstep_detections) # Shuffles detections in each time step
            tset_detections.append(np.array(tstep_detections))
            tset_gt.append(tstep_gt)

        # Defining Resolution
        if resolution >1:
            
            concatenated_detection_arrays = []
            concatenated_gt_arrays = []
        
            for i in range(0, len(tset_detections), resolution):
                if i + resolution <= len(tset_detections):
                    for j in range(i+1, i + resolution):
                        detection_array = np.vstack((tset_detections[i], tset_detections[j]))
                        
                        ids_i, dict_i = tset_gt[i]
                        ids_j, dict_j = tset_gt[j]
                        # Combine the IDs and remove duplicates
                        combined_ids = list(set(ids_i + ids_j))
                        # Combine the dictionaries
                        combined_dict = dict_i.copy()
                        combined_dict.update(dict_j)
                        gt_array = [combined_ids, combined_dict]
                        
                    detection_array[:,-1] = detection_array[0,-1] # editiing last time step
                    concatenated_detection_arrays.append(detection_array)
                    concatenated_gt_arrays.append(gt_array)
        else:
            concatenated_detection_arrays = tset_detections
            concatenated_gt_arrays = tset_gt


        batch_detections.append(concatenated_detection_arrays)
        batch_gt.append(concatenated_gt_arrays)
        
        # Last five time steps have to have object 
        for i, b_gt in enumerate(batch_gt):
            for step in b_gt[-5:]: 
                if step[0] == [] and batch_detections:
                #if b_gt[-1][0] == []:
                    #if i == 0:
                    batch_detections.pop(i)
                    batch_gt.pop(i)
                    print(rand_idx_start)
                    break

    return batch_detections, batch_gt
"""if (batch_detections[0][-1][:,0]+1).any() and (batch_detections[1][-1][:,0]+1).any():
    object_in_last_time_step = True
else:
    object_in_last_time_step = False"""
def unpack_data(batch, gt_batch):

    # Training Data
    training = tuple()
    trajectories = tuple()
    umids = tuple()
    for train_set, gt_set in zip(batch, gt_batch):
        training_set = []
        trajectories_set = {}
        umids_set = []

        for tstep, tgtstep in zip(train_set, gt_set):
            for detection in tstep: 
                id, mapped_id, r, vr, phi, x, y, vx, vy, t = detection[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

                # training
                training_set.append([r, vr, phi, t])
                # trajectories
                mapped_id = int(mapped_id)
                if mapped_id >-1:
                    bbox = np.array(tgtstep[1].get(str(int(id))))
                    
                    bbox[:-1,1] = -bbox[:-1,1] # American coords system
                    #bbox[:-1,1] = -bbox[:-1,1] + 50 # CIOU loss cant handle negative values...
                    bb_flat = []
                    for i in bbox:
                        for j in i:
                            bb_flat.append(j)

                    bb_flat = np.append(bb_flat, t)
                    trajectories_set.setdefault(mapped_id, []).append(bb_flat)
                # unique_measurment_ids
                umids_set.append(mapped_id)
                
        for id in trajectories_set:
            trajectories_set[id] = np.array(trajectories_set[id]) # have to convert to ndarray

        #trajectories += (dict(sorted(trajectories_set.items())),) # might be a bad idea to order objects 
        trajectories += (trajectories_set,)
        training += (np.array(training_set),)
        umids += (np.array(umids_set),)

    # Labels
    labels_pv = tuple()
    labels_bb = tuple()
    ulids = tuple()
    for traj_set in trajectories:
        labelpv_set = []
        labelbb_set = []
        ulids_set = []
        for obj, values in traj_set.items():
            labelpv_set.append(values[-1][:-3]) # [-1 (last step)][:-1 (omit time and bb)]
            labelbb_set.append(values[-1][4:-1]) # [-1 (last step)][:-1 (omit time and bb)]
            ulids_set.append(obj)
        
        labels_pv += (np.array(labelpv_set),)
        labels_bb += (np.array(labelbb_set),)
        ulids += (np.array(ulids_set),)
    #run_plot(trajectories, labels, training, training_example_to_plot=0) #plot debugging

    return training, labels_pv, labels_bb, umids, ulids, trajectories
"""
while True:
    data, gt_data= extract_json("*testTD", n_batches=2, time_steps=20, resolution=2)
    training_data, labels, unique_measurement_ids, unique_label_ids, trajectories= unpack_data(data, gt_data)    

"""
"""
### test the plot
import random
def output_truth_plot(trajectories, ax, labels, batch, training_example_to_plot):
    # Get ground-truth, predicted state, and logits for chosen training example
    truth = labels[training_example_to_plot]

    # Plot xy position of measurements, alpha-coded by time
    measurements = batch[training_example_to_plot]
    colors = np.zeros((measurements.shape[0], 4))
    unique_time_values = np.array(sorted(list(set(measurements[:, 3].tolist()))))
    
    def f(t):
        #Exponential decay for alpha in time
        idx = (np.abs(unique_time_values - t)).argmin()
        return 1/1.2**(len(unique_time_values)-idx)
    
    colors[:, 3] = [f(t) for t in measurements[:, 3].tolist()]

    measurements_cpu = measurements
    #measurements_cpu[:, 0] += 2
    ax.scatter(measurements_cpu[:, 0]*np.cos(measurements_cpu[:, 2]),
               measurements_cpu[:, 0]*np.sin(measurements_cpu[:, 2]),
               marker='x', c=colors, zorder=np.inf)
    
    counter = 0
    for traj_key, traj_values in trajectories[training_example_to_plot].items():
        
        truth_to_plot = truth[counter]
        counter += 1
        
        if truth_to_plot[-2] != traj_values[-1][-3] or truth_to_plot[-1] != traj_values[-1][-2]:
            print('FALSE', [truth_to_plot[-2], traj_values[-1][-3]], [truth_to_plot[-1], traj_values[-1][-2]])
        else:
            print([truth_to_plot[-2], traj_values[-1][-3]], [truth_to_plot[-1], traj_values[-1][-2]])
        
        
        traj_cpx = (traj_values[:,0] + traj_values[:,2] + traj_values[:,4] + traj_values[:,6])/4 
        traj_cpy = (traj_values[:,1] + traj_values[:,3] + traj_values[:,5] + traj_values[:,7])/4 

        bbx = traj_values[-1][::2][:-2]
        bby = traj_values[-1][1::2][:-1]

        random_color = (random.random(), random.random(), random.random())
        ax.scatter(traj_cpx, traj_cpy, marker='D', color=random_color, s=2)
        #ax.scatter(bbx, bby, marker='*', color=random_color)
        points = np.array([ [bbx[0], bby[0]], [bbx[2], bby[2]], [bbx[3], bby[3]], [bbx[1], bby[1]] ])
        polygon = Polygon(points, closed=True, edgecolor=random_color, fill=None, linewidth=2)
    
        # Add the polygon to the plot
        ax.add_patch(polygon)

        # Plot ground-truth
        cpx = traj_cpx[-1]
        cpy = traj_cpy[-1]

        ax.plot(cpx, cpy, marker='D', color=random_color, label='Matched Predicted Object', markersize=5)

        # Plot velocity
        ax.arrow(cpx, cpy, truth_to_plot[-2], truth_to_plot[-1], color=random_color,
                    head_width=0.2, length_includes_head=True)
        


def run_plot(trajectories, labels, training_data, training_example_to_plot=0):
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    fig.canvas.setWindowTitle('Training Progress')
    #fig.figure(num='Training Progress')
    gs = GridSpec(2, 3, figure=fig)
    output_ax = fig.add_subplot(gs[:, 1:])
    output_ax.set_ylabel('Y')
    output_ax.set_xlabel('X')
    output_ax.set_aspect('equal', 'box')
    output_ax.cla()
    output_ax.grid('on')

    output_truth_plot(trajectories , output_ax, labels, batch=training_data, training_example_to_plot=0)
    plt.show()


while True:
    data, gt_data= extract_json("*test2", n_batches=2, time_steps=20)
    training_data, labels, unique_measurement_ids, unique_label_ids, trajectories= unpack_data(data, gt_data)    
 """
