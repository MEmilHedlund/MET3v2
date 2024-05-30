from itertools import count
import numpy as np
import torch
from torch import Tensor
from util.misc import NestedTensor
from util.carla_to_mt3 import extract_json, unpack_data


class DataGenerator:
    def __init__(self, params, rngs=None):
        self.params = params
        assert 0 <= params.data_generation.n_prediction_lag <= params.data_generation.n_timesteps, "Prediction lag has to be smaller than the total number of time-steps."
        self.device = params.training.device
        self.n_timesteps = params.data_generation.n_timesteps

    def get_batch(self):  
        # Unpack results
        data, gt_data= extract_json("*testTD", n_batches=1, time_steps=20, resolution=2)
        training_data, labels_pv, labels_bb, unique_measurement_ids, unique_label_ids, trajectories= unpack_data(data, gt_data)
        
        labels_pv = [Tensor(l).to(torch.device(self.device)) for l in labels_pv]
        labels_bb = [Tensor(l).to(torch.device(self.device)) for l in labels_bb]
        trajectories = list(trajectories)
        unique_measurement_ids = [list(u) for u in unique_measurement_ids]
        unique_label_ids = list(unique_label_ids)

        # Pad training data
        max_len = max(list(map(len, training_data)))
        training_data, mask = pad_to_batch_max(training_data, max_len)

        # Pad unique ids
        for i in range(len(unique_measurement_ids)):
            unique_id = unique_measurement_ids[i]
            n_items_to_add = max_len - len(unique_id)
            unique_measurement_ids[i] = np.concatenate([unique_id, [-2] * n_items_to_add])[None, :]
        unique_measurement_ids = np.concatenate(unique_measurement_ids)

        training_nested_tensor = NestedTensor(Tensor(training_data).to(torch.device(self.device)),
                                              Tensor(mask).bool().to(torch.device(self.device)))
        unique_measurement_ids = Tensor(unique_measurement_ids).to(self.device)

        return training_nested_tensor, labels_pv, labels_bb, unique_measurement_ids, unique_label_ids, trajectories


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0

    return training_data_padded, mask