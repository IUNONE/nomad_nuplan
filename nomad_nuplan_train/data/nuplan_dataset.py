import numpy as np
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import io
import lmdb
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

from nomad_nuplan_train.data.data_utils import to_local_coords

class Nuplan_Dataset(Dataset):
    def __init__(self, data_config: Dict[str, Any], model_config, split: str):
        self.split = split
        self.data_folder = data_config['data_folder']
        self.data_traj_folder = data_config['data_traj_folder']
        self.waypoint_spacing = data_config['waypoint_spacing']
        self.image_size = data_config['image_size']
        self.context_size = model_config['vision_encoder']['context_size']
        self.len_traj_pred = model_config['len_traj_pred']
        self.normalize = model_config['action_norm']
        self.transform = data_config["tranform"]

        # get all senario clip name to list [] : self.traj_names 
        traj_names_file = os.path.join(self.data_folder, split, "traj_names.txt")
        if os.path.exists(traj_names_file):
            with open(traj_names_file, "r") as f:
                file_lines = f.read()
                self.traj_names = file_lines.split("\n")
            if "" in self.traj_names:
                self.traj_names.remove("")
        else:
            raise FileNotFoundError(f"Traj split file {traj_names_file} not found")
        
        #--------------------------------------------------------------------------------------------

        # directly load data index from pkl
        index_to_data_path = os.path.join(self.data_folder, split, "traj_index.pkl")
        if os.path.exists(index_to_data_path):
            with open(index_to_data_path, "rb") as f:
                self.index_to_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"Traj index file {index_to_data_path} not found")
        
        # directly load img from lmdb
        cache_filename = os.path.join(self.data_folder, split, "img_cache.lmdb")
        if os.path.exists(cache_filename):
            self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)
        else:
            raise FileNotFoundError(f"Img file {cache_filename} not found")
        
        self.trajectory_cache = {}  # to cache traj in each scenario clip

    def _load_image_from_scenario_T(self, trajectory_name, timestamp):
        '''
            load image in scenario trajectory_name at timestamp
            return a Tensor ( img resize to self.image_size) [C, H, W], where H,W = self.image_size
        '''
        img_path = os.path.join(trajectory_name, f"{str(timestamp)}.jpg")
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(img_path.encode())
                image_bytes = bytes(image_buffer)
            img = Image.open(io.BytesIO(image_bytes)).resize(self.image_size)
            return TF.to_tensor(img)
        except TypeError:
            print(f"Failed to load image {img_path} from lmdb cache file")

    def _get_waypoints(self, traj_data, curr_time):
        '''
            Args:
                traj_data : dict of position, yaw, timestamp [150, 2], [150,], [150,]
                curr_time : int
            Return:
                future_waypoints : [len_traj_pred, 2]
                goal_pos : [1, 2]
        '''
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]             # [self.len_traj_pred+1,]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]  # [self.len_traj_pred+1, 2]
        goal_pos = traj_data["position"][-1].reshape((1,2))                        # [1,2]                                           

        assert yaw.shape == (self.len_traj_pred + 1,)
        assert positions.shape == (self.len_traj_pred + 1, 2)
        assert goal_pos.shape == (1,2)

        # to current frame local coord
        future_waypoints = to_local_coords(positions, positions[0], yaw[0])[1:]
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0]).reshape(2)
        assert future_waypoints.shape == (self.len_traj_pred, 2)
        assert goal_pos.shape == (2,)

        if self.normalize:
            future_waypoints[:, :2] /= self.waypoint_spacing
            goal_pos /= self.waypoint_spacing
        
        return future_waypoints, goal_pos
    
    def _get_traj_from_pkl(self, trajectory_name):
        '''
            return traj data [dict] in a scenario clip (trajectory_name) from 'traj_data.pkl'
        '''
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_traj_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors
                obs_image (torch.Tensor): [C*(context+1), H, W] containing the image of the robot's observation in past 20ms and current frame
                goal_pos (torch.Tensor): [1, 2] containing the goal position in current frame local coordinate
                action_label (torch.Tensor): [len_traj_pred, 2] containing the waypoints in future 32ms
                dist_label (torch.Tensor):[1, ] containing the distance labels from the observation to the goal 
        """
        traj_name, timestamp = self.index_to_data[i]

        imgs_timestamps = list(range(timestamp + -self.context_size * self.waypoint_spacing,timestamp + 1,self.waypoint_spacing,))
        imgs_list = [self._load_image_from_scenario_T(f, t) for f, t in [(traj_name, t) for t in imgs_timestamps]] # list of [ （C,H,W）xN ]
        imgs_list = self._transform(imgs_list) # list of [ （C,H,W）xN ]
        obs_image = torch.cat(imgs_list)  # ---> cat to [C*N, H, W]
        curr_traj_data = self._get_traj_from_pkl(traj_name)
        future_waypoints, goal_pos = self._get_waypoints(curr_traj_data, timestamp)
        distance = (len(curr_traj_data) - timestamp - 1) // self.waypoint_spacing
        assert (len(curr_traj_data)- timestamp -1) % self.waypoint_spacing == 0

        # training_step: obs_image, goal_pos, future_waypoints, distance = batch
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(future_waypoints, dtype=torch.float32),
            torch.as_tensor(distance, dtype=torch.int64),    
        )
    
    def _transform(self, obs_image):
        
        if 'norm' in self.transform:
            transform = ([
                transforms.Normalize(self.transform['norm']['mean'], self.transform['norm']['std']),
            ])

        if transform is not None:
            transform = transforms.Compose(transform)
            # obs_images list of [C, H, W] * N
            return [transform(img) for img in obs_image]
        else:
            return obs_image

        

