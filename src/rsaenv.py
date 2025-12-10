#custom environment code
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import csv

# import nwutil module for Request and generate_sample_graph
from nwutil import Request, generate_sample_graph
class RSAEnv(gym.Env):

    def __init__(self, capacity=20, data_dir='data/train', seed=None):

        # initializes the RSA env.

        super(RSAEnv, self).__init__()
        
        self.capacity = capacity
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed)

            # Defining paths
        self.paths = {
            # 0 -> 3
            (0, 3): 
                [[0, 1, 2, 3],          # P1: 3 jumps (shorter)
                [0, 8, 7, 6, 3]],        # P2: 4 jumps (longer , diverse route)
            # 0 -> 4
            (0, 4):
                [[0, 1, 5, 4],          # P3: 3 jumps (shorter)
                [0, 8, 7, 6, 3, 4]],     # P4: 5 jumps (longer, diverse route)
            # 7 -> 3
            (7, 3): 
                [[7, 1, 2, 3],          # P5: 3 jumps
                [7, 6, 3]],              # P6: 2 jumps (shortest path)
            # 7 -> 4
            (7, 4): 
                [[7, 1, 5, 4],          # P7: 3 jumps
                [7, 6, 3, 4]]           # P8: 3 jumps (via different intermediate nodes)
            
        }

        # ------------------------------------------------------------------------------------------
        # Load requests from CSV files in data/train directory
        #-------------------------------------------------------------------------------------------

        self.requests = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                self.requests += self._load_requests(os.path.join(self.data_dir, file))

        if len(self.requests) == 0:
            raise RuntimeError("No CSV files found in data/train")
        
        # ------------------------------------------------------------------------------------------
        # Generate network topology using generate_sample_graph()
        #-------------------------------------------------------------------------------------------

        self.graph = generate_sample_graph()

        for u, v, data in self.graph.edges(data=True):
            data['state'].capacity = self.capacity
            data['state'].wavelengths = [False] * self.capacity  # All wavelengths are free initially
            data['state'].utilization = 0.0
            data['state'].lightpaths = {}

        self.link_list = list(self.graph.edges())
        self.num_links = len(self.link_list)

        #-------------------------------------------------------------------------------------------
        # Define action and observation spaces
        #-------------------------------------------------------------------------------------------

        self.action_space = spaces.Discrete(8)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(35,), dtype=np.float32
        )    
        
        self.step_idx = 0
        self.active_lightpaths = {}
        self.req_counter = 0
        self.blocked = 0
        self.success = 0
        self.current_request = self.requests[0]

    #-------------------------------------------------------------------------------------------
    #CSV loader
    #------------------------------------------------------------------------------------------
    def _load_requests(self, filepath):
        reqs = []
        with open(filepath, "r") as f:
            for row in csv.DictReader(f):
                reqs.append(Request(
                    int(row["source"]),
                    int(row["destination"]),
                    int(row["holding_time"])
                ))
        return reqs
    
    #-------------------------------------------------------------------------------------------
    #Reset function
    #------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset links
        for u, v, data in self.graph.edges(data=True):
            data["state"].wavelengths = [False] * self.capacity
            data["state"].lightpaths = {}

        self.step_idx = 0
        self.req_counter = 0
        self.blocked = 0
        self.success = 0
        self.active_lightpaths.clear()

        # first request
        self.current_request = self.requests[0]
        return self._obs(), {}
    
    #-------------------------------------------------------------------------------------------
    # Step function
    #------------------------------------------------------------------------------------------
    def step(self, action: int):
        """
        One step = handle one request.
        Order:
        1) Release expired connections
        2) Attempt allocation
        3) Compute reward
        4) Advance request pointer
        5) Return (obs, reward, terminated, truncated, info)
        """

        # 1. Release expired lightpaths
        self._release_expired()

        # 2. Try to allocate current request
        success = self._allocate(action)

        # 3. Reward
        if success:
            reward = 0.0
            self.success += 1
        else:
            reward = -1.0
            self.blocked += 1

        # 4. Advance request pointer
        self.step_idx += 1
        terminated = (self.step_idx >= len(self.requests))

        if not terminated:
            self.current_request = self.requests[self.step_idx]

        # 5. Build observation + info
        obs = self._obs()
        info = {"blocked": self.blocked}

        return obs, reward, terminated, False, info
    
    #-------------------------------------------------------------------------------------------
    # Release expired lightpaths
    #-------------------------------------------------------------------------------------------
    def _release_expired(self):
        now = self.step_idx
        to_remove = []

        for rid, (path, wl, exp) in self.active_lightpaths.items():
            if now >= exp:
                # free wl on all links
                for u, v in zip(path, path[1:]):
                    state = self.graph[u][v]["state"]

                    # remove expired entries
                    if wl in state.lightpaths:
                        state.lightpaths[wl] = [t for t in state.lightpaths[wl] if t > now]

                        # if empty → wavelength becomes free
                        if len(state.lightpaths[wl]) == 0:
                            state.wavelengths[wl] = False

                to_remove.append(rid)

        for rid in to_remove:
            del self.active_lightpaths[rid]

    #-------------------------------------------------------------------------------------------
    def _allocate(self, action: int):
        """Allocate wavelength for current request given action."""

        req = self.current_request
        src, dst = req.source, req.destination

        # Map action → path
        path = self._map_action_to_path(src, dst, action)
        if path is None:
            return False

        # Find a wavelength available on ALL links in the path
        wl = self._first_fit(path)
        if wl is None:
            return False

        # Determine expiration time
        expiration_time = self.step_idx + req.holding_time
        request_id = self.req_counter
        self.req_counter += 1

        # Allocate wavelength on each link along the path (index version)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = self.graph[u][v]["state"]
            link_state.wavelengths[wl] = True
            link_state.lightpaths.setdefault(wl, []).append(expiration_time)

        # Track active allocation
        self.active_lightpaths[request_id] = (path, wl, expiration_time)

        return True
