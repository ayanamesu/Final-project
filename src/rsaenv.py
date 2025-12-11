#custom environment code
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import csv
import os

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

                # release wavelength on every hop
                for u, v in zip(path, path[1:]):
                    state = self._get_link_state(u, v)

                    if wl in state.lightpaths:
                        # remove expired reservations
                        state.lightpaths[wl] = [t for t in state.lightpaths[wl] if t > now]

                        # if empty: wavelength becomes free
                        if len(state.lightpaths[wl]) == 0:
                            state.wavelengths[wl] = False

                to_remove.append(rid)

        # remove expired allocations
        for rid in to_remove:
            del self.active_lightpaths[rid]


    #-------------------------------------------------------------------------------------------
    def _allocate(self, action: int):
        req = self.current_request
        src, dst = req.source, req.destination

        # choose path for this request
        path = self._map_action_to_path(src, dst, action)
        if path is None:
            return False

        # find wavelength
        wl = self._first_fit(path)
        if wl is None:
            return False

        expiration_time = self.step_idx + req.holding_time
        rid = self.req_counter
        self.req_counter += 1

        # allocate wavelength on every hop
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            state = self._get_link_state(u, v)

            state.wavelengths[wl] = True
            state.lightpaths.setdefault(wl, []).append(expiration_time)

        self.active_lightpaths[rid] = (path, wl, expiration_time)
        return True


    #-------------------------------------------------------------------------------------------
    # Find wavelength available on ALL links in path (first-fit)
    #-------------------------------------------------------------------------------------------
    def _first_fit(self, path):
        """Return smallest wavelength index free on ALL links of the path."""
        for wl in range(self.capacity):
            ok = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                state = self._get_link_state(u, v)

                if state.wavelengths[wl]:  # True = occupied
                    ok = False
                    break

            if ok:
                return wl

        return None

    #-------------------------------------------------------------------------------------------
# Map action → actual path for the current request
#-------------------------------------------------------------------------------------------
    def _map_action_to_path(self, src, dst, action):
        """
        Given (src, dst) and an action index 0–7, return the corresponding path list.
        Returns None if action is invalid for that request.
        """

        # Case: 0 → 3
        if (src, dst) == (0, 3):
            if action < 2:
                return self.paths[(0, 3)][action]

        # Case: 0 → 4
        elif (src, dst) == (0, 4):
            if 2 <= action < 4:
                return self.paths[(0, 4)][action - 2]

        # Case: 7 → 3
        elif (src, dst) == (7, 3):
            if 4 <= action < 6:
                return self.paths[(7, 3)][action - 4]

        # Case: 7 → 4
        elif (src, dst) == (7, 4):
            if 6 <= action < 8:
                return self.paths[(7, 4)][action - 6]

        return None

#-------------------------------------------------------------------------------------------
# Return the link-state object for edge (u, v)
#-------------------------------------------------------------------------------------------
    def _get_link_state(self, u, v):
        """Return link state regardless of edge direction."""
        if self.graph.has_edge(u, v):
            return self.graph[u][v]["state"]
        elif self.graph.has_edge(v, u):
            return self.graph[v][u]["state"]
        else:
            raise KeyError(f"Edge ({u}, {v}) not found in graph")

#-------------------------------------------------------------------------------------------
# get observation
#-------------------------------------------------------------------------------------------

    def _obs(self):
        obs = []

        # --- PART 1: link utilizations (12 values) ---
        # utilization = (# wavelengths IN USE) / capacity
        for u, v, data in sorted(self.graph.edges(data=True)):
            used = 0
            for wl in range(self.capacity):
                if data["state"].wavelengths[wl]:
                    used += 1
            utilization = used / self.capacity
            obs.append(utilization)

        # --- PART 2: available wavelengths per link (12 values) ---
        for u, v, data in sorted(self.graph.edges(data=True)):
            free = 0
            for wl in range(self.capacity):
                if not data["state"].wavelengths[wl]:
                    free += 1
            obs.append(free / self.capacity)   # normalized

        # --- PART 3: current request features (3 values) ---
        req = self.current_request
        obs.append(req.source / 8.0)
        obs.append(req.destination / 8.0)
        obs.append(min(req.holding_time / 100.0, 1.0))

        # --- PART 4: path availability mask (8 binary values) ---
        # This is important and speed up learning significantly. It tells the agent which actions are valid.
        # 1.0 means path is freely available (at least one wavelength free),
        # 0.0 means path is not available (no wavelength free) or action is invalid for this (src, dst).

        for action in range(8):
            path = self._map_action_to_path(req.source, req.destination, action)

            if path is None:
                obs.append(0.0)
                continue

            wl = self._first_fit(path)
            obs.append(1.0 if wl is not None else 0.0)

        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        total = self.blocked + self.success
        blocking_rate = self.blocked / total if total > 0 else 0.0

        return {
            "step": self.step_idx,
            "blocked": self.blocked,
            "successful": self.success,
            "blocking_rate": blocking_rate,
            "active_lightpaths": len(self.active_lightpaths),
            "current_request": self.current_request
        }

    def render(self, mode='human'):
        if mode != 'human':
            return

        info = self._get_info()

        print("\n=== Step", info["step"], "===")

        if self.current_request:
            print(f"Current Request: {self.current_request}")

        print(f"Blocked: {info['blocked']}, Successful: {info['successful']}")
        print(f"Active lightpaths: {info['active_lightpaths']}")
        print(f"Blocking rate: {info['blocking_rate']:.3f}")


