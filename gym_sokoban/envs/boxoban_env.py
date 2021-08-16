from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np
from gym.spaces import Box

class BoxobanEnv(SokobanEnv):
    num_boxes = 4
    dim_room=(10, 10)

    def __init__(self,
             max_steps=120,
             difficulty='unfiltered', split='train', render_mode='tiny_rgb_array', fix_room = False):
        self.difficulty = difficulty
        self.split = split
        self.verbose = False
        self.render_mode = render_mode
        self.fix_room = fix_room
        self.room_selected = False
        self.selected_map = None
        self.choice_room = False
        super(BoxobanEnv, self).__init__(self.dim_room, max_steps, self.num_boxes, None)
        if render_mode == 'tiny_rgb_array':
            self.observation_space = Box(low=0, high=255, shape=(self.dim_room[0], self.dim_room[1], 3), dtype=np.uint8)
        


    def reset(self):
        self.cache_path = '.sokoban_cache'
        self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

        if not os.path.exists(self.cache_path):
           
            url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
            
            if self.verbose:
                print('Boxoban: Pregenerated levels not downloaded.')
                print('Starting download from "{}"'.format(url))

            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

            os.makedirs(self.cache_path)
            path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
            with open(path_to_zip_file, 'wb') as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(self.cache_path)
            zip_ref.close()
        
        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        if self.render_mode == 'tiny_rgb_array':
            starting_observation = room_to_tiny_world_rgb(self.room_state, self.room_fixed)
        elif self.render_mode == 'tiny_rgb_array':
            starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return starting_observation

    def step(self, action, observation_mode='tiny_rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def select_room(self):
        if ((self.room_selected == False and self.fix_room == True) or self.fix_room == False) and self.choice_room == False:
            generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
            source_file = join(self.train_data_dir, random.choice(generated_files))

            maps = []
            current_map = []
            
            with open(source_file, 'r') as sf:
                for line in sf.readlines():
                    if ';' in line and current_map:
                        maps.append(current_map)
                        current_map = []
                    if '#' == line[0]:
                        current_map.append(line.strip())
            
            maps.append(current_map)

            self.selected_map = random.choice(maps)
            # print(self.selected_map)
            self.room_selected = True
        elif self.choice_room:
            self.selected_map = ['##########', '##########', '# ########', '#      #.#', '# $ $    #', '# .   $. #', '#   ##@###', '# $ .#####', '#    #####', '##########']
        
        if self.verbose:
            print('Selected Level from File "{}"'.format(source_file))

        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(self.selected_map)


    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping




ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
