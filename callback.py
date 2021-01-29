from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import OrderedDict
from tracker import Tracker
import colour_detection as cd


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    actions = []
    directory = 'results/'
    save_file_screen = os.path.join(directory, 'screen', 'screen')
    env = None
    step = 1
    main_data_dict = OrderedDict()
    df_list = []
    df_list_mod = []
    num_steps = 10
    isLives = False

    def __init__(self, verbose=0, env_actions=[], env=None, num_steps=10, dir='results/', isLives=False):
        super(CustomCallback, self).__init__(verbose)
        self.actions = env_actions
        self.env = env.unwrapped
        self.num_steps = num_steps
        self.directory = dir
        self.isLives = isLives
        print("game has lives? ", self.isLives)
        # env <MaxAndSkipEnv<NoopResetEnv<TimeLimit<AtariEnv<MsPacmanNoFrameskip-v4>>>>>
        print("dir ", self.directory)
        print("env", env.unwrapped)
        print("mod ",  self.model)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    # dataframe is a db table
    def make_dataframes(self, df):
        # Make the main Dataframe
        main_df = pd.DataFrame.from_dict(
            CustomCallback.main_data_dict, orient='index')

        # call to save last items as seperate df
        # self.save_last_line(args.stream_folder, main_df)

        # Now that we've parsed down the main df, load all into our list
        # of DFs and our list of Names
        # self.df_list.append(main_df)
        df.append(main_df)

    def df_to_csv(self, filename, df_list):
        for df in df_list:
            # filename = "df.csv"
            filepath = os.path.join(self.directory, filename)
            print("Making csvs and path is: ")
            print(filepath)
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', index=False, header=False)
            else:
                df.to_csv(filepath, mode='a', index=False)

    def df_to_parquet(self):
        for df in self.df_list:
            filename = "df.parquet"
            filepath = os.path.join(self.directory, filename)
            print("Making parquets and path is: ")
            print(filepath)
            table = pa.Table.from_pandas(df)
            # Parquet with Brotli compression
            pq.write_table(table, filepath, compression='BROTLI')

    def save_frame(self, array, save_file, frame):
        if not (os.path.isdir(save_file)):
            os.makedirs(save_file)
            os.rmdir(save_file)
        plt.imsave(save_file + '_' + str(frame) + '.png', array)

    def save_observations(self, observations):
        for i in range(len(observations)):
            index = str("_") + '_' + str(i)
            observation = observations[i]
            self.save_frame(observation, self.save_file_screen, index)

     # TODO: maybe move to a separate file
    def util(self):
        total_life = total_game = steps_life = steps_game = 1
        prev_life = 3

        print("in util func")
        for key, value in CustomCallback.main_data_dict.items():

            if(key < 2):
                CustomCallback.main_data_dict[key]['step_reward'] = value['episode_reward']
            else:
                CustomCallback.main_data_dict[key]['step_reward'] = value['episode_reward'] - \
                    CustomCallback.main_data_dict[key-1]['episode_reward']

            if(self.isLives):
                # game over (epoch)
                if(value['lives'] == 0):
                    # reset values
                    total_game += 1
                    total_life += 1
                    steps_game = steps_life = 1

                # lost a life (episode)
                elif(value['lives'] != prev_life and prev_life != 0):
                    total_life += 1
                    # steps_game += steps_life
                    steps_life = 1

                # normal step
                prev_life = value['lives']
                CustomCallback.main_data_dict[key]['steps_life'] = steps_life
                CustomCallback.main_data_dict[key]['total_life'] = total_life
                CustomCallback.main_data_dict[key]['steps_game'] = steps_game
                CustomCallback.main_data_dict[key]['total_game'] = total_game

                steps_life += 1
                steps_game += 1

            # find coordinates of pacman and ghosts
            # subfolder = os.path.join(self.directory, 'screen')
            # dir = self.directory.replace("/", "")
            # filepath = dir + "\screen\screenshot" + str(key) + ".png"

            # pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist = cd.find_all_coords(
            #     filepath)
            # CustomCallback.main_data_dict[key]['pacman_coord_x'] = pacman_coord[0]
            # CustomCallback.main_data_dict[key]['pacman_coord_y'] = pacman_coord[1]
            # CustomCallback.main_data_dict[key]['pink_ghost_coord_x'] = pink_ghost_coord[0]
            # CustomCallback.main_data_dict[key]['pink_ghost_coord_y'] = pink_ghost_coord[1]
            # CustomCallback.main_data_dict[key]['to_pink_ghost'] = to_pink_ghost
            # CustomCallback.main_data_dict[key]['red_ghost_coord_x'] = red_ghost_coord[0]
            # CustomCallback.main_data_dict[key]['red_ghost_coord_y'] = red_ghost_coord[1]
            # CustomCallback.main_data_dict[key]['to_red_ghost'] = to_red_ghost
            # CustomCallback.main_data_dict[key]['green_ghost_coord_x'] = green_ghost_coord[0]
            # CustomCallback.main_data_dict[key]['green_ghost_coord_y'] = green_ghost_coord[1]
            # CustomCallback.main_data_dict[key]['to_green_ghost'] = to_green_ghost
            # CustomCallback.main_data_dict[key]['orange_ghost_coord_x'] = orange_ghost_coord[0]
            # CustomCallback.main_data_dict[key]['orange_ghost_coord_y'] = orange_ghost_coord[1]
            # CustomCallback.main_data_dict[key]['to_orange_ghost'] = to_orange_ghost

            # CustomCallback.main_data_dict[key]['pill_one_eaten'] = pill_eaten[0]
            # CustomCallback.main_data_dict[key]['to_pill_one'] = pill_dist[0]
            # CustomCallback.main_data_dict[key]['pill_two_eaten'] = pill_eaten[1]
            # CustomCallback.main_data_dict[key]['to_pill_two'] = pill_dist[1]
            # CustomCallback.main_data_dict[key]['pill_three_eaten'] = pill_eaten[2]
            # CustomCallback.main_data_dict[key]['to_pill_three'] = pill_dist[2]
            # CustomCallback.main_data_dict[key]['pill_four_eaten'] = pill_eaten[3]
            # CustomCallback.main_data_dict[key]['to_pill_four'] = pill_dist[3]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        # save screenshot to folder
        subfolder = os.path.join(self.directory, 'screen')
        filepath = os.path.join(
            subfolder, 'screenshot' + str(CustomCallback.step) + '.png')
        self.env.ale.saveScreenPNG(filepath)
        
        # episode_rewards, episode_lengths = evaluate_policy(self.model, self.env,
        #                                                        n_eval_episodes=self.n_eval_episodes,
        #                                                        render=self.render,
        #                                                        deterministic=self.deterministic,
        #
        #                                                        return_episode_rewards=True)

        # episode_rewards is a list that gets appended per epoch
        # take the episode reward of the latest epoch

        # print("step: ", CustomCallback.step,  " rew: ", self.locals['episode_rewards'][-1])
        if(CustomCallback.step % 1000 == 0):
            print("at step ", str(CustomCallback.step))
        step_stats = {CustomCallback.step: {
            'action': self.locals['env_action'],
            'action_name': self.actions[self.locals['env_action']],
            'episode_reward': self.locals['episode_rewards'][-1],
            'state': CustomCallback.step
            # 'lives':self.locals['info']['ale.lives']
        }
        }

        # add step to dict
        CustomCallback.main_data_dict.update(step_stats)
        if(self.isLives == True):
            CustomCallback.main_data_dict[CustomCallback.step]['lives'] = self.locals['info']['ale.lives']

        
        # find coordinates of pacman and ghosts
        key = CustomCallback.step
        # subfolder = os.path.join(self.directory, 'screen')
        # dir = self.directory.replace("/", "")
        # filepath = dir + "\screen\screenshot" + str(key) + ".png"

        
        pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist = cd.find_all_coords(
            filepath)
        CustomCallback.main_data_dict[key]['pacman_coord_x'] = pacman_coord[0]
        CustomCallback.main_data_dict[key]['pacman_coord_y'] = pacman_coord[1]
        CustomCallback.main_data_dict[key]['pink_ghost_coord_x'] = pink_ghost_coord[0]
        CustomCallback.main_data_dict[key]['pink_ghost_coord_y'] = pink_ghost_coord[1]
        CustomCallback.main_data_dict[key]['to_pink_ghost'] = to_pink_ghost
        CustomCallback.main_data_dict[key]['red_ghost_coord_x'] = red_ghost_coord[0]
        CustomCallback.main_data_dict[key]['red_ghost_coord_y'] = red_ghost_coord[1]
        CustomCallback.main_data_dict[key]['to_red_ghost'] = to_red_ghost
        CustomCallback.main_data_dict[key]['green_ghost_coord_x'] = green_ghost_coord[0]
        CustomCallback.main_data_dict[key]['green_ghost_coord_y'] = green_ghost_coord[1]
        CustomCallback.main_data_dict[key]['to_green_ghost'] = to_green_ghost
        CustomCallback.main_data_dict[key]['orange_ghost_coord_x'] = orange_ghost_coord[0]
        CustomCallback.main_data_dict[key]['orange_ghost_coord_y'] = orange_ghost_coord[1]
        CustomCallback.main_data_dict[key]['to_orange_ghost'] = to_orange_ghost

        CustomCallback.main_data_dict[key]['pill_one_eaten'] = pill_eaten[0]
        CustomCallback.main_data_dict[key]['to_pill_one'] = pill_dist[0]
        CustomCallback.main_data_dict[key]['pill_two_eaten'] = pill_eaten[1]
        CustomCallback.main_data_dict[key]['to_pill_two'] = pill_dist[1]
        CustomCallback.main_data_dict[key]['pill_three_eaten'] = pill_eaten[2]
        CustomCallback.main_data_dict[key]['to_pill_three'] = pill_dist[2]
        CustomCallback.main_data_dict[key]['pill_four_eaten'] = pill_eaten[3]
        CustomCallback.main_data_dict[key]['to_pill_four'] = pill_dist[3]

        # remove ss
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            print("screenshot does not exist")

        
        # convert dict to different types
        if(CustomCallback.step == self.num_steps):
            # print("dictionary ", CustomCallback.main_data_dict)
            self.make_dataframes(self.df_list)
            self.df_to_csv("df_og.csv", self.df_list)
            self.df_to_parquet()
            # test if parquet file is correctly created
            # print("reading parquet file")
            # print(pd.read_parquet(os.path.join(self.directory,  "df.parquet")))

            # calculate new info
            self.util()
            self.make_dataframes(self.df_list_mod)
            self.df_to_csv("df_mod.csv", self.df_list_mod)
            # self.df_to_parquet()
            print("done!")
        CustomCallback.step = CustomCallback.step + 1
