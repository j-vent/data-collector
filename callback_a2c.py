from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import OrderedDict
from ghost_tracker import GhostTracker
import colour_detection as cd
from stable_baselines.common.atari_wrappers import make_atari
import cv2 as cv
from PIL import Image as im 

class CustomCallbackA(BaseCallback):
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
    og_env = make_atari('MsPacmanNoFrameskip-v4')
    def __init__(self, verbose=0, env_actions=[], env=None, num_steps=10, dir='results/', isLives=False, og_env = ""):
        super(CustomCallbackA, self).__init__(verbose)
        self.actions = env_actions
        self.env = env
        self.num_steps = num_steps
        self.directory = dir
        self.isLives = isLives
        self.og_env = og_env.unwrapped
        print("num stepss", self.num_timesteps)
        print("game has lives? ", self.isLives)
        # env <MaxAndSkipEnv<NoopResetEnv<TimeLimit<AtariEnv<MsPacmanNoFrameskip-v4>>>>>
        print("dir ", self.directory)
        print("env", self.env)
        print("og_env", self.og_env)
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
            CustomCallbackA.main_data_dict, orient='index')

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
        episode_reward = 0
        total_reward = 0
        game_reward = 0
        print("in util func")
        for key, value in CustomCallbackA.main_data_dict.items():
            # ignore step reward because it doesn't do cumulative_reward
            # if(key < 2):
            #     CustomCallbackA.main_data_dict[key]['step_reward'] = value['cumulative_episode_reward']
            # else:
            #     if( CustomCallbackA.main_data_dict[key-1]['lives'] == 0):
            #         CustomCallbackA.main_data_dict[key]['step_reward'] = 0
            #     else:
            #         CustomCallbackA.main_data_dict[key]['step_reward'] = value['cumulative_episode_reward'] - \
            #             CustomCallbackA.main_data_dict[key-1]['cumulative_episode_reward']
                
            # episode_reward += CustomCallbackA.main_data_dict[key]['step_reward'] 
            episode_reward = value['cumulative_episode_reward']
            
            if(self.isLives):
                # game over (epoch)
                if(value['lives'] == 0):
                    # print("game over", key)
                    # not sure if this is correct
                    # total_reward += game_reward
                    # CustomCallbackA.main_data_dict[key]['game_reward'] = game_reward
                    # CustomCallbackA.main_data_dict[key]['total_life'] = total_life
                    # CustomCallbackA.main_data_dict[key]['episode_reward'] = episode_reward
                    # reset values
                    total_game += 1
                    # total_life += 1
                    steps_game = steps_life = 0
                    game_reward = 0
                    episode_reward = 0

                # lost a life (episode)
                # elif(value['lives'] != prev_life and prev_life != 0):
                # record BEFORE lives is decremented
                elif(key != self.num_steps and value['lives'] != CustomCallbackA.main_data_dict[key+1]['lives']):
                    # print("elif ", key)
                    # not sure if this is correct
                    CustomCallbackA.main_data_dict[key]['total_life'] = total_life
                    # CustomCallbackA.main_data_dict[key]['episode_reward'] = episode_reward
                    # game_reward += episode_reward
                    # total_reward += episode_reward
                    total_life += 1
                    # steps_game += steps_life
                    steps_life = 1
                    episode_reward = 0

                total_reward += episode_reward
                # normal step
                prev_life = value['lives']
                CustomCallbackA.main_data_dict[key]['steps_life'] = steps_life
                
                CustomCallbackA.main_data_dict[key]['steps_game'] = steps_game
                CustomCallbackA.main_data_dict[key]['total_game'] = total_game
                CustomCallbackA.main_data_dict[key]['total_reward'] = total_reward

                steps_life += 1
                steps_game += 1
        
            # find coordinates of pacman and ghosts
            # subfolder = os.path.join(self.directory, 'screen')
            # dir = self.directory.replace("/", "")
            # filepath = dir + "\screen\screenshot" + str(key) + ".png"

            # pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist = cd.find_all_coords(
            #     filepath)
            # CustomCallbackA.main_data_dict[key]['pacman_coord_x'] = pacman_coord[0]
            # CustomCallbackA.main_data_dict[key]['pacman_coord_y'] = pacman_coord[1]
            # CustomCallbackA.main_data_dict[key]['pink_ghost_coord_x'] = pink_ghost_coord[0]
            # CustomCallbackA.main_data_dict[key]['pink_ghost_coord_y'] = pink_ghost_coord[1]
            # CustomCallbackA.main_data_dict[key]['to_pink_ghost'] = to_pink_ghost
            # CustomCallbackA.main_data_dict[key]['red_ghost_coord_x'] = red_ghost_coord[0]
            # CustomCallbackA.main_data_dict[key]['red_ghost_coord_y'] = red_ghost_coord[1]
            # CustomCallbackA.main_data_dict[key]['to_red_ghost'] = to_red_ghost
            # CustomCallbackA.main_data_dict[key]['green_ghost_coord_x'] = green_ghost_coord[0]
            # CustomCallbackA.main_data_dict[key]['green_ghost_coord_y'] = green_ghost_coord[1]
            # CustomCallbackA.main_data_dict[key]['to_green_ghost'] = to_green_ghost
            # CustomCallbackA.main_data_dict[key]['orange_ghost_coord_x'] = orange_ghost_coord[0]
            # CustomCallbackA.main_data_dict[key]['orange_ghost_coord_y'] = orange_ghost_coord[1]
            # CustomCallbackA.main_data_dict[key]['to_orange_ghost'] = to_orange_ghost

            # CustomCallbackA.main_data_dict[key]['pill_one_eaten'] = pill_eaten[0]
            # CustomCallbackA.main_data_dict[key]['to_pill_one'] = pill_dist[0]
            # CustomCallbackA.main_data_dict[key]['pill_two_eaten'] = pill_eaten[1]
            # CustomCallbackA.main_data_dict[key]['to_pill_two'] = pill_dist[1]
            # CustomCallbackA.main_data_dict[key]['pill_three_eaten'] = pill_eaten[2]
            # CustomCallbackA.main_data_dict[key]['to_pill_three'] = pill_dist[2]
            # CustomCallbackA.main_data_dict[key]['pill_four_eaten'] = pill_eaten[3]
            # CustomCallbackA.main_data_dict[key]['to_pill_four'] = pill_dist[3]

        # TODO: display total reward somewhere?? 
    def total_episode_reward_logger(self, rew_acc, rewards, masks, writer, steps):
        """
        calculates the cumulated episode reward, and prints to tensorflow log the output
        :param rew_acc: (np.array float) the total running reward
        :param rewards: (np.array float) the rewards
        :param masks: (np.array bool) the end of episodes
        :param writer: (TensorFlow Session.writer) the writer to log to
        :param steps: (int) the current timestep
        :return: (np.array float) the updated total running reward
        :return: (np.array float) the updated total running reward
        """

        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                # summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                # writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    #summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    #writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

        return rew_acc
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # save screenshot to folder
        # subfolder = os.path.join(self.directory, 'screen')
        # filepath = os.path.join(
        #     subfolder, 'screenshot' + str(self.num_timesteps) + '.png')
        # self.env.ale.saveScreenPNG("obs_ale.png")
        # obs = []
        # what timestep you think
        print("timestep ",CustomCallbackA.step )
        # what timestep a2c learn is on 
        print("num timestep",self.num_timesteps )
        # print("locs ", self.locals)
        # works at timestep 10, num_timestep 40
        # if(CustomCallbackA.step >= 10):
        #     # img = self.locals['obs'][0]
        #     # print("rolloutsss ", self.locals['rollout'])
        #     obs, states, rewards, masks, actions, values, ep_infos, true_reward = self.locals['rollout']
        #     print("true rewards", true_reward)
        #     print("rewards ", rewards)
        #     print("masks", masks)
        #     print("ep info", ep_infos)
        #     episode_reward = [0,0,0,0]
        #     # # 4,5 are n_envs, n_steps respectively
        #     new_rew = self.total_episode_reward_logger(episode_reward,true_reward.reshape((4, 5)),
        #                                         masks.reshape((4, 5)),
        #                                         None, self.num_timesteps)
        #     print("rew!! ", new_rew)
            # print("rew!! ", true_reward)
        #     print("shape" , img.shape) 
        #     # print("type of array ", type(img))
        #     # img[:,:,0] = np.ones([5,5])*64/255.0
        #     # img[:,:,1] = np.ones([5,5])*128/255.0
        #     # img[:,:,2] = np.ones([5,5])*192/255.0

        #     # cv.imwrite('pacman_ss.jpg', img)
        #     # cv.imshow("image", img)
        # img = self.locals['obs'][0]
        # data = im.fromarray(img) 
       
        # print("rew ", self.env.step_wait())
        # print(self.locals)
        #print("rew ", self.env.get_original_reward())
        print("rew func ", self.env.get_episode_rewards())
        # print("other rew", self.env.episode_rewards )
        subfolder = os.path.join(self.directory, 'screen')
        # filepath = os.path.join(subfolder, 'screenshot' + str(self.num_timesteps) + '.png')
        filepath = os.path.join(subfolder) 
        # mpl.image.imsave(filepath, img)
        
        # print("locals ", self.locals)
            # obs = self.locals['obs']
            
        
        # val = self.env.get_images()
        # mpl.image.imsave("obs0.png", val[0])
        # mpl.image.imsave("obs1.png", val[1])
        # mpl.image.imsave("obs2.png", val[2])
        # mpl.image.imsave("obs3.png", val[3])

            # print("val ", val)
        obs = self.locals['obs']
        # print("rews ", self.num_timesteps , " ",self.locals['rewards'])
        # print("mbrews ", self.num_timesteps , " ",self.locals['mb_rewards'])
        obs = self.env.get_images()
        img_name = '_screenshot' + str(self.num_timesteps)
        mpl.image.imsave(filepath+"env_0"+img_name+"_0.png", obs[0])
        mpl.image.imsave(filepath+"env_1"+img_name+"_1.png", obs[1])
        mpl.image.imsave(filepath+"env_2"+img_name+"_2.png", obs[2])
        mpl.image.imsave(filepath+"env_3"+img_name+"_3.png", obs[3])
           
        # data.save(filepath) 
        # obs = cv.imread(filepath)   # reads an image in the BGR format
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   # BGR -> RGB
        # print("img ", obs)
            # save image to disk
            # cv.imwrite('pacman_ppo2.jpg', img)\

        # documentation says only with one env :(
        # episode_rewards, episode_lengths = evaluate_policy(self.model, self.env,
        #                                                        n_eval_episodes=250,
        #                                                        render=self.render,
        #                                                        deterministic=self.deterministic,
        #                                                        return_episode_rewards=True)
        # if(self.num_timesteps > 8):
        #     print("rollout", self.locals['rollout'])
        # episode_rewards is a list that gets appended per epoch
        # take the episode reward of the latest epoch

        # print("step: ", CustomCallbackA.step,  " rew: ", self.locals['episode_rewards'][-1])
        # if CustomCallbackA.step  % 1000 == 0:
        #     print("at step ", CustomCallbackA.step)
        #print("at step ", str(CustomCallbackA.step))
        #print("locals ", self.locals)
        #print("actions", self.locals['mb_actions'])
            #print("locals", self.locals)
        step_stats = {CustomCallbackA.step: {
            'action_env_0': self.locals['actions'][0],
            'action_env_1': self.locals['actions'][1],
            'action_env_2': self.locals['actions'][2],
            'action_env_3': self.locals['actions'][3],
            'action_name_env_0': self.actions[self.locals['actions'][0]],
            'action_name_env_1': self.actions[self.locals['actions'][1]],
            'action_name_env_2': self.actions[self.locals['actions'][2]],
            'action_name_env_3': self.actions[self.locals['actions'][3]],
            'cumulative_episode_reward_env_0': self.locals['rewards'][0],
            'cumulative_episode_reward_env_1': self.locals['rewards'][1],
            'cumulative_episode_reward_env_2': self.locals['rewards'][2],
            'cumulative_episode_reward_env_3': self.locals['rewards'][3],
            'state': CustomCallbackA.step,
            #'lives':self.locals['infos']['ale.lives']
            }
        }

        # add step to dict
        CustomCallbackA.main_data_dict.update(step_stats)
        if(self.isLives == True and CustomCallbackA.step == 1):
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_0'] = 3
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_1'] = 3
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_2'] = 3
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_3'] = 3
        if(CustomCallbackA.step > 2):
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_0'] = self.locals['infos'][0]['ale.lives']
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_1'] = self.locals['infos'][1]['ale.lives']
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_2'] = self.locals['infos'][2]['ale.lives']
            CustomCallbackA.main_data_dict[CustomCallbackA.step]['lives_env_3'] = self.locals['infos'][3]['ale.lives']
        
        # # find coordinates of pacman and ghosts
        # key = CustomCallbackA.step
        # subfolder = os.path.join(self.directory, 'screen')
        # dir = self.directory.replace("/", "")
        # filepath = dir + "\screen\screenshot" + str(key) + ".png"

        
        # pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist, hasBlueGhost = cd.find_all_coords(
        #     filepath)
        # CustomCallbackA.main_data_dict[key]['pacman_coord_x'] = pacman_coord[0]
        # CustomCallbackA.main_data_dict[key]['pacman_coord_y'] = pacman_coord[1]
        # CustomCallbackA.main_data_dict[key]['pink_ghost_coord_x'] = pink_ghost_coord[0]
        # CustomCallbackA.main_data_dict[key]['pink_ghost_coord_y'] = pink_ghost_coord[1]
        # CustomCallbackA.main_data_dict[key]['to_pink_ghost'] = to_pink_ghost
        # CustomCallbackA.main_data_dict[key]['red_ghost_coord_x'] = red_ghost_coord[0]
        # CustomCallbackA.main_data_dict[key]['red_ghost_coord_y'] = red_ghost_coord[1]
        # CustomCallbackA.main_data_dict[key]['to_red_ghost'] = to_red_ghost
        # CustomCallbackA.main_data_dict[key]['green_ghost_coord_x'] = green_ghost_coord[0]
        # CustomCallbackA.main_data_dict[key]['green_ghost_coord_y'] = green_ghost_coord[1]
        # CustomCallbackA.main_data_dict[key]['to_green_ghost'] = to_green_ghost
        # CustomCallbackA.main_data_dict[key]['orange_ghost_coord_x'] = orange_ghost_coord[0]
        # CustomCallbackA.main_data_dict[key]['orange_ghost_coord_y'] = orange_ghost_coord[1]
        # CustomCallbackA.main_data_dict[key]['to_orange_ghost'] = to_orange_ghost

        # CustomCallbackA.main_data_dict[key]['pill_one_eaten'] = pill_eaten[0]
        # CustomCallbackA.main_data_dict[key]['to_pill_one'] = pill_dist[0]
        # CustomCallbackA.main_data_dict[key]['pill_two_eaten'] = pill_eaten[1]
        # CustomCallbackA.main_data_dict[key]['to_pill_two'] = pill_dist[1]
        # CustomCallbackA.main_data_dict[key]['pill_three_eaten'] = pill_eaten[2]
        # CustomCallbackA.main_data_dict[key]['to_pill_three'] = pill_dist[2]
        # CustomCallbackA.main_data_dict[key]['pill_four_eaten'] = pill_eaten[3]
        # CustomCallbackA.main_data_dict[key]['to_pill_four'] = pill_dist[3]

        # # find blue ghosts, if any
        # if(hasBlueGhost):
        #     imagePeeler = GhostTracker()
        #     # print("About to seek pacman at ", CustomCallbackA.step)
        #     # ghost_coords = imagePeeler.wheresPacman(obs)
        #     ghost_coords = imagePeeler.wheresPacman(self.locals['obs'])
        #     if(ghost_coords[0] != -1):
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost1_coord_x'] = ghost_coords[0]
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost1_coord_y'] = ghost_coords[1]
        #     if(ghost_coords[2] != -1):
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost2_coord_x'] = ghost_coords[2]
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost2_coord_y'] = ghost_coords[3]
        #     if(ghost_coords[4] != -1):
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost3_coord_x'] = ghost_coords[4]
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost3_coord_y'] = ghost_coords[5]
        #     if(ghost_coords[6] != -1):
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost4_coord_x'] = ghost_coords[6]
        #         CustomCallbackA.main_data_dict[key]['dark_blue_ghost4_coord_y'] = ghost_coords[7]

        #     print("ghost coords ", ghost_coords)

        # # remove screenshot
        # # if os.path.exists(filepath):
        # #     os.remove(filepath)
        # # else:
        # #     print("screenshot does not exist")

        
        # convert dict to different types
        # TODO: change to n envs
        if(CustomCallbackA.step == self.num_steps/4):
            # print("dictionary ", CustomCallbackA.main_data_dict)
            self.make_dataframes(self.df_list)
            self.df_to_csv("df_og.csv", self.df_list)
            self.df_to_parquet()
            # test if parquet file is correctly created
            # print("reading parquet file")
            # print(pd.read_parquet(os.path.join(self.directory,  "df.parquet")))

        #     # calculate new info
        #     #self.util()
        #     #self.make_dataframes(self.df_list_mod)
        #     #self.df_to_csv("df_mod.csv", self.df_list_mod)
        #     # self.df_to_parquet()
            print("done!")
        CustomCallbackA.step = CustomCallbackA.step + 1
        return True
