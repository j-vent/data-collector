"""
    Generates a stream of gameplay for a given agent.

    A folder 'stream' is created whose subfolders contain all the states, visually displayed frames, Q-values,
    saliency maps and features (output of the second to last layer).

    At the very end *overlay_stream* is used to overlay each frame with a saliency map.
    This can also be redone later using *overlay_stream* to save time while trying different overlay styles.
"""


import gym
import matplotlib.pyplot as plt
from custom_atari_wrapper import atari_wrapper, AltRewardsWrapper
import numpy as np
import keras
from argmax_analyzer import Argmax
import overlay_stream
import pandas as pd
import sys
import scipy
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import OrderedDict
import os
# from varname import nameof
import coloredlogs, logging
from highlights_state_selection import compute_states_importance, read_input_files
#import h5py
import pyarrow as pa
import pyarrow.parquet as pq

class DataVault:
    #dictionaries for storing all the info which will be shoved into dataframes later
    main_data_dict = OrderedDict()
    q_values_dict = OrderedDict()
    stacked_bar_dict = OrderedDict()
    obs_ordered_dict = OrderedDict()
    argmax_ordered_dict = OrderedDict()
    per_episode_action_distribution_dict = {}
    df_list = []
    df_names_list = []
    # In order clockwise from (0,0) in upper left-hand corner of image
    pp_locs = [np.array([9,17]),np.array([149,16]),np.array([149,148]),np.array([9,149])]
    
    #keep track of steps
    step = 1

    def __init__(self):
        logger = logging.getLogger()
        coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
        logger.setLevel(logging.DEBUG)
        
    def print_df(self, stats_df):
        print("DF: ")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(stats_df)
            
    def df_to_csv(self, stream_directory):
        counter = 1
        for df in self.df_list:
            filename = "df" + str(counter) + str(self.df_names_list[counter-1]) + ".csv"
            filepath = os.path.join(stream_directory, filename)
            print("Making csvs and path is: ")
            print(filepath)
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', index=False, header=False)
            else:
                df.to_csv(filepath, mode='a', index=False)
            counter = counter + 1
            
    def df_to_parquet(self, stream_directory):
        counter = 1
        for df in self.df_list:
            filename = "df" + str(counter) + str(self.df_names_list[counter-1]) + ".parquet"
            filepath = os.path.join(stream_directory, filename)
            print("Making parquets and path is: ")
            print(filepath)
            table = pa.Table.from_pandas(df)
            # Parquet with Brotli compression
            pq.write_table(table, filepath, compression='BROTLI')
            counter = counter + 1
            
    def save_last_line(self, save_path, main_df):
#        save_path = os.path.abspath(os.path.join(save_path, os.pardir))
#        save_path = os.path.abspath(os.path.join(save_path, os.pardir))
        print("Saving last line and save path is: ")
        print(save_path)
        # Check if save path exists and handle
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)
        csv_path = os.path.join(save_path , 'CSVs')
        if (os.path.exists(csv_path) is False):
            os.mkdir(csv_path)
        print("Saving last line and CSV path is: ")
        print(csv_path)
        # get big dict and grab last line
#        last_main_df = main_df.iloc[-1]
        last_main_df = main_df.tail(1)
        self.print_df(last_main_df)
        # save out to file as "last_main.csv"
        last_main_path = os.path.join(csv_path , 'last_main.csv')
        last_main_df.to_csv(last_main_path)
        
        # add to end the last line of q_values
#        last_q_df = q_values_df.iloc[-1]
#        self.print_df(last_q_df)
#        # save out to file as "last_q.csv"
#        last_q_path = os.path.join(csv_path , 'last_q.csv')
#        last_q_df.to_csv(last_q_path, index=False)
#
#        # add to end last line of observations
#        last_obs_df = obs_df.iloc[-1]
#        self.print_df(last_obs_df)
#        # save out to file as "last_obs.csv"
#        last_obs_path = os.path.join(csv_path , 'last_obs.csv')
#        last_obs_df.to_csv(last_obs_path, index=False)
#
#        # add to end last line of argmax
#        last_argmax_df = argmax_values_df.iloc[-1]
#        self.print_df(last_argmax_df)
#        # save out to file as "last_argmax.csv"
#        last_argmax_path = os.path.join(csv_path , 'last_argmax.csv')
#        last_argmax_df.to_csv(last_argmax_path, index=False)
        
    def load_last_line(self, load_path):
#        load_path = os.path.abspath(os.path.join(load_path, os.pardir))
        load_path = os.path.abspath(os.path.join(load_path, os.pardir))
        # Check if save path exists and handle
        if os.path.exists(load_path) is False:
            return
        csv_path = os.path.join(load_path , 'CSVs')
        if (os.path.exists(csv_path) is False):
            return
            
        
#        q_csv_path = os.path.join(csv_path, "last_q.csv")
#        print("Loading last line and file path is: ")
#        print(q_csv_path)
#        # load path is where csv with last line is
#        if os.path.exists(q_csv_path) is False:
#            return
#        # load into a pandas dataframe
#        last_q_df = pd.read_csv(q_csv_path)
#        print("Got info on last QQQQQQQ line")
#        self.print_df(last_q_df)
#        # Use to update main_data_dict
#        cols = list(last_q_df.columns)
#        print(cols)
#        self.step = int(cols[0])
#
#        temp_dict = { self.step: last_q_df[str(self.step)][0] }
#        print(temp_dict)
#        self.q_values_dict.update(temp_dict)
#        print("Q values dict is now: ")
#        print(self.q_values_dict)
#
#
#
#
#        obs_csv_path = os.path.join(csv_path, "last_obs.csv")
#        print("Loading last line and file path is: ")
#        print(obs_csv_path)
#        # load path is where csv with last line is
#        if os.path.exists(obs_csv_path) is False:
#            return
#        # load into a pandas dataframe
#        else:
#            last_obs_df = pd.read_csv(obs_csv_path)
#            print("Got info on last line")
#            self.print_df(last_obs_df)
#        # Use to update main_data_dict
#        temp_dict = { self.step : last_obs_df[str(self.step)][0] }
        
        
        
        main_csv_path = os.path.join(csv_path, "last_main.csv")
        print("Loading last line and file path is: ")
        print(main_csv_path)
        # load path is where csv with last line is
        if os.path.exists(main_csv_path) is False:
            return
        # load into a pandas dataframe
        last_df = pd.read_csv(main_csv_path)
        print("Got info on last line of main dictionary!!")
        self.print_df(last_df)
        # Use to update main_data_dict
#        temp_dict = { self.step: }
#        temp_dict = last_df.to_dict('list')
#        print("List: ")
#        print(temp_dict)
#        temp_dict = last_df.to_dict('dict')
#        print("Dict: ")
#        print(temp_dict)
#        temp_dict = last_df.to_dict('series')
#        print("Series: ")
#        print(temp_dict)
        temp_dict = last_df.to_dict('index')
        print("Index: ")
        print(temp_dict)
        del(temp_dict[0]['Unnamed: 0'])
        print("New dict: ")
        print(temp_dict)
        temp_dict[self.step] = temp_dict.pop(0)
        print("After pop....")
        print(temp_dict)
#        temp_dict = last_df.to_dict('records')
#        print("Records: ")
#        print(temp_dict)
        self.main_data_dict.update(temp_dict)
        print(self.main_data_dict)

        
        # return action totals:
#        for action_number in range(len(action_total_sums)):
#            index_name = "action " + str(action_number) + " total sum"
#            action_total_sums[action_number] = self.main_data_dict[self.step][index_name]
#            print("Set index; ")
#            print(action_number)
#            print("to value; ")
#            print(self.main_data_dict[self.step][index_name])
#        print("New action total sums: ")
#        print(action_total_sums)
#        return action_total_sums

    def get_distances(self, characters_array, blue_ghost_array, pp, gg):
        bg = [0,0,0,0]
        print("Getting distance and ")
        print(np.sum(characters_array[0].current_coord))
        if (np.sum(characters_array[0].current_coord) != 0):
            print("Found non-0 coord for pacman")
            for index, pill in enumerate(self.pp_locs):
                print("Pill coord: " + str(pill))
                print("PM coord: " + str(characters_array[0].current_coord))
                pp[index] = np.abs(characters_array[0].current_coord - pill).sum()
                
                # Get distance to all normal ghosts
                if (np.sum(characters_array[index + 1].current_coord) != 0):
                    print("Found non-0 coord for normal ghost")
                    gg[index] = np.abs(characters_array[0].current_coord - characters_array[index + 1].current_coord).sum()
                
                # Get distance to all blue ghosts
                if (np.sum(blue_ghost_array[index].current_coord) != 0):
                    print("Found non-0 coord for blue ghost")
                    bg[index] = np.abs(characters_array[0].current_coord - blue_ghost_array[index].current_coord).sum()
        return pp, gg, bg
            
        
    def store_data(self, action, action_name, action_episode_sums, action_total_sums, reward, done, lives, mean_reward, characters, bgs, pill_eaten):
        logger = logging.getLogger()
        coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
        logger.setLevel(logging.DEBUG)
        #need to find some other way to store: Observations, argmax, features, q value
        
        end_of_episode = False
        end_of_epoch = False
        pp_dist = [154, 157,117,116]
        # order of list is red_ghost, pink_ghost, blue_ghost, orange_ghost
        # originally starts out 18 away from all of them, since all are in box
        ghost_dist = [18,18,18,18]
        blue_ghost_dist = [0,0,0,0]
        # If dataframe is not empty...
        if len(self.main_data_dict) != 0:
            #check contents of dictionary
#            logger.info("About to logger.info dictionary...")
#            for keys,values in self.main_data_dict.items():
#                logger.info(keys)
#                logger.info(values)
            lastElem = list(self.main_data_dict.keys())[-1]
#            logger.info("Last element is: " + str(lastElem))
            
            last_lives_left = self.main_data_dict[lastElem]['lives']
            episode = self.main_data_dict[lastElem]['episode']
            epoch = self.main_data_dict[lastElem]['epoch']
            episode_reward = self.main_data_dict[lastElem]['episode_reward'] + reward
            epoch_reward = self.main_data_dict[lastElem]['epoch_reward'] + reward
            total_reward = self.main_data_dict[lastElem]['total_reward'] + reward
            self.step = self.step + 1
            episode = self.main_data_dict[lastElem]['episode']
            epoch = self.main_data_dict[lastElem]['epoch']
            episode_step = self.main_data_dict[lastElem]['episode_step'] + 1
            epoch_step = self.main_data_dict[lastElem]['epoch_step'] + 1
        else:
            last_lives_left = 3
            episode = epoch = episode_step = epoch_step = self.step
            episode_reward = epoch_reward = total_reward = reward
            
        pp_dist, ghost_dist, blue_ghost_dist = self.get_distances(characters, bgs, pp_dist, ghost_dist)
            
        eoe_flag = False
        #first check if new episode or new epoch started
        if (lives != last_lives_left):
            eoe_flag = True
            episode_reward = reward
            episode = episode + 1
            episode_step = 1
            end_of_episode = True
            for x in range(len(action_episode_sums)):
                action_episode_sums[x] = 0
            
            if (done):
            # Have used up all three lives, therfore an "epoch" is over, and need to zero out accumulators
                pill_eaten = [False, False, False, False]
                epoch_reward = reward
                epoch = epoch + 1
                end_of_epoch = True
                epoch_step = 1
#                logger.info("end of episode and epoch is true ")
                eoe_flag = True
        
        # check if a pill was eaten
        for index, pill in enumerate(pp_dist):
            if pill_eaten[index] == False:
                print("Pill " + str(index+1) + " not recorded as eaten yet")
                print(pill)
                if pill < 5:
                    print("Setting pill " + str(index+1) + " to eaten")
                    pill_eaten[index] = True
                    
        # Up correct action sum
        temp_action_episode_sum = action_episode_sums[action]
        action_episode_sums[action] = temp_action_episode_sum + 1
        
        temp_action_total_sum = action_total_sums[action]
        action_total_sums[action] = temp_action_total_sum + 1
#        logger.info("end of episode and epoch is: ")
#        logger.info(end_of_episode)
#        logger.info(end_of_epoch)
        for index, pill in enumerate(pill_eaten):
            print("Just before recording and pill eaten is " + str(index+1) + " " + str(pill))
            
        step_stats = { self.step: {
            'action_name': action_name,
            'action': action,
            'reward': reward,
            'episode_reward': episode_reward,
            'epoch_reward': epoch_reward,
            'total_reward': total_reward,
            'lives': lives,
            'end_of_episode': end_of_episode,
            'end_of_epoch': end_of_epoch,
            'episode': episode,
            'episode_step': episode_step,
            'epoch': epoch,
            'epoch_step': epoch_step,
            'state': self.step,
            'mean_reward': mean_reward,
            'to_pill_one': pp_dist[0],
            'pill_one_eaten': pill_eaten[0],
            'to_pill_two': pp_dist[1],
            'pill_two_eaten': pill_eaten[1],
            'to_pill_three': pp_dist[2],
            'pill_three_eaten': pill_eaten[2],
            'to_pill_four': pp_dist[3],
            'pill_four_eaten': pill_eaten[3],
            # order of list is red_ghost, pink_ghost, blue_ghost, orange_ghost
            'to_red_ghost': ghost_dist[0],
            'to_pink_ghost': ghost_dist[1],
            'to_blue_ghost': ghost_dist[2],
            'to_orange_gohst': ghost_dist[3],
            'pacman_coord_x': characters[0].current_coord[0],
            'pacman_coord_y': characters[0].current_coord[1],
            'red_ghost_coord_x': characters[1].current_coord[0],
            'red_ghost_coord_y': characters[1].current_coord[1],
            'pink_ghost_coord_x': characters[2].current_coord[0],
            'pink_ghost_coord_y': characters[2].current_coord[1],
            'blue_ghost_coord_x': characters[3].current_coord[0],
            'blue_ghost_coord_y': characters[3].current_coord[1],
            'orange_ghost_coord_x': characters[4].current_coord[0],
            'orange_ghost_coord_y': characters[4].current_coord[1],
            'pacman_direction': characters[0].direction,
            'red_ghost_direction': characters[1].direction,
            'pink_ghost_direction': characters[2].direction,
            'blue_ghost_direction': characters[3].direction,
            'orange_ghost_direction': characters[4].direction,
            'dark_blue_ghost1_coord_x': bgs[0].current_coord[0],
            'dark_blue_ghost1_coord_y': bgs[0].current_coord[1],
            'dark_blue_ghost2_coord_x': bgs[1].current_coord[0],
            'dark_blue_ghost2_coord_y': bgs[1].current_coord[1],
            'dark_blue_ghost3_coord_x': bgs[2].current_coord[0],
            'dark_blue_ghost3_coord_y': bgs[2].current_coord[1],
            'dark_blue_ghost4_coord_x': bgs[3].current_coord[0],
            'dark_blue_ghost4_coord_y': bgs[3].current_coord[1]
            }
        }
        
#        step_q_values = { self.step: {
#            'q_values': np.squeeze(q_values)
#            }
#        }
#
#        step_observations = { self.step: {
#            'observation': np.squeeze(observation)
#            }
#        }
#
#        step_argmax = { self.step: {
#            'argmax': np.squeeze(argmax)
#            }
#        }
            
            
        # add carefully the action sums to the dictionary
        for action_number in range(len(action_episode_sums)):
#            logger.info("Action in list: ")
#            logger.info(action_number)
            index_name = "action " + str(action_number) + " episode sum"
            step_stats[self.step][index_name] = action_episode_sums[action_number]
            index_name = "action " + str(action_number) + " total sum"
            step_stats[self.step][index_name] = action_total_sums[action_number]
            
#            logger.info("About to logger.info dictionary after steps update...")
#            for keys,values in self.main_data_dict.items():
#                logger.info(keys)
#                logger.info(values)
            
                
    #    logger.info(step_stats)
        #add to the dictionary
        self.main_data_dict.update(step_stats)
#        self.q_values_dict.update(step_q_values)
#        self.obs_ordered_dict.update(step_observations)
#        self.argmax_ordered_dict.update(step_argmax)
#        logger.info("Updated main dictionary: ")
#        logger.info(self.main_data_dict)
        
        return (action_episode_sums, action_total_sums, pill_eaten)

    def make_dataframes(self, args):
        # Make the main Dataframe
        main_df = pd.DataFrame.from_dict(self.main_data_dict, orient='index')
        q_values_df = pd.DataFrame.from_dict(self.q_values_dict, orient='index')
        obs_df = pd.DataFrame.from_dict(self.obs_ordered_dict, orient='index')
        stacked_bar_df = pd.DataFrame.from_dict(self.per_episode_action_distribution_dict, orient='index')
        argmax_values_df = pd.DataFrame.from_dict(self.argmax_ordered_dict, orient='index')
        
        #call to save last items as seperate df
        self.save_last_line(args.stream_folder, main_df)
        
        # Now that we've parsed down the main df, load all into our list
        # of DFs and our list of Names
        self.df_list.append(main_df)
        self.df_names_list.append("main")
        
#        self.df_list.append(stacked_bar_df)
#        self.df_names_list.append("stacked_bar")
#
#        self.df_list.append(q_values_df)
#        self.df_names_list.append("q_values_df")
#
#        self.df_list.append(obs_df)
#        self.df_names_list.append("obs_df")
#
#        self.df_list.append(argmax_values_df)
#        self.df_names_list.append("argmax_df")
        
        print("Now DF list is length: " + str(len(self.df_list)))
        print("Now DF names list is length: " + str(len(self.df_names_list)))
