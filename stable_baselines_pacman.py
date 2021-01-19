# TODO: alphabetize imports 
import gym
from collections import OrderedDict
import os
import tensorflow as tf
import pandas as pd
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
import pyarrow as pa
import pyarrow.parquet as pq
from callback import CustomCallback
# env = make_atari('BreakoutNoFrameskip-v4')
env = make_atari('MsPacmanNoFrameskip-v4')

#get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

step_callback = CustomCallback()

# omit this and just pass in previously trained models
# model = DQN(CnnPolicy, env, verbose=1)
# # random agent
# model.learn(total_timesteps=25000, callback = step_callback) # 25000
# model.save("deepq_pacman_callback")

# del model # remove to demonstrate saving and loading

model = DQN.load("deepq_pacman_random") 
# model = DQN.load("deepq_pacman_callback") 
# TODO: pass model into here, change to args.agent_model
# model_path = os.path.join('models', 'MsPacman_5M_power_pill')
# model = DQN.load(model_path)

main_data_dict = OrderedDict()

directory = 'results/'
if not os.path.isdir(directory):
    os.mkdir(directory)

action_file = os.path.join(directory, 'actions.txt')
reward_file = os.path.join(directory, 'rewards.txt')
action_list = []
reward_list = []
df_list = []

# action names ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']

# dataframe is a db table 
def make_dataframes():
        # Make the main Dataframe
        main_df = pd.DataFrame.from_dict(main_data_dict, orient='index')

        # call to save last items as seperate df
        # self.save_last_line(args.stream_folder, main_df)
        
        # Now that we've parsed down the main df, load all into our list
        # of DFs and our list of Names
        df_list.append(main_df)


def df_to_csv(stream_directory):
        counter = 1
        for df in df_list:
            # str(df_names_list[counter-1])
            filename = "df" + str(counter) +  "temp.csv"
            filepath = os.path.join(stream_directory, filename)
            print("Making csvs and path is: ")
            print(filepath)
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', index=False, header=False)
            else:
                df.to_csv(filepath, mode='a', index=False)
            counter = counter + 1

def df_to_parquet(stream_directory):
        counter = 1
        for df in df_list:
            # str(self.df_names_list[counter-1])
            filename = "df" + str(counter) + "temp.parquet"
            filepath = os.path.join(stream_directory, filename)
            print("Making parquets and path is: ")
            print(filepath)
            table = pa.Table.from_pandas(df)
            # Parquet with Brotli compression
            pq.write_table(table, filepath, compression='BROTLI')
            counter = counter + 1

obs = env.reset()
reward = 0;
while True:
    action, _states = model.predict(obs)
    print("action taken ", action)
    # print("states ", _states) # why none??
    action_list.append(action)
    obs, rewards, dones, info = env.step(action)
    reward += rewards
    # lives = env.ale.lives()
    # print("lives remaining ", lives)
    # q values storing
    env.render()
    if(dones):
        reward_list.append(reward)
        break

# TODO: save data at each STEP, right now just record at end of one game

with open(action_file, "w") as text_file:
        text_file.write(str(action_list))

with open(reward_file, "w") as text_file:
        text_file.write(str(reward_list))

# TODO: add 'action_name': action_name, other fields
step_stats = { 
                
                'action': action,
                'reward': reward
            }

main_data_dict.update(step_stats)
make_dataframes()
df_to_csv(directory);
df_to_parquet(directory);
# test if parquet file is correctly created
print("reading parquet file", pd.read_parquet("results/df1temp.parquet"))
print("finished!!")
