from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import OrderedDict

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
    num_steps = 10
    counter = 1

    def __init__(self, verbose=0, env_actions=[], env =None, num_steps=10, dir = 'results/'):
        super(CustomCallback, self).__init__(verbose)
        self.actions = env_actions
        self.env = env.unwrapped
        self.num_steps = num_steps
        self.directory = dir
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
    def make_dataframes(self):
        # Make the main Dataframe
        main_df = pd.DataFrame.from_dict(CustomCallback.main_data_dict, orient='index')

        # call to save last items as seperate df
        # self.save_last_line(args.stream_folder, main_df)
        
        # Now that we've parsed down the main df, load all into our list
        # of DFs and our list of Names
        self.df_list.append(main_df)

    def df_to_csv(self):
        for df in self.df_list:
            filename = "df.csv"
            filepath = os.path.join(self.directory, filename)
            print("Making csvs and path is: ")
            print(filepath)
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', index=False, header=False)
            else:
                df.to_csv(filepath, mode='a', index=False)
           

    def df_to_parquet(self):
        for df in self.df_list:
            # str(self.df_names_list[counter-1])
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


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # episode = live, epoch = game
        # # screen output.
        # TODO: get proper screen outputs
        
        # print("obs: ", self.locals['episode_rewards'])
        # # self.save_observations(self.locals['obs'])
        # tried using built-in screen capture but no env object to work with 
        # print("here ")
        # step_str = str(CustomCallback.step)
        # print("Ss going into ", os.path.join(self.directory,'screenshot' + str(CustomCallback.step) + '.png'))
        subfolder = os.path.join(self.directory, 'screen')
        filepath =  os.path.join(subfolder, 'screenshot' + str(CustomCallback.step) + '.png')
        self.env.ale.saveScreenPNG(filepath)
        # episode_rewards, episode_lengths = evaluate_policy(self.model, self.env,
        #                                                        n_eval_episodes=self.n_eval_episodes,
        #                                                        render=self.render,
        #                                                        deterministic=self.deterministic,
        #                                                        return_episode_rewards=True)
        # pretty sure it is a list that gets appended per life?
        print("step: ", CustomCallback.step,  " rew: ", self.locals['episode_rewards'][-1])
        # episode rewards kind of weird, not sure if correct field is used
        step_stats = { CustomCallback.step: {
                'action': self.locals['env_action'],
                'action_name': self.actions[self.locals['env_action']],
                'episode_reward': self.locals['episode_rewards'][-1],
                'state': CustomCallback.step,
                'lives':self.locals['info']['ale.lives']
            }
        }

        # add step to dict and increment static step variable
        CustomCallback.main_data_dict.update(step_stats)
        CustomCallback.step = CustomCallback.step + 1
        
        # convert dict to different types
        if(CustomCallback.step == self.num_steps):
            self.make_dataframes()
            self.df_to_csv()
            self.df_to_parquet()
            # test if parquet file is correctly created
            print("reading parquet file")
            print(pd.read_parquet(os.path.join(self.directory,  "df.parquet")))
            print("finished!!")


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    