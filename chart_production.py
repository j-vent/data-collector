import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
sns.set(style="whitegrid")
from matplotlib.colors import ListedColormap
import argparse

def print_df(stats_df):
    print("DF: ")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(stats_df)

def make_save_file_in_stream_dir(stream_folder, filename):
    if not (os.path.isdir(stream_folder)):
        os.makedirs(stream_folder)
        
    directory = ''
    directory = os.path.join(directory, stream_folder)

    save_file_plot = os.path.join(directory, filename)
    return save_file_plot

def rewards_per_life_scatterplot(pd_dataframe, stream_folder):
    f, ax = plt.subplots(figsize=(6.5,6.5))
    lives_ranking = [1,2,3,4]
    index = pd_dataframe.index
    sns.scatterplot(x=index,y="total reward", hue = "episode", hue_order = lives_ranking, sizes=(1,4),data=pd_dataframe,ax=ax)
    
    filename = "RewardsPerLife.png"
    save_file_plot = make_save_file_in_stream_dir(stream_folder, filename)
    print("Save path is... ")
    print(save_file_plot)
    
    plt.savefig(save_file_plot, dpi = 200)
    plt.clf()

def rewards_over_time(pd_dataframe, stream_folder):
    ax = sns.relplot(x="episode", y="total reward", kind="line",sort=False, data=pd_dataframe)
    ax.set(xlabel="Episode", ylabel = "Reward")
    filename = "RewardsOverTime.png"
    save_file_plot = make_save_file_in_stream_dir(stream_folder, filename)
    print("Save path is... ")
    print(save_file_plot)

    plt.savefig(save_file_plot, dpi = 200)
    plt.clf()
       
def pie_of_life(stream_folder, episode, life, action_labels, action_counts):
    ''' Episode is a set of three lives ending in a terminal state
        In each round of the game, agent has three lives '''
    
    # Make square figures and axes
    plt.figure(1, figsize=(20,10))

    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]

    plt.pie(action_counts,labels=action_labels, autopct='%.0f%%', shadow=True, colors=colors)
    
    titleString = "Actions in Episode " + str(episode) + ", Life: " + str(life)

    plt.title(titleString, fontsize=16)


    
    filename = "pie_" + str(episode) + "_" + str(life) + ".png"
    save_file_plot = make_save_file_in_stream_dir(stream_folder, filename)
    print("Save path is... ")
    print(save_file_plot)
    
    plt.savefig(save_file_plot, dpi = 200)
    plt.clf()
    
def big_scatter(pd_dataframe, stream_folder):
    filename = "Scatterplot.png"
    save_file_plot = make_save_file_in_stream_dir(stream_folder, filename)
    print("Save path is... ")
    print(save_file_plot)
    
    sns.scatterplot(x = 'episode step', y = 'episode reward', data = pd_dataframe, hue = 'episode')
    
    plt.savefig(save_file_plot, dpi = 200)
    plt.clf()
    
def big_line_with_dots(pd_dataframe, stream_folder):
    filename = "LineBubblePlot.png"
    save_file_plot = make_save_file_in_stream_dir(stream_folder, filename)
    print("Save path is... ")
    print(save_file_plot)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    x_val = df1['step']
    y_val = df1['total reward']
    
    plt.plot(x_val, y_val, c='blue')
    
#    sns.lineplot(x = 'step', y = 'total reward', data = pd_dataframe, ax = ax)
    
    # get all rows with reward > 10
    reward_pd = pd_dataframe.loc[pd_dataframe['reward'] > 0]
    
#    print_df(pd_dataframe)
    print_df(reward_pd)
    
    reward_palette = sns.color_palette("OrRd", 10)
    x_val = reward_pd[['step']].values
    print("reward step val is: ")
    print(x_val)
    y_val = reward_pd[['total reward']].values
    print("reward val is: ")
    print(y_val)
    plt.scatter(x_val, y_val, c='orange')
#    sns.scatterplot(x='step', y='reward', data = reward_pd, size = 4, palette = reward_palette, ax = ax)
    # get all rows with state is in list of summary states
    if (os.path.exists('./ramTest/summary_states.npy')):
        summary_states = np.load('./ramTest/summary_states.npy')
        x_val = summary_states
        key_states_pd = pd_dataframe.loc[pd_dataframe['step'].isin(summary_states)]
        
        print_df(key_states_pd)
        
        key_palette = sns.color_palette("BuGn", 10)
        y_val =key_states_pd[['total reward']].values
        print("key states val is: ")
        print(x_val)
        print("key states step val is: ")
        print(y_val)
        plt.scatter(x_val, y_val, c='green')
#        sns.scatterplot(x='step', y='reward', data = key_states_pd, size = 6, palette = key_palette , ax = ax)
    
    
    plt.savefig(save_file_plot, dpi = 200)
    plt.clf()

def matplotlib_stacked_bars_attempt(pd_dataframe, stream_folder):
#    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
#    men_means = [20, 35, 30, 35, 27]
#    women_means = [25, 32, 34, 20, 25]
#    men_std = [2, 3, 4, 1, 2]
#    women_std = [3, 5, 2, 3, 3]
#    width = 0.35       # the width of the bars: can also be len(x) sequence
#
#    fig, ax = plt.subplots()
#
#    ax.bar(labels, men_means, width, yerr=men_std, label='Men')
#    ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
#           label='Women')
#
#    ax.set_ylabel('Scores')
#    ax.set_title('Scores by group and gender')
#    ax.legend()
#
#    plt.savefig('stacked_bars.png', dpi=200)
    pd_dataframe.plot(kind='bar', stacked=True,
    colormap=ListedColormap(sns.color_palette("GnBu", 10)),
    figsize=(12,6))
    
    filename = "stacked_bars.png"
    save_file_plot = make_save_file_in_stream_dir(stream_folder, filename)
    print("Save path is... ")
    print(save_file_plot)
    
    plt.savefig(save_file_plot, dpi = 200)
    plt.clf()

def generate_visualizations(df1, df2, stream_folder):
    rewards_per_life_scatterplot(df1, stream_folder)
    
    rewards_over_time(df1, stream_folder)
    
    action_labels = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]
    tempDF = df1.loc[df1['end of episode'] == True]
    
    big_scatter(df1, stream_folder)
    
    big_line_with_dots(df1, stream_folder)
    
    for row, index in tempDF.head().iterrows():
        temp_row = row - 1
        print("ROW IS: " + str(temp_row))
        episode = df1.loc[df1.index[temp_row], 'epoch']
        print("EPISODE IS: " + str(episode))
        life = df1.loc[df1.index[temp_row], 'episode']
        print("LIFE IS: " + str(life))
        action_counts = []
        action_counts.append(df1.loc[df1.index[temp_row], "action 0 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 1 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 2 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 3 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 4 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 5 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 6 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 7 episode sum"])
        action_counts.append(df1.loc[df1.index[temp_row], "action 8 episode sum"])
        print("Action counts are: ")
        print(action_counts)
        pie_of_life(stream_folder, episode, life, action_labels, action_counts)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream-folder', type=str, default=dt_string)
    args = parser.parse_args()
    
    # load csv to pd
    file_path = os.path.join(args.stream_folder, "df1.csv")
    df1 = pd.read_csv(file_path)
    file_path = os.path.join(args.stream_folder, "df2.csv")
    df2 = pd.read_csv(file_path)
    # set up new folder
    stream_folder = os.path.join(args.stream_folder, "vis_test")
    if not (os.path.isdir(stream_folder)):
        os.makedirs(stream_folder)
        
    directory = ''
    directory = os.path.join(directory, stream_folder)
    # call functions
    generate_visualizations(df1, df2, stream_folder)
