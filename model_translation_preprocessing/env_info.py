#!/usr/bin/env python3

import sys, gym, time, os

if __name__ == '__main__':

    envList = [
        'AdventureNoFrameskip-v4',
        'AirRaidNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'AmidarNoFrameskip-v4',
        'AssaultNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
        'AsteroidsNoFrameskip-v4',
        'AtlantisNoFrameskip-v4',
        'BankHeistNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'BerzerkNoFrameskip-v4',
        'BowlingNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'CarnivalNoFrameskip-v4',
        'CentipedeNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'CrazyClimberNoFrameskip-v4',
#        'DefenderNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'DoubleDunkNoFrameskip-v4',
        'ElevatorActionNoFrameskip-v4',
        'EnduroNoFrameskip-v4',
        'FishingDerbyNoFrameskip-v4',
        'FreewayNoFrameskip-v4',
        'FrostbiteNoFrameskip-v4',
        'GopherNoFrameskip-v4',
        'GravitarNoFrameskip-v4',
        'HeroNoFrameskip-v4',
        'IceHockeyNoFrameskip-v4',
        'JamesbondNoFrameskip-v4',
        'JourneyEscapeNoFrameskip-v4',
#        'KaboomNoFrameskip-v4',
        'KangarooNoFrameskip-v4',
        'KrullNoFrameskip-v4',
        'KungFuMasterNoFrameskip-v4',
        'MontezumaRevengeNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'NameThisGameNoFrameskip-v4',
        'PhoenixNoFrameskip-v4',
        'PitfallNoFrameskip-v4',
        'PongNoFrameskip-v4',
        'PooyanNoFrameskip-v4',
        'PrivateEyeNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'RiverraidNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',
        'RobotankNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'SkiingNoFrameskip-v4',
        'SolarisNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4',
        'StarGunnerNoFrameskip-v4',
        'TennisNoFrameskip-v4',
        'TimePilotNoFrameskip-v4',
        'TutankhamNoFrameskip-v4',
        'UpNDownNoFrameskip-v4',
        'VentureNoFrameskip-v4',
        'VideoPinballNoFrameskip-v4',
        'WizardOfWorNoFrameskip-v4',
        'YarsRevengeNoFrameskip-v4',
        'ZaxxonNoFrameskip-v4',
        ]

    directory = '..'
    directory = os.path.join(directory, 'envImages')
    
    
    for env in envList:
        env1 = gym.make(env)
        print("\n For the environment: ")
        print(env)
        env1.reset()
        save_name = env + ".png"
        save_file_path = os.path.join(directory, save_name)
        env1.env.ale.saveScreenPNG(save_file_path)
        print("Available actions are: ")
        action_names = env1.unwrapped.get_action_meanings()
        print(action_names)
        env1.close()
