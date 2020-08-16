#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from numpy import arange, array, divide, linspace, mean, median, square, sqrt
from pathos.pools import _ProcessPool as Pool
from pickle import dump, load

# environment settings
envSettings = {
    'ENVTYPE': '3',
    'MODELNAMELOAD': None,
    # 'MODELNAME': 'gs-e3s8n500e100g100av10st200mpr2e-1mpo2e-3ar10x4v1relubn1_gen012_avg__417.5',
    'MODELNAME': None,
    'PATHMF2005': None,
    'PATHMP6': None,
    'PATHSURROGATE': None,
    'PATHSURROGATE': ['bestModelUnweightedInitial', 'bestModelUnweighted'],
    'SAVEPLOT': False,
    'SAVEPLOTALLAGENTS': False,
    'MANUALCONTROL': False,
    'RENDER': False,
    'SEEDENV': 500000001,
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100
}

# game settings
gameSettings = {
    'NGAMES': 500,
    'NGAMESPARALLEL': 12,
    'NAGENTSTEPS': 200
}

reSimulate = False


def runGameSurrogate(args):
    from FloPyArcade import FloPyArcade
    from FloPyArcade import FloPyAgent
    from os.path import join
    import time

    # envSettings, gameSettings, agentModel = args[0], args[1], args[2]
    envSettings, gameSettings = args[0], args[1]

    agent = FloPyAgent()
    if envSettings['MODELNAME'] is not None:
        agentModel = agent.loadAgentModel(modelNameLoad=envSettings['MODELNAME'])
    else:
        agentModel = None


    if envSettings['PATHSURROGATE'] is not None:
        modelSteady = envSettings['PATHSURROGATE'][0]
        modelTransient = envSettings['PATHSURROGATE'][1]
        envSettings['PATHSURROGATE'] = [modelSteady, modelTransient]

    t0 = time.time()
    # surrogate game
    game = FloPyArcade( 
        agent=agentModel,
        modelNameLoad=None,
        modelName='FloPyArcadeSurrogateSeed'+str(envSettings['SEEDENV']).zfill(9),
        NAGENTSTEPS=gameSettings['NAGENTSTEPS'],
        PATHMF2005=envSettings['PATHMF2005'],
        PATHMP6=envSettings['PATHMP6'],
        surrogateSimulator=envSettings['PATHSURROGATE'],
        flagSavePlot=envSettings['SAVEPLOT'],
        flagSavePlotAllAgents=envSettings['SAVEPLOTALLAGENTS'],
        flagManualControl=envSettings['MANUALCONTROL'],
        flagRender=envSettings['RENDER'],
        keepTimeSeries=False
        )

    game.play(
        ENVTYPE=envSettings['ENVTYPE'],
        seed=envSettings['SEEDENV']
        )
    gameSurrogateTime = time.time() - t0
    print('surrogate time', gameSurrogateTime)

    gameSurrogateTimeSteps = game.timeSteps + 1
    gameSurrogateReward = game.rewardTotal

    return gameSurrogateTimeSteps, gameSurrogateReward, gameSurrogateTime

def runGameSimulated(args):
    from FloPyArcade import FloPyArcade
    from FloPyArcade import FloPyAgent
    from os.path import join
    from pickle import dump
    from tensorflow.keras.models import load_model as TFload_model
    import time

    # envSettings, gameSettings, agentModel = args[0], args[1], args[2]
    envSettings, gameSettings = args[0], args[1]

    agent = FloPyAgent()
    if envSettings['MODELNAME'] is not None:
        agentModel = agent.loadAgentModel(modelNameLoad=envSettings['MODELNAME'])
    else:
        agentModel = None

    t0 = time.time()
    # simulated game
    game = FloPyArcade(
        agent=agentModel,
        modelNameLoad=None,
        modelName='FloPyArcadeSurrogateSeed'+str(envSettings['SEEDENV']).zfill(9),
        NAGENTSTEPS=gameSettings['NAGENTSTEPS'],
        PATHMF2005=envSettings['PATHMF2005'],
        PATHMP6=envSettings['PATHMP6'],
        surrogateSimulator=None,
        flagSavePlot=envSettings['SAVEPLOT'],
        flagSavePlotAllAgents=envSettings['SAVEPLOTALLAGENTS'],
        flagManualControl=envSettings['MANUALCONTROL'],
        flagRender=envSettings['RENDER'],
        keepTimeSeries=False
        )

    game.play(
        ENVTYPE=envSettings['ENVTYPE'],
        seed=envSettings['SEEDENV']
        )
    gameSimulatedTime = time.time() - t0
    print('real time', gameSimulatedTime)

    gameSimulatedTimeSteps = game.timeSteps + 1
    gameSimulatedReward = game.rewardTotal

    return gameSimulatedTimeSteps, gameSimulatedReward, gameSimulatedTime

def main(envSettings, gameSettings, reSimulate):

    argsList = []
    for run in range(gameSettings['NGAMES']):
        args = []
        envSettingsIdv = envSettings.copy()
        envSettingsIdv['SEEDENV'] = envSettings['SEEDENV'] + run
        args = [envSettingsIdv, gameSettings]
        argsList.append(args)

    if reSimulate:
        gameSimulatedTimeSteps_, gameSimulatedRewards, gameSimulatedTimes = [], [], []
        pool = Pool(processes=gameSettings['NGAMESPARALLEL'])
        results = pool.map_async(runGameSimulated, argsList)
        results = results.get()
        pool.close()
        pool.join()
        pool.terminate()
        outputs = [result for result in results]
        print(outputs)
        for run in range(gameSettings['NGAMES']):
            gameSimulatedTimeSteps_.append(outputs[run][0])
            gameSimulatedRewards.append(outputs[run][1])
            gameSimulatedTimes.append(outputs[run][2])

        with open('C:\\FloPyArcade\\dev\\gameSimulatedTimeSteps.p', 'wb') as fp:
            dump(gameSimulatedTimeSteps_, fp)
        with open('C:\\FloPyArcade\\dev\\gameSimulatedRewards.p', 'wb') as fp:
            dump(gameSimulatedRewards, fp)
        with open('C:\\FloPyArcade\\dev\\gameSimulatedTimes.p', 'wb') as fp:
            dump(gameSimulatedTimes, fp)

    gameSurrogateTimeSteps_, gameSurrogateRewards, gameSurrogateTimes = [], [], []
    pool = Pool(processes=gameSettings['NGAMESPARALLEL'])
    results = pool.map_async(runGameSurrogate, argsList)
    results = results.get()
    pool.close()
    pool.join()
    pool.terminate()
    outputs = [result for result in results]
    for run in range(gameSettings['NGAMES']):
        gameSurrogateTimeSteps_.append(outputs[run][0])
        gameSurrogateRewards.append(outputs[run][1])
        gameSurrogateTimes.append(outputs[run][2])

    with open('C:\\FloPyArcade\\dev\\gameSurrogateTimeSteps.p', 'wb') as fp:
        dump(gameSurrogateTimeSteps_, fp)
    with open('C:\\FloPyArcade\\dev\\gameSurrogateRewards.p', 'wb') as fp:
        dump(gameSurrogateRewards, fp)
    with open('C:\\FloPyArcade\\dev\\gameSurrogateTimes.p', 'wb') as fp:
        dump(gameSurrogateTimes, fp)


def plot(envSettings, gameSettings):

    import matplotlib.pyplot as plt

    with open('C:\\FloPyArcade\\dev\\gameSurrogateTimeSteps.p', 'rb') as fp:
        gameSurrogateTimeSteps_ = load(fp)[0:gameSettings['NGAMES']]
    with open('C:\\FloPyArcade\\dev\\gameSurrogateRewards.p', 'rb') as fp:
        gameSurrogateRewards = load(fp)[0:gameSettings['NGAMES']]
    with open('C:\\FloPyArcade\\dev\\gameSurrogateTimes.p', 'rb') as fp:
        gameSurrogateTimes = load(fp)[0:gameSettings['NGAMES']]
    with open('C:\\FloPyArcade\\dev\\gameSimulatedTimeSteps.p', 'rb') as fp:
        gameSimulatedTimeSteps_ = load(fp)[0:gameSettings['NGAMES']]
    with open('C:\\FloPyArcade\\dev\\gameSimulatedRewards.p', 'rb') as fp:
        gameSimulatedRewards = load(fp)[0:gameSettings['NGAMES']]
    with open('C:\\FloPyArcade\\dev\\gameSimulatedTimes.p', 'rb') as fp:
        gameSimulatedTimes = load(fp)[0:gameSettings['NGAMES']]

    # calculating timestep speed increase factors
    print(len(gameSimulatedTimes))
    print(len(gameSimulatedTimeSteps_))
    print(len(gameSurrogateTimes))
    print(len(gameSurrogateTimeSteps_))
    print(gameSimulatedTimes)
    print(gameSimulatedTimeSteps_)
    print(gameSurrogateTimes)
    print(gameSurrogateTimeSteps_)
    gameSurrogateSpeedIncreases = divide(divide(gameSimulatedTimes, gameSimulatedTimeSteps_), divide(gameSurrogateTimes, gameSurrogateTimeSteps_))

    # comparing outcome of simulation and surrogate simulation
    tp, fp, tn, fn, nWins, nLosses = 0, 0, 0, 0, 0, 0
    for i in range(len(gameSimulatedRewards)):
        positiveSimulated, positiveSurrogate = False, False
        if gameSimulatedRewards[i] != 0.:
            positiveSimulated = True
            nWins += 1
        elif gameSimulatedRewards[i] == 0.:
            nLosses += 1
        if gameSurrogateRewards[i] != 0.:
            positiveSurrogate = True
        if positiveSimulated:
            if positiveSimulated == positiveSurrogate:
                tp += 1
            elif positiveSimulated != positiveSurrogate:
                fp += 1
        if not positiveSimulated:
            if positiveSimulated == positiveSurrogate:
                tn += 1
            elif positiveSimulated != positiveSurrogate:
                fn += 1


    # is this correct?!


    mseTimeSteps = mean(sqrt(square(array(gameSurrogateTimeSteps_) - array(gameSimulatedTimeSteps_))))
    mseRewards = mean(sqrt(square(array(gameSurrogateRewards) - array(gameSimulatedRewards))))
    # precision: what proportion of positive identifications was actually correct?
    precision = tp / (tp + fp)
    # recall: what proportion of actual positives was identified correctly?
    if (tp + fn) == 0.:
        recall = 0.
    else:
        recall = tp / (tp + fn)
    trueWinsPercentage = tp/nWins
    trueLossesPercentage = tn/nLosses

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(linspace(-2010., 2010., num=10), linspace(-2010., 2010., num=10), lw=0.2)
    plt.scatter(gameSimulatedTimeSteps_, gameSurrogateTimeSteps_, s=3)
    plt.text(20, 180, 'rmse: ' + '{:.0f}'.format(mseTimeSteps))
    plt.xlabel('# time steps (simulation)')
    plt.ylabel('# time steps (surrogate simulation)')
    plt.xlim(-1., 200.)
    plt.ylim(-1., 200.)
    plt.subplot(132)
    plt.plot(linspace(-2010., 2010., num=10), linspace(-2010., 2010., num=10), lw=0.2)
    plt.scatter(gameSimulatedRewards, gameSurrogateRewards, s=3)
    plt.text(100, 900,
        'true wins: ' + '{:.1f}'.format(100. * trueWinsPercentage) + ' %\n' + 
        'true losses: ' + '{:.1f}'.format(100. * trueLossesPercentage) + ' %\n' +
        # 'precision: ' + '{:.2f}'.format(precision) + '\n' + 
        # 'recall: ' + '{:.2f}'.format(recall) + '\n' +
        'rmse: ' + '{:.0f}'.format(mseRewards))
    plt.xlabel('reward (simulation)')
    plt.ylabel('reward (surrogate simulation)')
    plt.xlim(-30., 1100.)
    plt.ylim(-30., 1100.)
    plt.subplot(133)
    plt.hist(gameSurrogateSpeedIncreases, zorder=1)
    plt.axvline(x=median(gameSurrogateSpeedIncreases), zorder=2, c='black')
    plt.xlabel('factor of computation\nacceleration from surrogate')
    plt.ylabel('occurences')
    plt.tight_layout()
    plt.savefig('C:\\FloPyArcade\\dev\\surrogateComparisonAndSpeedup.png', dpi=500)


if __name__ == '__main__':
    main(envSettings, gameSettings, reSimulate)
    plot(envSettings, gameSettings)