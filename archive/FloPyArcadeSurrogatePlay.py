#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyAgent
from numpy import arange
from pathos.pools import _ProcessPool as Pool

# environment settings
envSettings = {
    'ENVTYPE': '3',
    # 'MODELNAMELOAD': 'gs-e3s8n500e100g100av10st200mpr2e-1mpo2e-3ar10x4v1relubn1_gen012_avg__417.5',
    'MODELNAMELOAD': None,
    'MODELNAME': None,
    'PATHMF2005': None,
    'PATHMP6': None,
    'SAVEPLOT': False,
    'SAVEPLOTALLAGENTS': False,
    'MANUALCONTROL': False,
    'RENDER': False,
    'SEEDENV': 1, # 500000000
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100
}

# game settings
gameSettings = {
    'NGAMES': 500000,
    'NGAMESPARALLEL': 4,
    'NAGENTSTEPS': 200
}

overwrite = False


def runGame(args):
    from FloPyArcade import FloPyArcade
    from os import mkdir
    from os.path import exists, join
    from pickle import dump

    envSettings, gameSettings, wrkspc, overwrite = args[0], args[1], args[2], args[3]

    game = FloPyArcade(
        modelNameLoad=envSettings['MODELNAMELOAD'],
        modelName='FloPyArcadeSurrogateSeed'+str(envSettings['SEEDENV']).zfill(9),
        NAGENTSTEPS=gameSettings['NAGENTSTEPS'],
        PATHMF2005=envSettings['PATHMF2005'],
        PATHMP6=envSettings['PATHMP6'],
        flagSavePlot=envSettings['SAVEPLOT'],
        flagManualControl=envSettings['MANUALCONTROL'],
        flagRender=envSettings['RENDER'],
        keepTimeSeries=True,
        nLay=envSettings['NLAY'],
        nRow=envSettings['NROW'],
        nCol=envSettings['NCOL']
        )

    fileResults = join(wrkspc, 'dev', 'gameDataNew', 'env' + envSettings['ENVTYPE'] + 's' + str(envSettings['SEEDENV']).zfill(9) + '.p')
    if not exists(fileResults) or overwrite:
        print('running game with seed', envSettings['SEEDENV'])
        game.play(
            ENVTYPE=envSettings['ENVTYPE'],
            seed=envSettings['SEEDENV']
            )

        data = game.timeSeries
        filehandler = open(fileResults, 'wb')
        dump(data, filehandler)
        filehandler.close()

def main(envSettings, gameSettings):
    agent = FloPyAgent()
    wrkspc = agent.wrkspc

    args = []
    for run in range(gameSettings['NGAMES']):
        envSettingsIdv = envSettings.copy()
        envSettingsIdv['SEEDENV'] = envSettings['SEEDENV'] + run
        args.append([envSettingsIdv, gameSettings, wrkspc, overwrite])
    chunksTotal = agent.yieldChunks(args,
        gameSettings['NGAMESPARALLEL']*agent.maxTasksPerWorker)
    for chunk in chunksTotal:
        print('games up to seed #', chunk[-1][0]['SEEDENV'])
        p = Pool(processes=gameSettings['NGAMESPARALLEL'])
        pool_mapAsync = p.map_async(runGame, chunk)
        pool_mapAsync.wait()
        p.close()
        p.join()
        p.terminate()


if __name__ == '__main__':
    main(envSettings, gameSettings)