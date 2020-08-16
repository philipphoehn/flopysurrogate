#!/usr/bin/python3
# -*- coding: utf-8 -*-

from glob import glob
from os import environ, makedirs
from os.path import dirname, exists, join, realpath
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, asarray, divide, load, save, sqrt
from numpy import load as numpyLoad
from pickle import dump, load
from sklearn import preprocessing
from FloPyArcade import FloPyEnv
from FloPyArcade import FloPyAgent

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from logging import getLogger, FATAL
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
getLogger('tensorflow').setLevel(FATAL)
FloPyAgent().GPUAllowMemoryGrowth()


surrogateSettings = {
    'ENVTYPE': '3',
    'PATHMF2005': None,
    'PATHMP6': None,
    'SEEDENV': 1,
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100,
    'SURROGATENAME': 'surrogateEnv3_classTest',
    'COMPILEDATASET': True,
    'TRAIN': True,
    'CONTINUETRAIN': False,
    'TEST': True,
    'NMODELTRAINS': 1,
    'MAXSAMPLES': 2000,
    'BATCHSIZE': 128,
    'LEARNINGRATE': 0.00061,
    'EPOCHS': 10,
    'PATIENCE': 10
}

# trainSamplesLimit = None
# testSamplesLimit = None
# learningRateInterval = [-3.8, -4.2]
# learningRateInterval = [-5.5, -5.5]


class FloPyEnvSurrogate():

    def __init__(self, surrogateSettings):
        ''' Constructor. '''
        self.wrkspc = dirname(realpath(__file__))
        self.wrkspcDev = join(self.wrkspc, 'dev')
        if not exists(self.wrkspcDev):
            makedirs(self.wrkspcDev)
        self.env = FloPyEnv(ENVTYPE=surrogateSettings['ENVTYPE'],
                            initWithSolution=False)
        
        self.surrogateSettings = surrogateSettings
        self.ENVTYPE = surrogateSettings['ENVTYPE']
        self.PATHMF2005 = surrogateSettings['PATHMF2005']
        self.PATHMP6 = surrogateSettings['PATHMP6']
        self.SEEDENV = surrogateSettings['SEEDENV']
        self.NLAY = surrogateSettings['NLAY']
        self.NROW = surrogateSettings['NROW']
        self.NCOL = surrogateSettings['NCOL']
        self.SURROGATENAME = surrogateSettings['SURROGATENAME']
        self.COMPILEDATASET = surrogateSettings['COMPILEDATASET']
        self.TRAIN = surrogateSettings['TRAIN']
        self.CONTINUETRAIN = surrogateSettings['CONTINUETRAIN']
        self.TEST = surrogateSettings['TEST']
        self.MAXSAMPLES = surrogateSettings['MAXSAMPLES']
        self.BATCHSIZE = surrogateSettings['BATCHSIZE']
        self.LEARNINGRATE = surrogateSettings['LEARNINGRATE']
        self.EPOCHS = surrogateSettings['EPOCHS']
        self.PATIENCE = surrogateSettings['PATIENCE']
        self.NRANDOMMODELS = surrogateSettings['NMODELTRAINS']
    
    def compileDataset(self):
        '''
        Assemble datasets. Choose between training, validation and testing.
        '''
        
        filesGames = glob(join(self.wrkspc, 'dev', 'gameDataNewInitialsOnly', 'env3*.p'))
        X, Y, i = [], [], 1
        print('Compiling dataset ...')
        for data in (filesGames if self.MAXSAMPLES is None else filesGames[:self.MAXSAMPLES]):
            filehandler = open(data, 'rb')
            data = load(filehandler)
            filehandler.close()
            stress, state = self.extractDatasetFullHeadField(data)
            X.append(stress)
            Y.append(state)
            i += 1

        X, Y = asarray(X), asarray(Y)
        self.nInput, self.nPredictions = len(X[0]), len(Y[0])
        save(join(self.wrkspcDev, 'surrogateInitialXtrain_'), X)
        save(join(self.wrkspcDev, 'surrogateInitialYtrain_'), Y)
        save(join(self.wrkspcDev, 'surrogateInitialXval_'), X)
        save(join(self.wrkspcDev, 'surrogateInitialYval_'), Y)
        save(join(self.wrkspcDev, 'surrogateInitialXtest_'), X)
        save(join(self.wrkspcDev, 'surrogateInitialYtest_'), Y)

    def loadDataset(self, type='train'):
        '''
        Load dataset.
        '''

        X = numpyLoad(join(self.wrkspc, 'dev', 'surrogateInitialX' + type + '_.npy'))
        Y = numpyLoad(join(self.wrkspc, 'dev', 'surrogateInitialY' + type + '_.npy'))
        self.nInput, self.nPredictions = len(X[0]), len(Y[0])
        return X, Y

    def extractDatasetFullHeadField(self, data):
        ''' Extract full head field. '''
        stresses = data['stressesNormalized']
        stress = asarray(stresses[0])
        states = data['headsFullField']
        state = asarray(states[0])
        state = state.flatten()
        state = divide(state, self.env.maxH)
        
        return stress, state

    def extractDatasetWellHeadField(self):
        ''' Extract specific dataset describing head field around well. '''
        # wellX, wellY = stress[-4], stress[-3]
        pass

    def unnormalizeInitial(self, data, env):
        ''' Remove data normalization. '''
        
        from numpy import multiply
        data = multiply(data, env.maxH)

        return data

    def createSurrogateModel(self):
        ''' Create a neural network model as surrogate simulator. '''
        self.model = Sequential()
        self.model.add(Dense(input_dim=self.nInput, units=100))
        self.model.add(Activation('relu'))
        self.model.add(Dense(input_dim=100, units=250))
        self.model.add(Activation('relu'))
        self.model.add(Dense(input_dim=250, units=1000))
        self.model.add(Activation('relu'))
        self.model.add(Dense(input_dim=1000, units=2500))
        self.model.add(Activation('relu'))
        self.model.add(Dense(input_dim=2500, units=self.nPredictions))
        self.model.add(Activation('linear'))

    def generator(self, X_data, y_data, batch_size):
        ''' Yield data in batches. '''
        samples_per_epoch = X_data.shape[0]
        number_of_batches = samples_per_epoch/batch_size
        counter = 0

        while 1:
            X_batch = array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
            y_batch = array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
            counter += 1
            yield X_batch, y_batch

            # restarting counter to yield data in the next epoch as well
            if counter >= number_of_batches:
                counter = 0

    def trainSurrogate(self, useGenerator=True):
        '''
        Train a surrogate simulator for defined environment.
        Simulation results have to be available.
        
        CURRENTLY FEEDING IN-SAMPLE DATA, TO FIRST ACHIEVE OVERFIT.
        '''        

        if not exists(join(self.wrkspc, 'dev', 'modelCheckpoints')):
            makedirs(join(self.wrkspc, 'dev', 'modelCheckpoints'))

        X, Y = self.loadDataset('train')
        batchSize = self.surrogateSettings['BATCHSIZE']
        
        if self.CONTINUETRAIN == True:
            self.model = load_model(join(self.wrkspcDev,
                                         self.SURROGATENAME + '.h5'))
        else:
            self.createSurrogateModel()
        
        # for i in range(nRandomModels):
        #     learningRate = 10**(np.random.uniform(learningRateInterval[0],
        #                                           learningRateInterval[1]))

        optimizer = Adam(learning_rate=surrogateSettings['LEARNINGRATE'],
                         beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                         amsgrad=False)

        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

        earlyStopping = EarlyStopping(monitor='val_loss',
            patience=self.surrogateSettings['PATIENCE'], 
            verbose=0, mode='min')

        checkpoint = ModelCheckpoint(join(self.wrkspc, 'dev',
                                          'modelCheckpoints',
                                          self.SURROGATENAME) + '.h5',
                                     verbose=0, monitor='val_loss',
                                     save_best_only=True, mode='auto',
                                     restore_best_weights=True)

        if not useGenerator:
            results = self.model.fit(X, Y,
                use_multiprocessing=False,
                batch_size=batchSize, epochs=epochs, validation_split=0.5,
                shuffle=True,
                callbacks=[earlyStopping, checkpoint],
                verbose=1)

        if useGenerator:
            # potentially leaks memory but saves memory initially
            # https://stackoverflow.com/questions/58150519/resourceexhaustederror-oom-when-allocating-tensor-in-keras
            results = self.model.fit_generator(self.generator(X, Y, batchSize),
                use_multiprocessing=False,
                epochs=self.surrogateSettings['EPOCHS'],
                steps_per_epoch = len(Y)/batchSize,
                validation_data = self.generator(X,Y,batchSize*2),
                validation_steps = len(Y)/batchSize*2,
                callbacks=[earlyStopping, checkpoint],
                verbose=1)

        vallossMin = min(results.history['val_loss'])
        with open(join(self.wrkspc, 'dev', 'modelCheckpoints',
                       self.SURROGATENAME + '_valLoss' + f'{vallossMin:03f}' + '.p'),
                  'wb') as f:
            dump(results.history, f)

        self.model.save(join(self.wrkspc, 'dev', self.SURROGATENAME + '.h5'))

    def testSurrogate(self):
        ''' Test surrogate simulator performance. '''
        
        Xtest, Ytest = self.loadDataset('test')
        self.model = load_model(join(self.wrkspcDev,
                             self.SURROGATENAME + '.h5'))
        
        predsAll, simsAll = [], []
        for it in range(nPredictions):
            predsAll.append([])
            simsAll.append([])

        # iterating through test samples
        for i in range(len(Xtest)):
            prediction = model.predict(Xtest[i, :].reshape(1, -1), batch_size=1)
            predictionFlatten = array(prediction).flatten()

            prediction = unnormalizeInitial(prediction, env)

            simulated = Ytest[i, :]
            simulated = unnormalizeInitial(simulated, env)

            for k in range(nPredictions):
                self.predsAll[k].append(prediction[k])
                self.simsAll[k].append(simulated[k])

            if (i % 1000 == 0):
                # print('predict', i, len(Xtest), 'prediction', prediction, 'simulated', simulated)
                print('predict', i, len(Xtest))

    def plotPredictions(self):
        ''' Plot predictions.
        '''

        for i in range(nPredictions):
            plt.figure(1)
            plt.subplot(211)
            sim = self.simsAll[i]
            pred = self.predsAll[i]
            plt.scatter(sim, pred, s=0.4, lw=0., marker='.')
            plt.xlim(left=0, right=10.5)
            plt.ylim(bottom=0, top=10.5)
            plt.subplot(212)
            plt.hist(sim, bins=30)
            plt.xlim(left=0, right=10.5)
            plt.savefig(join(wrkspc, 'dev', self.SURROGATENAME + 'pred' + str(i+1).zfill(2) + '.png'), dpi=1000)
            plt.close('all')

    def runGame(args):
        ''' Simulate as single game. '''
        
        from FloPyArcade import FloPyArcade
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
            flagSavePlotAllAgents=envSettings['SAVEPLOTALLAGENTS'],
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
    
    def play(envSettings, gameSettings):
        ''' Call simulations to generate simulation dataset. '''
        agent = FloPyAgent()
        wrkspc = agent.wrkspc

        # for run in range(gameSettings['NGAMES']):
        #     envSettingsIdv = envSettings.copy()
        #     envSettingsIdv['SEEDENV'] = envSettings['SEEDENV'] + run
        #     runGame([envSettingsIdv, gameSettings, wrkspc, overwrite])
    
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


def main(surrogateSettings):
    surrogate = FloPyEnvSurrogate(surrogateSettings)
    if surrogateSettings['COMPILEDATASET']:
        surrogate.compileDataset()
    if surrogateSettings['TRAIN']:
        surrogate.trainSurrogate()


if __name__ == '__main__':
    main(surrogateSettings)