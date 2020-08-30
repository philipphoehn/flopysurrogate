# from __future__ import print_function
from glob import glob
from tensorflow.keras import Input, Model
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import load_model, model_from_json, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from numpy import argmin, array, mean, shape, sort, square, sqrt
from numpy.random import uniform, randint
from os import remove
from os.path import join, sep
from pickle import dump, load
from psutil import cpu_count
from random import choice
from sklearn import preprocessing
import time

from FloPyArcade import FloPyEnv

wrkspc = 'C:\\FloPyArcade'
recompileDataset = False
recompileRepeatedly = False
reTrain = True
reTest = False
reTestOnlyBest = True
reLoadBestAndTest = False
trainGameLimit = None
testGameLimit = None
trainSamplesLimit = None
testSamplesLimit = None
predictBatchSize = 16384
flagOnInitialState = False
flagWeight = False
useTensorboard = False

# currently near
# bs512ly3n392x261x733lr9.41e-05eps2.46e-11reluglorot_uniformdc1.00e+00do0.00e+00pat239UnweightedHistory_val_loss8.74e-05

maxModelsKept = 1000
nRandomModels = 100000
nLayers_ = [4, 3, 2]
nNeuronsInterval = [2, 3.2]               # log scale
learningRateInterval = [-3.8, -4.5]         # log scale
epsilonsInterval = [-10, -11]            # log scale
# batchSizes = [512]
batchSizes = [1024, 512, 256]
# batchSizes = [65536, 32768, 16384, 8192, 4096]
# activations = ['relu']
# activations = ['tanh']
# activations = ['sigmoid']
# activations = ['sigmoid', 'tanh', 'relu']
# activations = ['relu']
activations = ['relu', 'sigmoid', 'softsign']
# activations = ['relu']
# activations = ['sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'hard_sigmoid', 'exponential', 'linear']
# kernelInitializers = ['glorot_uniform', 'glorot_normal']
# kernelInitializers = ['glorot_uniform']
kernelInitializers = ['glorot_uniform', 'he_uniform', 'random_uniform', 'lecun_uniform']
# kernelInitializers = ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal', 'random_uniform', 'random_normal', 'lecun_uniform', 'lecun_normal']
# dropoutsInterval = [0.0, 0.5]
dropoutsInterval = [0.0, 0.0]
patiencesInterval = [2, 3]          # log scale
# patiencesInterval = [1.2, 2]          # log scale
epochs = 100000000
decaysInterval = [0, 0]

backend.set_floatx('float32')


def collectInputAndStateFromFile(data, flagOnInitialState):

    filehandler = open(data, 'rb')
    data = load(filehandler)
    filehandler.close()

    states = data['statesNormalized']
    stresses = data['stressesNormalized']
    rewards = data['rewards']
    doneFlags = data['doneFlags']
    successFlags = data['successFlags']

    inputData = []
    trueData = []
    if flagOnInitialState:
        # stresses should include particle location as heads prediction depends on it (this is states[0:3])
        # NOTE: will keep for the moment, but states[0][3:5] are not changing in ENV3 and could be dropped from the entire dataset

        inputDataTemp = states[0][0:3] + stresses[0]
        # inputDataTemp = stresses[0]
        # print('states[0][0:3]', states[0][0:3])
        # print('stresses[0]', stresses[0])
        # predictions should only be heads, therefore particle locations (first three)
        # and well information (last four) ignored
        trueDataTemp = states[0][3:-4]



        # this is temporary to test predicting well-surrounding heads only
        # trueDataTemp = states[0][3+20:-4]
        # trueDataTemp = states[0][3+11:-4]
        # print(trueDataTemp)
        # print('inputDataTemp', inputDataTemp)
        # time.sleep(100)
        inputData.append(inputDataTemp)
        trueData.append(trueDataTemp)

    if not flagOnInitialState:
        for i in range(1, len(states)-1):
        # for i in range(1, len(states)-1):
            # current states are particle locations + heads
            # ignoring last 4 as these are repeated in stresses (these are wellQ, wellX, wellY, wellZ)
            # stresses here are 2 boundary conditions and wellQ, wellX, wellY, wellZ
            # adds new stresses and previous stresses
            # what is this doing? NEW: adding previous states -- DOES THIS WORK? AND MAYBE IMPROVE?
            # ignoring first two stresses as these are just the constant head forcings
            # inputDataTemp = states[i-1][:-4] + stresses[i-1][-3:] + states[i][:-4] + stresses[i][-3:]

            # THIS ASSUMES HAVING THE HEAD FIELD BEFORE PREDICTING THE PARTICLE
            inputDataTemp = states[i-1][:-4] + stresses[i-1] + states[i][3:-4] + stresses[i]

            # THIS DOES NOT ASSUME HAVING THE HEAD FIELD BEFORE PREDICTING THE PARTICLE
            # inputDataTemp = states[i-1][:-4] + stresses[i-1] + stresses[i]

            trueDataTemp = states[i][0:3]
            



            # this is temporary to test predicting particle location
            # inputDataTemp = states[i-1][:-4] + stresses[i-1][-3:] + states[i][:-4] + stresses[i][-3:]


            # temporarily only considering the current state and stress
            # inputDataTemp = states[i][:-4] + stresses[i][-3:]
            # # trueDataTemp = states[i+1][0:3]
            # # trueDataTemp = states[i+1][13:23]
            # trueDataTemp = states[i+1][:-4]


            inputData.append(inputDataTemp)
            trueData.append(trueDataTemp)

    data = [inputData, trueData]

    return data

def test_(model, wrkspc, suffix, Xtest, Ytest, nPredictions, env, plotRegression=False, predictInBatches=False):

    simsAll = [[] for i in range(nPredictions)]
    predsAll = [[] for i in range(nPredictions)]
    simsNorm = [[] for i in range(nPredictions)]
    predsNorm = [[] for i in range(nPredictions)]


    if not predictInBatches:
        t0 = time.time()
        # iterating through test samples
        for i in range(1, len(Xtest)-1):
            # actually no need to predict wellQ and wellCoords
            prediction = model.predict(Xtest[i, :].reshape(1, -1), batch_size=1)
            predictionFlatten = np.array(prediction[0])
            if flagOnInitialState:
                # for initial state prediction, only heads are predicted
                prediction = {'heads': predictionFlatten}
                simulated = {'heads': Ytest[i, :]}
            elif not flagOnInitialState:
                prediction = observationsVectorToDict(predictionFlatten)
                simulated = observationsVectorToDict(Ytest[i, :])
            prediction = env.unnormalize(prediction)
            simulated = env.unnormalize(simulated)
            if flagOnInitialState:
                prediction = np.array(prediction['heads'])
                simulated = np.array(simulated['heads'])
            elif not flagOnInitialState:
                prediction = np.array(observationsDictToVector(prediction))
                simulated = np.array(observationsDictToVector(simulated)).flatten()

            for k in range(nPredictions):
                predsAll[k].append(prediction[k])
                simsAll[k].append(simulated[k])
            if (i % 1000 == 0):
                # print('predict', i, len(Xtest), 'prediction', prediction, 'simulated', simulated)
                print('predict', i, len(Xtest))

    if predictInBatches:
        t0 = time.time()
        # iterating through test samples
        # actually no need to predict wellQ and wellCoords
        predictionBatch = model.predict(Xtest, use_multiprocessing=True, batch_size=predictBatchSize)


        # prediction = model.predict(Xtest[i, :].reshape(1, -1), batch_size=1)
        # print('debug shape', np.shape(predictionBatch))
        # print('debug shape', np.shape(Xtest))
        # print('debug shape', np.shape(Ytest))
        # print('debug shape', Ytest[0, :])
        # print('debug shape', type(Ytest[0, :]))
        # print('debug shape', np.shape(Ytest[0, :]))

        # rerun data collection, as incorrect particle heads were read

        for i in range(np.shape(Ytest)[0]):
            # print('debug predictionBatch', predictionBatch[i].reshape(1, -1)[0])
            predictionFlatten = np.array(predictionBatch[i, :])
            # predictionFlatten = np.array(predictionBatch[i][0])
            # print('debug', predictionFlatten)
            # print('debug', 1)
            for k in range(nPredictions):
                predsNorm[k].append(predictionFlatten[k])
                simsNorm[k].append(Ytest[i, :][k])
                # predsNorm[k].append(Ytest[i, :][k])
            # print('debug', 2)
            if flagOnInitialState:
                # for initial state prediction, only heads are predicted
                prediction = {'heads': predictionFlatten}
                simulated = {'heads': list(Ytest[i, :])}
            elif not flagOnInitialState:
                prediction = observationsVectorToDict(predictionFlatten)
                simulated = observationsVectorToDict(Ytest[i, :])
            # print('debug', 3)
            # ?????
            prediction = env.unnormalize(prediction)
            # print('debug', simulated)
            simulated = env.unnormalize(simulated)
            # print('debug', simulated)
            # time.sleep(10000)
            # print('debug', 4)
            if flagOnInitialState:
                prediction = np.array(prediction['heads'])
                simulated = np.array(simulated['heads'])
            elif not flagOnInitialState:
                prediction = np.array(observationsDictToVector(prediction))
                simulated = np.array(observationsDictToVector(simulated))
            # print('debug', 5)

            # print('prediction', prediction)
            # print('simulated', simulated)
            # time.sleep(10000)

            for k in range(nPredictions):
                predsAll[k].append(prediction[k])
                simsAll[k].append(simulated[k])


    # calculating mean squared error
    msesAll = []
    for i in range(nPredictions):
        sim = simsNorm[i]
        pred = predsNorm[i]
        msePred = (square(array(sim) - array(pred))).mean(axis=None)
        msesAll.append(msePred)
    mse = mean(msesAll)
    print('time batch test preditions', time.time() - t0)

    if plotRegression:
        for i in range(nPredictions):
            plt.figure(1)
            plt.subplot(211)
            sim = simsAll[i]
            pred = predsAll[i]
            error_mse = (square(array(sim) - array(pred))).mean(axis=None)
            error_rmse = sqrt((square(array(sim) - array(pred))).mean(axis=None)) # '%.5f' % .1
            plt.scatter(sim, pred, s=0.4, lw=0., marker='.', alpha=0.5, zorder=2)
            plt.plot([-20.0, 20.0], [-20.0, 20.0], lw=0.05, color='black', zorder=1,
                label='mse: ' + ('%.3f' % error_mse) + '\nrmse: ' + ('%.3f' % error_rmse))
            plt.legend(frameon=False)
            if flagOnInitialState or i > 2:
                plt.xlim(left=0, right=10.5)
                plt.ylim(bottom=0, top=10.5)
            plt.subplot(212)
            plt.hist(sim, bins=30)
            if flagOnInitialState or i > 2:
                plt.xlim(left=0, right=10.5)
            else:
                dx = sqrt((max(sim)-min(sim))**2)
                plt.xlim(left=min(sim)-(dx*0.1), right=max(sim)+dx*0.1)
            plt.savefig(join(wrkspc, 'dev', 'bestModel' + suffix + '_pred' + str(i+1).zfill(2) + '.png'), dpi=2000)
            print('debug saving', join(wrkspc, 'dev', 'bestModel' + suffix + '_pred' + str(i+1).zfill(2) + '.png'))
            plt.close('all')
    # print('debug minmax', min(simsAll[0]), max(simsAll[0]), min(predsAll[0]), max(predsAll[0]))

    return mse

def createSequentialModel(lenInput, nPredictions, nNeuronsLayers, activation, kernelInitializer, nLayers, dropout):
    model = Sequential()
    layer = Dense(nNeuronsLayers[0], name='hl1', activation=activation, kernel_initializer=kernelInitializer, input_dim=lenInput)
    model.add(layer)
    for i in range(nLayers-1):
        # print('layer', i, 'hl' + str(i+2))
        layer = Dense(nNeuronsLayers[i+1], name='hl' + str(i+2), activation=activation, kernel_initializer=kernelInitializer)
        model.add(layer)
        do = Dropout(dropout)
        model.add(do)
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    # model.add(Dense(nPredictions, activation=activation))
    # does this need to be linear to work properly?
    model.add(Dense(nPredictions, activation='linear'))

    return model

def unnormalize(data, env):
    from numpy import multiply

    data['particleCoords'] = multiply(data['particleCoords'],
        env.minX + env.extentX)
    data['heads'] = multiply(data['heads'],
        env.maxH)
    data['wellQ'] = multiply(data['wellQ'], env.minQ)
    data['wellCoords'] = multiply(data['wellCoords'], env.minX + env.extentX)
    return data

def calculate_weightsforregression(y, nbins=30):
    '''
    Calculate weights to balance a binary dataset
    '''
    from numpy import digitize, divide, histogram, ones, subtract, sum, take

    # hist, bin_edges = histogram(y, bins=nbins, density=True)
    # returning relative frequency
    hist, bin_edges, patches = plt.hist(y, bins=nbins, density=True)
    # returning true counts in bin
    # hist, bin_edges, patches = plt.hist(y, bins=nbins, density=False)
    # to receive indices 1 has to be subtracted
    freqIdx = subtract(digitize(y, bin_edges, right=True), 1)
    freqs = take(hist, freqIdx)

    if 0. in freqs:
        sample_weights = ones(len(freqs))
    else:
        # choosing 100 in numerator to avoid too small weights with biased datasets
        # potentially make this dependent on the magnitude difference between min and max?
        # print('min freq', min(freqs))
        sample_weights = divide(1., freqs)

    # print('min, max', min(sample_weights), max(sample_weights))

    return np.asarray(sample_weights)

def listExistingModelsAndLosses(suffix, sortedbyloss=True):
    filesJSON = glob(join(wrkspc, 'dev', 'modelCheckpoints', '*' + suffix + '.json'))
    filesLoss = []
    i = 0
    for fileJSON in filesJSON:
        # print('debug fileJSON', fileJSON)
        filesHistory = glob(fileJSON.replace('.json', '') + 'History*.p')
        if len(filesHistory) > 0:
            fileHistory = filesHistory[0]
            # removing redudant history files
            i = 0
            for file in filesHistory:
                if i > 0:
                    remove(file)
                i += 1

            history = load(open(fileHistory, 'rb'))
            valLoss = min(history['val_loss'])
            filesLoss.append(valLoss)

        elif len(filesHistory) == 0:
            # removing weights and json files with missing history, as no information on them is available
            del filesJSON[i]
            remove(fileJSON)
            remove(fileJSON.replace('.json', 'Weights.h5'))
        i += 1


    if sortedbyloss:
        sortedObjects = list(zip(*sorted(zip(filesLoss, filesJSON))))
        filesLoss, filesJSON = sortedObjects[0], sortedObjects[1]
        # filesLoss = list(filesLoss)
        # filesJSON = list(filesJSON)

    return filesJSON, filesLoss

def continueAsBest(vallossCurrent, modelName, suffix):
    continueFlag = False
    bestModelLoss = 100000.
    files = glob(join(wrkspc, 'dev', 'modelCheckpoints', '*' + suffix + 'History*.p'))
    if len(files) == 0:
        continueFlag = True
    else:
        losses, idx = [], 0
        for file in files:
            loss = float(file.split('loss')[-1].split('.p')[0])
            losses.append(loss)
            # print('debug', modelName + suffix, file)
            if modelName + suffix in file:
                currentModelIdx = idx
                lossCurrentFound = loss
            idx += 1

        st = np.sort(losses)
        rankCurrent = int(float([i for i, k in enumerate(st) if k == lossCurrentFound][0]) + 1)
        bestModelIdx = argmin(losses)
        bestModelName = files[bestModelIdx].split(sep)[-1].split('History')[0]
        bestModelLoss = losses[bestModelIdx]
        # if the current model is better
        # if float('{:.2e}'.format(vallossCurrent)) <= bestModelLoss:
        #     continueFlag = True
        if rankCurrent == 1:
            continueFlag = True
    
    # print('debug bestModellLoss', bestModelLoss, 'vallossCurrent', vallossCurrent, 'continueFlag', continueFlag)
    return continueFlag, rankCurrent

def observationsVectorToDict(observationsVector):
    """Convert list of observations to dictionary."""
    observationsDict = {}
    observationsDict['particleCoords'] = observationsVector[:3]
    # print('observationsDict[particleCoords]', observationsDict['particleCoords'])
    observationsDict['heads'] = observationsVector[3:]
    # print('observationsDict[heads]', observationsDict['heads'])
    return observationsDict

def observationsDictToVector(observationsDict):
    """Convert dictionary of observations to list."""
    observationsVector = []
    for obs in observationsDict['particleCoords']:
        observationsVector.append(obs)
    for obs in observationsDict['heads']:
        observationsVector.append(obs)
    return observationsVector

def GPUAllowMemoryGrowth():
    """Allow GPU memory to grow to enable parallelism on a GPU."""
    from tensorflow.compat.v1 import ConfigProto, set_random_seed
    from tensorflow.compat.v1 import Session as TFSession
    # from tensorflow.compat.v1.keras import backend as K
    from tensorflow.compat.v1.keras.backend import set_session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = TFSession(config=config)
    # K.set_session(sess)
    set_session(sess)

class LossAndErrorPrintingCallback(Callback):

  # def on_train_batch_end(self, batch, logs=None):
  #   print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  # def on_test_batch_end(self, batch, logs=None):
  #   print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_epoch_end(self, epoch, logs=None):
    try:
        print('training huber loss {:.4e}, validation mse {:.4e}, epoch {}'.format(logs['loss'], logs['val_mean_squared_error'], epoch))
    except:
        print('training mse {:.4e}, validation mse {:.4e}, epoch {}'.format(logs['loss'], logs['val_loss'], epoch))


def main():
    # try:
    env = FloPyEnv()
    GPUAllowMemoryGrowth()
    pthTensorboard = join(wrkspc, 'dev', 'tensorboard')

    suffix = ''
    if flagWeight:
        suffix += 'Weighted'
    elif not flagWeight:
        suffix += 'Unweighted'
    if flagOnInitialState:
        suffix += 'Initial'
    elif not flagOnInitialState:
        suffix += ''
    suffix += 'Particle'

    if recompileRepeatedly:
        recompileRepetitions = 100000
    elif not recompileRepeatedly:
        recompileRepetitions = 1
    # temporarily do this over and over again
    for repeat in range(recompileRepetitions):
        if recompileRepeatedly:
            time.sleep(100)

        t0recompile = time.time()

        if recompileDataset:
            # states, rewards, doneFlags, successFlags
            # t0 = time.time()
            filesPerGame = glob(join(wrkspc, 'dev', 'gameDataNew', 'env3*.p'))
            # filesPerGame = glob(join(wrkspc, 'dev', 'gameDataNew', 'env3*', 'env3*.p'))
            # print('debug files', len(filesPerGame))
            # print('time listing', t0 - time.time())
            # time.sleep(100000)
            if flagOnInitialState:
                filesPerGame += glob(join(wrkspc, 'dev', 'gameDataNewInitialsOnly', 'env3*.p'))
                # filesPerGame += glob(join(wrkspc, 'dev', 'gameDataNewInitialsOnly', 'env3*', 'env3*.p'))
            if trainGameLimit is not None:
                filesPerGame = filesPerGame[0:trainGameLimit]

            X, Y, args = [], [], []
            for file in filesPerGame:
                args.append([file, flagOnInitialState])

            cpuCount = cpu_count()
            cpuCount = 4
            with Pool(processes=cpuCount) as pool:
                dataMP = pool.starmap(collectInputAndStateFromFile, args)

            for data in dataMP:
                for entry in data[0]:
                    X.append(entry)
            for data in dataMP:
                for entry in data[1]:
                    Y.append(entry)

            X = array(X)
            Y = array(Y)
            X = X.reshape(shape(X)[0], shape(X)[1])
            Y = Y.reshape(shape(Y)[0], shape(Y)[1])

            lenInput = len(X[0])
            nPredictions = len(Y[0])
            # print('debug', 'lenInput', lenInput, 'nPredictions', nPredictions)

            filehandler = open(join(wrkspc, 'dev', 'surrogateXtrain' + suffix + '.p'), 'wb')
            dump(X, filehandler, protocol=4)
            filehandler.close()
            filehandler = open(join(wrkspc, 'dev', 'surrogateYtrain' + suffix + '.p'), 'wb')
            dump(Y, filehandler, protocol=4)
            filehandler.close()

    for i in range(nRandomModels):
        batchSize = choice(batchSizes)
        nLayers = choice(nLayers_)
        # nNeurons = randint(nNeuronsInterval[0], nNeuronsInterval[1])
        # from numpy.random import uniform, randint
        learningRate = 10**(uniform(learningRateInterval[0], learningRateInterval[1]))
        activation = choice(activations)
        kernelInitializer = choice(kernelInitializers)
        dropout = uniform(dropoutsInterval[0], dropoutsInterval[1])
        epsilon = 10**(uniform(epsilonsInterval[0], epsilonsInterval[1]))
        patience = int(10**(uniform(patiencesInterval[0], patiencesInterval[1])))
        decay = 10**(uniform(decaysInterval[0], decaysInterval[1]))

        nNeuronsLayers = [int(10**uniform(nNeuronsInterval[0], nNeuronsInterval[1])) for i in range(nLayers)]


        # temporarily fixed
        # nNeuronsLayers = [300 for i in range(nLayers)]
        nNeuronsLayersStr = ''
        for i in range(nLayers):
            nNeuronsLayersStr += str(nNeuronsLayers[i])
            if i < nLayers-1:
                nNeuronsLayersStr += 'x'

        modelName = 'bs' + str(batchSize) + 'ly' + str(nLayers) + 'n' + str(nNeuronsLayersStr) + 'lr' + str('{:.2e}'.format(learningRate)) + 'eps' + str('{:.2e}'.format(epsilon)) + activation + kernelInitializer + 'dc' + '{:.2e}'.format(decay) + 'do' + '{:.2e}'.format(dropout) + 'pat' + str(patience)
        print('training model', modelName)


        # going through test set
        if recompileDataset:

            # states, rewards, doneFlags, successFlags
            filesPerGame = glob(join(wrkspc, 'dev', 'gameDataNewTest', 'env3*.p'))
            # filesPerGame = glob(join(wrkspc, 'dev', 'gameDataNewTest', 'env3*', 'env3*.p'))
            if flagOnInitialState:
                filesPerGame += glob(join(wrkspc, 'dev', 'gameDataNewTestInitialsOnly', 'env3*.p'))
                # filesPerGame += glob(join(wrkspc, 'dev', 'gameDataNewTestInitialsOnly', 'env3*', 'env3*.p'))
            if testGameLimit is not None:
                filesPerGame = filesPerGame[0:testGameLimit]

            Xtest, Ytest, args = [], [], []
            for file in filesPerGame:
                args.append([file, flagOnInitialState])

            cpuCount = cpu_count()
            cpuCount = 4
            with Pool(processes=cpuCount) as pool:
                dataMP = pool.starmap(collectInputAndStateFromFile, args)

            for data in dataMP:
                for entry in data[0]:
                    Xtest.append(entry)
            for data in dataMP:
                for entry in data[1]:
                    Ytest.append(entry)

            Xtest = array(Xtest)
            Ytest = array(Ytest)
            Xtest = Xtest.reshape(shape(Xtest)[0], shape(Xtest)[1])
            Ytest = Ytest.reshape(shape(Ytest)[0], shape(Ytest)[1])

            lenInput = len(Xtest[0])
            nPredictions = len(Ytest[0])
            # print('debug', 'lenInput', lenInput, 'nPredictions', nPredictions)

            filehandler = open(join(wrkspc, 'dev', 'surrogateXtest' + suffix + '.p'), 'wb')
            dump(Xtest, filehandler, protocol=4)
            filehandler.close()
            filehandler = open(join(wrkspc, 'dev', 'surrogateYtest' + suffix + '.p'), 'wb')
            dump(Ytest, filehandler, protocol=4)
            filehandler.close()

        if not recompileDataset:
            filehandler = open(join(wrkspc, 'dev', 'surrogateXtest' + suffix + '.p'), 'rb')
            Xtest = load(filehandler)
            filehandler.close()
            filehandler = open(join(wrkspc, 'dev', 'surrogateYtest' + suffix + '.p'), 'rb')
            Ytest = load(filehandler)
            filehandler.close()


        if reTrain:
            if not recompileDataset:
                filehandler = open(join(wrkspc, 'dev', 'surrogateXtrain' + suffix + '.p'), 'rb')
                X = load(filehandler)
                filehandler.close()
                filehandler = open(join(wrkspc, 'dev', 'surrogateYtrain' + suffix + '.p'), 'rb')
                Y = load(filehandler)
                filehandler.close()

                lenInput = len(X[0])
                nPredictions = len(Y[0])

            if trainSamplesLimit != None:
                X = X[0:trainSamplesLimit]
                Y = Y[0:trainSamplesLimit]

            if flagWeight:
                # Instantiate an end-to-end model predicting both state of hydraulic heads and particle location
                inputState = Input(shape=(lenInput,), name='state')  # Variable-length sequence of ints
                layer = Dense(nNeurons, name='hl1', activation=activation)(inputState)
                model = Model(inputs=[inputState],
                                    outputs=[part01, part02, part03,
                                             head01, head02, head03, head04, head05,
                                             head06, head07, head08, head09, head10,
                                             head11, head12, head13, head14, head15,
                                             head16, head17, head18, head19, head20])

            if not flagWeight:
                model = createSequentialModel(lenInput, nPredictions, nNeuronsLayers, activation, kernelInitializer, nLayers, dropout)

            optimizer = Adam(learning_rate=learningRate,
                beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=True)
                #decay=decay)

            # optimizer = SGD(lr=learningRate, decay=1e-7, momentum=0.9, nesterov=True)

            model.compile(loss='mean_squared_error', optimizer=optimizer)
            # model.compile(loss='huber_loss', optimizer=optimizer, metrics=['mean_squared_error'])

            # callbacks used during optimization
            tensorboard_callback = TensorBoard(log_dir=pthTensorboard)

            # earlyStopping = EarlyStopping(monitor='val_mean_squared_error', mode='min',
            earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                patience=patience, verbose=0,
                restore_best_weights=True)

            checkpoint = ModelCheckpoint(join(wrkspc, 'dev', 'modelCheckpoints',
                modelName) + suffix + 'Weights.h5',
                verbose=0, monitor='val_loss',
                save_best_only=True, save_weights_only=True,
                mode='auto')

            # if unweighted use this
            # results = model.fit(X, [Y[:, i] for i in range(nPredictions)],

            callbacks = [earlyStopping, checkpoint, LossAndErrorPrintingCallback()]
            if useTensorboard:
                callbacks.append(tensorboard_callback)
            results = model.fit(
                x=X, y=Y,
                batch_size=batchSize, epochs=epochs, verbose=0,
                validation_split=0.5, shuffle=True,
                callbacks=callbacks,
                #workers=2,
                use_multiprocessing=True) # multiprocessing recommended https://github.com/keras-team/keras/issues/5008
            # print('time taken', t0 - time.time())

            print('training model name', modelName)

            vallossMin = min(results.history['val_loss'])
            with open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + 'History_val_loss' + '{:.2e}'.format(vallossMin) + '.p'), 'wb') as f:
                dump(results.history, f, protocol=4)

            mse = test_(model, wrkspc, suffix, Xtest, Ytest, nPredictions, env, plotRegression=False, predictInBatches=True)
            fileTestLoss = open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + '_test_loss' + '{:.2e}'.format(mse) + '.txt'), 'w') 
            fileTestLoss.write(str(mse))
            fileTestLoss.close() 

            json_config = model.to_json()
            with open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + '.json'), 'w') as json_file:
                json_file.write(json_config)

            continueSave = False
            continueSave, rank = continueAsBest(vallossMin, modelName, suffix)
            listFiles, listLosses = listExistingModelsAndLosses(suffix, sortedbyloss=True)

            if rank <= maxModelsKept:
                # removing redundant files above threshold of models to keep
                while len(listFiles) >= maxModelsKept:
                    remove(listFiles[-1])
                    print('debug removing redundant files')
                    listFiles = listFiles[:-1]
                print('keeping model', modelName+suffix, 'rank', rank)
            else:
                # remove(join(wrkspc, 'dev', 'modelCheckpoints', modelName) + suffix + '.json')
                print('removing model weights', modelName+suffix, 'rank', rank)
                remove(join(wrkspc, 'dev', 'modelCheckpoints', modelName) + suffix + 'Weights.h5')

            if rank == 1:
                json_config = model.to_json()
                with open(join(wrkspc, 'dev', 'bestModel') + suffix + '.json', 'w') as json_file:
                    json_file.write(json_config)
                model.save_weights(join(wrkspc, 'dev', 'bestModel') + suffix + 'Weights.h5')


        if not reTrain and not reLoadBestAndTest:
            # model = load_model(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + '.h5'))
            with open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + '.json')) as json_file:
                json_config = json_file.read()
            model = model_from_json(json_config)
            model.load_weights(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + 'Weights.h5'))


        if reTest or reTestOnlyBest or reLoadBestAndTest:

            if reLoadBestAndTest:
                files = glob(join(wrkspc, 'dev', 'modelCheckpoints', '*' + suffix + 'History*.p'))
                losses, idx = [], 0
                for file in files:
                    loss = float(file.split('loss')[-1].split('.p')[0])
                    losses.append(loss)
                    # if file == join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + 'History_val_loss' + f'{vallossCurrent:03f}' + '.p'):
                    #     currentModelIdx = idx
                    #     lossCurrentFound = loss
                    idx += 1

                bestModelPath = files[np.argmin(losses)].split('History')[0]
                modelName = bestModelPath.split(sep)[-1].replace(suffix, '')
                # print(files[np.argmin(losses)])
                with open(bestModelPath + '.json') as json_file:
                    json_config = json_file.read()
                model = model_from_json(json_config)
                model.load_weights(bestModelPath + 'Weights.h5')


            continueTest = True
            if reTestOnlyBest:
                continueTest, rank = continueAsBest(vallossMin, modelName, suffix)
            if continueTest:
                print('testing new model', modelName)
            if not continueTest:
                print('not testing new model', modelName)

            if continueTest:
                if testSamplesLimit != None:
                    Xtest = Xtest[0:testSamplesLimit]
                    Ytest = Ytest[0:testSamplesLimit]

                lenInput = len(Xtest[0])
                nPredictions = len(Ytest[0])


            # development: temporary see best model results

            if continueTest or reLoadBestAndTest:

                print('DEBUG reLoadBestAndTest')
                test_(model, wrkspc, suffix, Xtest, Ytest, nPredictions, env, plotRegression=True, predictInBatches=True)

                # bestModel.save(join(wrkspc, 'dev', 'modelLoadSpeedTest'))
                # t0loadModel = time.time()
                # bestModel = load_model(join(wrkspc, 'dev', 'modelLoadSpeedTest'))
                # print('debug time loading model', time.time() - t0loadModel)
                # # this is the slowest: benchmarks 1.28s, 1.69s, 1.73s, 1.25s, 1.26s

                # from tensorflow.keras.models import model_from_json
                # # Save JSON config to disk
                # json_config = bestModel.to_json()
                # with open(join(wrkspc, 'dev', 'modelLoadSpeedTest.json'), 'w') as json_file:
                #     json_file.write(json_config)
                # # Save weights to disk
                # bestModel.save_weights(join(wrkspc, 'dev', 'modelLoadSpeedTestWeights.h5'))
                # t0loadModel = time.time()
                # # Reload the model from the 2 files we saved
                # with open(join(wrkspc, 'dev', 'modelLoadSpeedTest.json')) as json_file:
                #     json_config = json_file.read()
                # bestModel = model_from_json(json_config)
                # bestModel.load_weights(join(wrkspc, 'dev', 'modelLoadSpeedTestWeights.h5'))
                # print('debug time loading model json and weights', time.time() - t0loadModel)
                # # this is the second fastest: benchmark 0.17 s

                # t0loadModel = time.time()
                # model = createSequentialModel(lenInput, nPredictions, nNeurons, activation, nLayers, dropout)
                # bestModel.load_weights(join(wrkspc, 'dev', 'modelLoadSpeedTestWeights.h5'))
                # print('debug time loading model recreate and weights', time.time() - t0loadModel)
                # # this is the fastest: benchmark 0.12 s


                # bestModel.save(join(wrkspc, 'dev', 'bestModel' + suffix + '.h5'))
                # json_config = model.to_json()
                # with open(join(wrkspc, 'dev', 'bestModel' + suffix + '.json'), 'w') as json_file:
                #     json_file.write(json_config)
                # bestModel.save_weights(join(wrkspc, 'dev', 'bestModel' + suffix + 'Weights.h5'))
                # print('debug suffix', suffix)

                # print('Sleeping.')

                # time.sleep(10000)


    # except Exception as e:
    #     print(e)


if __name__ == '__main__':
    main()