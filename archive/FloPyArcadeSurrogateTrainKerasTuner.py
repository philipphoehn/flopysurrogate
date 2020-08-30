from __future__ import print_function
from glob import glob
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmin, array, sort, square, sqrt
from os import remove
from os.path import join, sep
from pickle import dump, load
from random import choice
from sklearn import preprocessing
import time

from FloPyArcade import FloPyEnv

wrkspc = 'C:\\FloPyArcade'
recompileDataset = False
reTrain = False
reTest = False
reTestOnlyBest = False
reLoadBestAndTest = True
trainGameLimit = None
trainSamplesLimit = None
testSamplesLimit = 1000
flagOnInitialState = True
flagWeight = False
epochs = 10000
patience = 10
epsilon = 1e-07


# in-loop order (outer to inner): batchSizes, nLayers_, nNeurons_, learningRates, activations, dropouts
activations = ['selu', 'relu', 'sigmoid', 'tanh']
# learningRates = [0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003, 0.00002, 0.00001]
learningRates = [0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
epsilons = [1e-07, 5e-08, 1e-08, 1e-09, 1e-09]
# nNeurons_ = [1000, 900, 800, 700, 600, 500, 400, 300, 200]
# nNeurons_ = [4000, 3000, 2000]
# nNeurons_ = [1000, 500, 100]
# nNeurons_ = [300, 200, 100, 75, 50, 25, 10]
nNeurons_ = [100, 75, 50, 25, 10]
batchSizes = [32768, 16384, 8192, 4096] # 32768, 65536
nLayers_ = [4, 2]
dropouts = [0.0, 0.25, 0.5, 0.75]


def createSequentialModel(lenInput, nPredictions, nNeurons, activation, nLayers, dropout):
    model = Sequential()
    layer = Dense(nNeurons, name='hl1', activation=activation, input_dim=lenInput)
    model.add(layer)
    for i in range(nLayers-1):
        # print('layer', i, 'hl' + str(i+2))
        layer = Dense(nNeurons, name='hl' + str(i+2), activation=activation)
        model.add(layer)
        do = Dropout(dropout)
        model.add(do)
    model.add(Dense(nPredictions, activation=activation))

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

def continueAsBest(vallossCurrent, suffix):
    continueFlag = False
    bestModelLoss = 100000.
    files = glob(join(wrkspc, 'dev', 'modelCheckpoints', '*' + suffix + '*History*.p'))
    if len(files) == 0:
        continueFlag = True
    else:
        losses, idx = [], 0
        for file in files:
            loss = float(file.split('loss')[-1].split('.p')[0])
            losses.append(loss)
            if file == join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + 'History_val_loss' + f'{vallossCurrent:03f}' + '.p'):
                currentModelIdx = idx
                lossCurrentFound = loss
            idx += 1

        print('debug min(losses)', min(losses))

        st = np.sort(losses)
        rankCurrent = float([i for i, k in enumerate(st) if k == lossCurrentFound][0]) + 1
        bestModelIdx = argmin(losses)
        bestModelName = files[bestModelIdx].split(sep)[-1].split('History')[0]
        bestModelLoss = losses[bestModelIdx]
        # if the current model is better
        if vallossCurrent <= bestModelLoss:
            continueFlag = True
    
    print('debug bestModelName', bestModelName, 'bestModelLoss', bestModelLoss)

    return continueFlag, rankCurrent

def observationsVectorToDict(observationsVector):
    """Convert list of observations to dictionary."""
    observationsDict = {}
    observationsDict['particleCoords'] = observationsVector[:3]
    observationsDict['heads'] = observationsVector[3:]
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


env = FloPyEnv()
GPUAllowMemoryGrowth()
pthTensorboard = join(wrkspc, 'dev', 'tensorboard')

if flagWeight:
    suffix = 'Weighted'
elif not flagWeight:
    suffix = 'Unweighted'
if flagOnInitialState:
    suffix += 'Initial'
elif not flagOnInitialState:
    suffix += ''


if recompileDataset:
    # states, rewards, doneFlags, successFlags
    filePerGame = glob(join(wrkspc, 'dev', 'gameData', 'env3*.p'))
    if flagOnInitialState:
        filePerGame += glob(join(wrkspc, 'dev', 'gameData', 'initialsOnly', 'env3*.p'))
    if trainGameLimit is not None:
        filePerGame = filePerGame[0:trainGameLimit]

    X, Y = [], []
    for data in filePerGame:
        filehandler = open(data, 'rb')
        data = load(filehandler)
        filehandler.close()

        states = data['statesNormalized']
        stresses = data['stressesNormalized']
        rewards = data['rewards']
        doneFlags = data['doneFlags']
        successFlags = data['successFlags']

        if flagOnInitialState:
            # last 4 entries: this needs to be e.g. well Q and location
            stress = np.asarray(stresses[0])
            # predictions should only be heads, therefore particle locations (first three)
            # and well information (last four) ignored
            state = np.asarray(states[0][3:-4])
            X.append(stress)
            Y.append(state)

        if not flagOnInitialState:
            for i in range(1, len(states)-1):
                # ignoring last 4 as these are repeated in stresses, adds new stresses and previous stresses
                inputCurrent = np.asarray(states[i][:-4] + stresses[i] + stresses[i-1][-3:])
                # print(stateCurrent, len(stateCurrent))
                stateNext = np.asarray(states[i+1][:-4])
                # rewardCurrent = np.asarray(rewards[i])
                # rewardNext = np.asarray(rewards[i+1])
                X.append(inputCurrent)
                Y.append(stateNext)

                # something wrong with first and last particle location???

    X = np.asarray(X)
    Y = np.asarray(Y)
    lenInput = len(X[0])
    nPredictions = len(Y[0])

    filehandler = open(join(wrkspc, 'dev', 'surrogateXtrain' + suffix + '.p'), 'wb')
    dump(X, filehandler, protocol=4)
    filehandler.close()
    filehandler = open(join(wrkspc, 'dev', 'surrogateYtrain' + suffix + '.p'), 'wb')
    dump(Y, filehandler, protocol=4)
    filehandler.close()


nRandomModels = 1000
for i in range(nRandomModels):
    batchSize = choice(batchSizes)
    nLayers = choice(nLayers_)
    nNeurons = choice(nNeurons_)
    learningRate = choice(learningRates)
    activation = choice(activations)
    dropout = choice(dropouts)
    epsilon = choice(epsilons)

    modelName = 'bs' + str(batchSize) + 'ly' + str(nLayers) + 'n' + str(nNeurons) + 'lr' + str(learningRate) + activation + 'do' + str(dropout)
    print('training model', modelName)

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
            model = createSequentialModel(lenInput, nPredictions, nNeurons, activation, nLayers, dropout)

        optimizer = Adam(learning_rate=learningRate,
            beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

        # model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # callbacks used during optimization
        tensorboard_callback = TensorBoard(log_dir=pthTensorboard)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=patience,
            verbose=0, mode='min')
        checkpoint = ModelCheckpoint(join(wrkspc, 'dev', 'modelCheckpoints',
            modelName) + suffix + '.h5',
            verbose=0, monitor='val_loss', save_best_only=True,
            mode='auto', restore_best_weights=True)

        # if unweighted use this
        # results = model.fit(X, [Y[:, i] for i in range(nPredictions)],
        results = model.fit(
            x=X, y=Y,
            batch_size=batchSize, epochs=epochs, verbose=2,
            validation_split=0.5, shuffle=True,
            callbacks=[tensorboard_callback, earlyStopping, checkpoint],
            #workers=2,
            use_multiprocessing=True) # multiprocessing recommended https://github.com/keras-team/keras/issues/5008
        # print('time taken', t0 - time.time())

        vallossMin = min(results.history['val_loss'])
        with open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + 'History_val_loss' + f'{vallossMin:03f}' + '.p'), 'wb') as f:
            dump(results.history, f, protocol=4)

        continueSave = False
        if reTestOnlyBest:
            continueSave, rank = continueAsBest(vallossMin, suffix)
        if continueSave:
            model.save(join(wrkspc, 'dev', 'bestModel' + suffix + '.h5'))
            json_config = model.to_json()
            with open(join(wrkspc, 'dev', 'bestModel' + suffix + '.json'), 'w') as json_file:
                json_file.write(json_config)
            bestModel.save_weights(join(wrkspc, 'dev', 'bestModel.weights'))
        if rank >= 50:
            print('debug rank', rank)
            remove(join(wrkspc, 'dev', 'modelCheckpoints', modelName) + suffix + '.h5')
        # if rank >= 50:
            # remove previous rank 50 or lower (if existing)


    if not reTrain and not reLoadBestAndTest:
        model = load_model(join(wrkspc, 'dev', 'modelCheckpoints', modelName + suffix + '.h5'))


    # going through test set
    if recompileDataset:
        filePerGame = glob(join(wrkspc, 'dev', 'gameDataTest', 'env3*.p'))

        X, Y = [], []
        for data in filePerGame:
            filehandler = open(data, 'rb')
            data = load(filehandler)
            filehandler.close()

            states = data['statesNormalized']
            stresses = data['stressesNormalized']
            rewards = data['rewards']
            doneFlags = data['doneFlags']
            successFlags = data['successFlags']

            if flagOnInitialState:
                # last 4 entries: this needs to be e.g. well Q and location
                stress = np.asarray(stresses[0])
                # predictions should only be heads, therefore particle locations (first three)
                # and well information (last four) ignored
                state = np.asarray(states[0][3:-4])
                X.append(stress)
                Y.append(state)

            if not flagOnInitialState:
                for i in range(1, len(states)-1):
                    # ignoring last 4 as these are repeated in stresses, adds new stresses and previous stresses
                    inputCurrent = np.asarray(states[i][:-4] + stresses[i] + stresses[i-1][-3:])
                    stateNext = np.asarray(states[i+1][:-4])
                    X.append(inputCurrent)
                    Y.append(stateNext)

        Xtest = np.asarray(X)
        Ytest = np.asarray(Y)

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

            bestModelPath = files[np.argmin(losses)].split('History')[0] + '.h5'
            print('debug best model loaded', files[np.argmin(losses)])
            bestModel = load_model(bestModelPath)

        continueTest = True
        if reTestOnlyBest:
            continueTest, _ = continueAsBest(vallossMin, suffix)
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

        if reLoadBestAndTest:
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





            # continueFlag, rankCurrent = continueAsBest(0.2, suffix)
            # print('Sleeping.')
            # sleep(100000)

            # bestModel.save(join(wrkspc, 'dev', 'bestModel' + suffix + '.h5'))
            # json_config = model.to_json()
            # with open(join(wrkspc, 'dev', 'bestModel' + suffix + '.json'), 'w') as json_file:
            #     json_file.write(json_config)
            # bestModel.save_weights(join(wrkspc, 'dev', 'bestModel' + suffix + 'Weights.h5'))
            # print('debug suffix', suffix)

            # print('Sleeping.')

            # time.sleep(10000)





            predsAll, simsAll = [], []
            for it in range(nPredictions):
                predsAll.append([])
                simsAll.append([])

            simNorm = [[] for i in range(nPredictions)]
            predNorm = [[] for i in range(nPredictions)]
            # iterating through test samples
            for i in range(1, len(Xtest)-1):
                # actually no need to predict wellQ and wellCoords
                if reLoadBestAndTest:
                    prediction = bestModel.predict(Xtest[i, :].reshape(1, -1), batch_size=1)
                if not reLoadBestAndTest:
                    prediction = model.predict(Xtest[i, :].reshape(1, -1), batch_size=1)

                predictionFlatten = np.array(prediction[0])
                for i in range(nPredictions):
                    simNorm[i].append(predictionFlatten[i])
                    predNorm[i].append(Ytest[i, :][i])
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
                    simulated = np.array(simulated['heads'])#.flatten()
                elif not flagOnInitialState:
                    prediction = np.array(observationsDictToVector(prediction))
                    simulated = np.array(observationsDictToVector(simulated)).flatten()

                for k in range(nPredictions):
                    predsAll[k].append(prediction[k])
                    simsAll[k].append(simulated[k])

                if (i % 1000 == 0):
                    # print('predict', i, len(Xtest), 'prediction', prediction, 'simulated', simulated)
                    print('predict', i, len(Xtest))



            msesAll = []
            for i in range(nPredictions):
                sim = simsAll[i]
                pred = predsAll[i]
                mse = (square(array(simNorm[i]) - array(predNorm[i]))).mean(axis=None)
                msesAll.append(mse)

                plt.figure(1)
                plt.subplot(211)
                plt.scatter(sim, pred, s=0.4, lw=0., marker='.')
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
                plt.savefig(join(wrkspc, 'dev', 'bestModel' + suffix + '_pred' + str(i+1).zfill(2) + '.png'), dpi=1000)
                print('debug saving', join(wrkspc, 'dev', 'bestModel' + suffix + '_pred' + str(i+1).zfill(2) + '.png'))
                plt.close('all')

            mseAll = np.mean(msesAll)
            print('debug mse', mseAll)