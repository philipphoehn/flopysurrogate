# from __future__ import print_function
from glob import glob
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import load_model, model_from_json, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from numpy import argmin, asarray, array, hstack, mean, shape, sort, square, sqrt
from numpy.random import uniform, randint
from os import remove
from os.path import join, sep
from pickle import dump, load
from psutil import cpu_count
from random import choice
from sklearn import preprocessing
import time

from FloPyArcade import FloPyEnv


# developed after
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/


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
flagWeight = False
useTensorboard = False

decay = 1e-6
NAGENSTEPS = 200
maxModelsKept = 1000
nRandomModels = 100000
nNeuronsInterval = [1.5, 3.2]               # log scale
learningRateInterval = [-3.5, -4.5]         # log scale
epsilonsInterval = [-10, -11]            # log scale
nFiltersInterval = [4, 128]
# batchSizes = [512, 256, 128]
batchSizes = [16384]
# batchSizes = [65536, 32768, 16384, 8192, 4096]
# activations = ['relu']
# activations = ['tanh']
# activations = ['sigmoid']
activations = ['sigmoid', 'tanh', 'relu']
# activations = ['relu', 'selu', 'elu', 'sigmoid', 'tanh']
# activations = ['sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'hard_sigmoid', 'exponential', 'linear']
kernelInitializers = ['glorot_uniform', 'he_uniform', 'random_uniform', 'lecun_uniform']
# kernelInitializers = ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal', 'random_uniform', 'random_normal', 'lecun_uniform', 'lecun_normal']
# dropoutsInterval = [0.0, 0.5]
dropoutsInterval = [0.0, 0.0]
# patiencesInterval = [0.8, 1.3]          # log scale
patiencesInterval = [1, 2]          # log scale
epochs = 1000000

n_steps = 8


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
    

    # might have to add sequences with empty values for initial predictions???


    from numpy import abs, sum

    X, y = [], []
    for i in range(len(sequences)):
        if i < n_steps-1:
            imcomplete = True

        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-45], sequences[end_ix-1, -45:]
        # this happens if particle indicates a lost game
        if sum(abs(seq_y[0:2])) != 0.0:
            X.append(seq_x)
            y.append(seq_y)

    # print('shape(array(seq_x))', shape(array(X)))
    # print('shape(array(seq_y))', shape(array(y)))
    # time.sleep(2)
    # print('---')

    return X, y
    # return array(X), array(y)

def collectInputAndStateFromFile(file, NAGENSTEPS, n_steps):

    filehandler = open(file, 'rb')
    data = load(filehandler)
    filehandler.close()

    states = data['statesNormalized']
    stresses = data['stressesNormalized']
    rewards = data['rewards']
    doneFlags = data['doneFlags']
    successFlags = data['successFlags']

    # potential padding for LSTMs
    # https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes

    # example LSTM implementation
    # https://github.com/jeffheaton/t81_558_deep_learning/blob/1f16da595603d04ee86e48b134c62e3f2f6603b2/t81_558_class_10_5_temporal_cnn.ipynb

    inputData, trueData = [], []
    lenOutputs = len(states[0][0:3] + stresses[0][:])

    # current states are particle locations + heads
    # ignoring last 4 as these are repeated in stresses (these are wellQ, wellX, wellY, wellZ)
    # stresses here are 2 boundary conditions and wellQ, wellX, wellY, wellZ
    # adds new stresses and previous stresses
    # what is this doing? NEW: adding previous states -- DOES THIS WORK? AND MAYBE IMPROVE?
    # ignoring first two stresses as these are just the constant head forcings

    inputDataTemp, trueDataTemp = {}, {}

    # len(stresses) are timesteps
    # len(timesteps) are features
    # as first three states are particle locations
    for i in range(0, 3):
        inputDataTemp['state' + str(i)] = []
        for step in range(0, NAGENSTEPS-2):
            # generating padded input per timestep
            try:
                inputDataTemp['state' + str(i)].append(states[step][i])
            except:
                inputDataTemp['state' + str(i)].append(0.0)
    for i in range(3, len(states[0][:-4])):
        inputDataTemp['state' + str(i)] = []
        for step in range(0, NAGENSTEPS-2):
            try:
                inputDataTemp['state' + str(i)].append(states[step][i])
            except:
                inputDataTemp['state' + str(i)].append(0.0)
    for i in range(0, len(stresses[0])-1):
        inputDataTemp['stress' + str(i)] = []
        for step in range(0, NAGENSTEPS-2):
            try:
                inputDataTemp['stress' + str(i)].append(stresses[step][i])
            except:
                inputDataTemp['stress' + str(i)].append(0.0)

    trueDataTemp = {}
    for i in range(0, 3):
        trueDataTemp['state' + str(i)] = []
        for step in range(0, NAGENSTEPS-2):
            try:
                trueDataTemp['state' + str(i)].append(states[step+1][i])
            except:
                trueDataTemp['state' + str(i)].append(0.0)
    trueDataTemp = {}
    # print('len(states)', len(states))
    # print(file)
    for i in range(3, len(states[0][:-4])):
        trueDataTemp['state' + str(i)] = []
        for step in range(0, NAGENSTEPS-2):
            try:
                trueDataTemp['state' + str(i)].append(states[step+1][i])
            except:
                trueDataTemp['state' + str(i)].append(0.0)
        # print(i)

    # trueDataTemp['reward'] = []
    # for step in range(0, NAGENSTEPS-2):
    #     try:
    #         trueDataTemp['reward'].append(rewards[step]/1000.)
    #     except:
    #         trueDataTemp['reward'].append(0.0)
    
    objs =  []
    for key in inputDataTemp.keys():
        if 'state' in key:
            obj = array(inputDataTemp[key])
            obj = obj.reshape((len(obj), 1))
            objs.append(obj)
    for key in inputDataTemp.keys():
        if 'stress' in key:
            obj = array(inputDataTemp[key])
            obj = obj.reshape((len(obj), 1))
            objs.append(obj)
    for key in trueDataTemp.keys():
        if 'state' in key:
            # print('key', key)
            obj = array(trueDataTemp[key])
            obj = obj.reshape((len(obj), 1))
            objs.append(obj)
    # time.sleep(100)
    # for key in trueDataTemp.keys():
    #     if 'reward' in key:
    #         obj = array(trueDataTemp[key])
    #         obj = obj.reshape((len(obj), 1))
    #         objects.append(obj)

    # later on select only those "windows" chunks that contain data


    # dataset = hstack((in_seq1, in_seq2, out_seq))
    from numpy import vstack
    dataset = hstack(tuple(objs))

    inputData, trueData = split_sequences(dataset, n_steps)
    inputData = array(inputData)
    trueData = array(trueData)

    data = [list(inputData), list(trueData)]

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
            prediction = observationsVectorToDict(predictionFlatten)
            simulated = observationsVectorToDict(Ytest[i, :])
            prediction = env.unnormalize(prediction)
            simulated = env.unnormalize(simulated)
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
    # time.sleep(10000)

    if plotRegression:
        for i in range(nPredictions):
            plt.figure(1)
            plt.subplot(211)
            sim = simsAll[i]
            pred = predsAll[i]
            plt.scatter(sim, pred, s=0.4, lw=0., marker='.', alpha=0.5, zorder=2)
            plt.plot([-20.0, 20.0], [-20.0, 20.0], lw=0.05, color='black', zorder=1)
            plt.subplot(212)
            plt.hist(sim, bins=30)
            dx = sqrt((max(sim)-min(sim))**2)
            plt.xlim(left=min(sim)-(dx*0.1), right=max(sim)+dx*0.1)
            plt.savefig(join(wrkspc, 'dev', 'bestModel' + suffix + '_pred' + str(i+1).zfill(2) + '.png'), dpi=2000)
            print('debug saving', join(wrkspc, 'dev', 'bestModel' + suffix + '_pred' + str(i+1).zfill(2) + '.png'))
            plt.close('all')
    # print('debug minmax', min(simsAll[0]), max(simsAll[0]), min(predsAll[0]), max(predsAll[0]))

    return mse

def createSequentialModel(lenInput, nPredictions, activation, kernelInitializer, nFilters):

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Conv1D, MaxPooling1D

    model = Sequential()
    model.add(Conv1D(filters=nFilters, kernel_size=3, activation=activation, input_shape=lenInput,
        strides=1, padding='valid', data_format='channels_last', dilation_rate=1, use_bias=True,
        kernel_initializer=kernelInitializer, bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=16, kernel_size=2, activation=activation))
    # model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation=activation, kernel_initializer=kernelInitializer))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(100, activation=activation, kernel_initializer=kernelInitializer))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(nPredictions, kernel_initializer=kernelInitializer))
    model.add(Activation('linear'))

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
    files = glob(join(wrkspc, 'dev', 'modelCheckpoints', '*' + suffix + '*History*.p'))
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
    
    print('debug bestModellLoss', bestModelLoss, 'vallossCurrent', vallossCurrent, 'continueFlag', continueFlag)
    return continueFlag, rankCurrent

def observationsVectorToDict(observationsVector):
    """Convert list of observations to dictionary."""
    observationsDict = {}
    observationsDict['particleCoords'] = observationsVector[:3]
    # observationsDict['rewards'] = observationsVector[3:]
    observationsDict['heads'] = observationsVector[3:]
    return observationsDict

def observationsDictToVector(observationsDict):
    """Convert dictionary of observations to list."""
    observationsVector = []
    for obs in observationsDict['particleCoords']:
        observationsVector.append(obs)
    # for obs in observationsDict['rewards']:
    #     observationsVector.append(obs)
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

    # pass

    def on_epoch_end(self, epoch, logs=None):
        print('training mse {:.4e}, validation mse {:.4e}, epoch {}'.format(logs['loss'], logs['val_loss'], epoch))


def main():
    # try:
    env = FloPyEnv()
    GPUAllowMemoryGrowth()
    pthTensorboard = join(wrkspc, 'dev', 'tensorboard')

    if flagWeight:
        suffix = 'Weighted'
    elif not flagWeight:
        suffix = 'Unweighted'
    suffix += 'CNN'

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
            if trainGameLimit is not None:
                filesPerGame = filesPerGame[0:trainGameLimit]

            X, Y, args = [], [], []
            for file in filesPerGame:
                args.append([file, NAGENSTEPS, n_steps])

            cpuCount = cpu_count()
            cpuCount = 16
            with Pool(processes=cpuCount) as pool:
                dataMP = pool.starmap(collectInputAndStateFromFile, args)

            dataX, dataY = [], []
            for data in dataMP:
                dataX += data[0]
                # for entry in data[0]:
                #     X.append(entry)
            for data in dataMP:
                dataY += data[1]
                # for entry in data[1]:
                #     Y.append(entry)

            X = array(dataX)
            Y = array(dataY)
            # X = X.reshape(shape(X)[0], shape(X)[1])
            # Y = Y.reshape(len(Y), shape(Y[0])[1])

            lenInput = np.shape(X[0,:])
            nPredictions = len(Y[0])

            filehandler = open(join(wrkspc, 'dev', 'surrogateXtrain' + suffix + '.p'), 'wb')
            dump(X, filehandler, protocol=4)
            filehandler.close()
            filehandler = open(join(wrkspc, 'dev', 'surrogateYtrain' + suffix + '.p'), 'wb')
            dump(Y, filehandler, protocol=4)
            filehandler.close()

    for i in range(nRandomModels):
        batchSize = choice(batchSizes)
        # nNeurons = randint(nNeuronsInterval[0], nNeuronsInterval[1])
        # from numpy.random import uniform, randint
        learningRate = 10**(uniform(learningRateInterval[0], learningRateInterval[1]))
        activation = choice(activations)
        kernelInitializer = choice(kernelInitializers)
        nFilters = int(uniform(nFiltersInterval[0], nFiltersInterval[1]))
        dropout = uniform(dropoutsInterval[0], dropoutsInterval[1])
        epsilon = 10**(uniform(epsilonsInterval[0], epsilonsInterval[1]))
        patience = int(10**(uniform(patiencesInterval[0], patiencesInterval[1])))

        modelName = 'bs' + str(batchSize) + 'lr' + str('{:.2e}'.format(learningRate)) + 'eps' + str('{:.2e}'.format(epsilon)) + activation + kernelInitializer + 'nf' + '{:.2e}'.format(nFilters) + 'do' + '{:.2e}'.format(dropout) + 'pat' + str(patience)
        print('training model', modelName)


        # going through test set
        if recompileDataset:

            # states, rewards, doneFlags, successFlags
            filesPerGame = glob(join(wrkspc, 'dev', 'gameDataNewTest', 'env3*.p'))
            # filesPerGame = glob(join(wrkspc, 'dev', 'gameDataNewTest', 'env3*', 'env3*.p'))
            if testGameLimit is not None:
                filesPerGame = filesPerGame[0:testGameLimit]

            Xtest, Ytest, args = [], [], []
            for file in filesPerGame:
                args.append([file, NAGENSTEPS, n_steps])

            cpuCount = cpu_count()
            cpuCount = 16
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
            # Xtest = Xtest.reshape(shape(Xtest)[0], shape(Xtest)[1])
            # Ytest = Ytest.reshape(shape(Ytest)[0], shape(Ytest)[1])

            # lenInput = len(X[0])
            # lenInput = np.shape(Xtest[0,:])
            lenInput = np.shape(Xtest[0,:])
            nPredictions = len(Ytest[0])

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

                # lenInput = len(X[0])
                lenInput = np.shape(X[0,:])
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
                model = createSequentialModel(lenInput, nPredictions, activation, kernelInitializer, nFilters)

            optimizer = Adam(learning_rate=learningRate,
                beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=True)
                # decay=decay)

            # optimizer = SGD(lr=learningRate, decay=1e-7, momentum=0.9, nesterov=True)

            # model.compile(loss='mean_squared_error', optimizer=optimizer)
            model.compile(loss='mse', optimizer=optimizer)

            # callbacks used during optimization
            tensorboard_callback = TensorBoard(log_dir=pthTensorboard)

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
            # callbacks = [LossAndErrorPrintingCallback()]
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

                # lenInput = len(Xtest[0])
                lenInput = np.shape(Xtest[0,:])
                nPredictions = len(Ytest[0])


            # development: temporary see best model results

            if continueTest or reLoadBestAndTest:

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