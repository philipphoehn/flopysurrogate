from __future__ import print_function
from glob import glob
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
from os.path import join
from pickle import dump, load
from sklearn import preprocessing
import time

from FloPyArcade import FloPyEnv

wrkspc = 'C:\\FloPyArcade'
weighted = True
recompileDatasetTrain = False
recompileDatasetTest = True
reTrain = True
reTest = True
modelName = 'surrogateModelStates_ar3x500lr1e-02bs32784ep10000pat10'
# modelName = 'surrogateModelStatesWeighted2'
trainGameLimit = None
trainSamplesLimit = None
testSamplesLimit = 10000
learningRate = 0.0006
batchSize = 32784
epochs = 10000
patience = 10
nNeurons = 500
# bs32784n100lr0.0004unweightedHistory_val_loss0.000646


learningRates = [0.0004]
nNeurons_ = [100]
batchSizes = [32784]
# learningRates = [0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002]
# nNeurons_ = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
# batchSizes = [16384, 8192, 4096] # 32768, 65536


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

if recompileDatasetTrain:
    # states, rewards, doneFlags, successFlags
    filePerGame = glob(join(wrkspc, 'dev', 'gameData', 'env3*.p'))
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
        for i in range(1, len(states)-1):
            # print(states[0])
            # print(states[1])
            # print(states[2])
            # time.sleep(4)
            # ignoring last 4 as these are repeated in stresses, adds new stresses and previous stresses
            inputCurrent = np.asarray(states[i][:-4] + stresses[i] + stresses[i-1][-3:])
            # time.sleep(1)
            # print(stateCurrent, len(stateCurrent))
            # to be predicted state excludes stresses


            # something wrong with first and last particle location???
            stateNext = np.asarray(states[i+1][:-4])
            rewardCurrent = np.asarray(rewards[i])
            rewardNext = np.asarray(rewards[i+1])
            X.append(inputCurrent)
            Y.append(stateNext)

    X = np.asarray(X)
    Y = np.asarray(Y)
    lenInput = len(X[0])
    nPredictions = len(Y[0])

    filehandler = open(join(wrkspc, 'dev', 'surrogateXtrain.p'), 'wb')
    dump(X, filehandler, protocol=4)
    filehandler.close()
    filehandler = open(join(wrkspc, 'dev', 'surrogateYtrain.p'), 'wb')
    dump(Y, filehandler, protocol=4)
    filehandler.close()


for batch_size in batchSizes:
    for nNeurons in nNeurons_:
        for learningRate in learningRates:
            modelName = 'bs' + str(batchSize) + 'n' + str(nNeurons) + 'lr' + str(learningRate)


            if reTrain:
                if not recompileDatasetTrain:
                    filehandler = open(join(wrkspc, 'dev', 'surrogateXtrain.p'), 'rb')
                    X = load(filehandler)
                    filehandler.close()
                    filehandler = open(join(wrkspc, 'dev', 'surrogateYtrain.p'), 'rb')
                    Y = load(filehandler)
                    filehandler.close()

                    lenInput = len(X[0])
                    nPredictions = len(Y[0])

                if trainSamplesLimit != None:
                    X = X[0:trainSamplesLimit]
                    Y = Y[0:trainSamplesLimit]

                inputState = Input(shape=(lenInput,), name='state')  # Variable-length sequence of ints

                hl1 = Dense(nNeurons, name='hl1', activation='relu')(inputState)
                hl2 = Dense(nNeurons, name='hl2', activation='relu')(hl1)
                hl3 = Dense(nNeurons, name='hl3', activation='relu')(hl2)

                # Stick a logistic regression for priority prediction on top of the features
                part01 = Dense(1, name='part01')(hl3)
                part02 = Dense(1, name='part02')(hl3)
                part03 = Dense(1, name='part03')(hl3)
                head01 = Dense(1, name='head01')(hl3)
                head02 = Dense(1, name='head02')(hl3)
                head03 = Dense(1, name='head03')(hl3)
                head04 = Dense(1, name='head04')(hl3)
                head05 = Dense(1, name='head05')(hl3)
                head06 = Dense(1, name='head06')(hl3)
                head07 = Dense(1, name='head07')(hl3)
                head08 = Dense(1, name='head08')(hl3)
                head09 = Dense(1, name='head09')(hl3)
                head10 = Dense(1, name='head10')(hl3)
                head11 = Dense(1, name='head11')(hl3)
                head12 = Dense(1, name='head12')(hl3)
                head13 = Dense(1, name='head13')(hl3)
                head14 = Dense(1, name='head14')(hl3)
                head15 = Dense(1, name='head15')(hl3)
                head16 = Dense(1, name='head16')(hl3)
                head17 = Dense(1, name='head17')(hl3)
                head18 = Dense(1, name='head18')(hl3)
                head19 = Dense(1, name='head19')(hl3)
                head20 = Dense(1, name='head20')(hl3)

                # Instantiate an end-to-end model predicting both state of hydraulic heads and particle location
                model = Model(inputs=[inputState],
                                    outputs=[part01, part02, part03,
                                             head01, head02, head03, head04, head05,
                                             head06, head07, head08, head09, head10,
                                             head11, head12, head13, head14, head15,
                                             head16, head17, head18, head19, head20])

                # model = Sequential()
                # model.add(Dense(input_dim=lenInput, units=100))
                # model.add(Activation('relu'))
                # model.add(Dense(input_dim=100, units=100))
                # model.add(Activation('relu'))
                # model.add(Dense(input_dim=100, units=100))
                # model.add(Activation('relu'))
                # model.add(Dense(input_dim=100, units=lenInput))
                # model.add(Activation('relu'))

                optimizer = Adam(learning_rate=learningRate,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

                sampleweightsAll = []
                for i in range(nPredictions):
                    sampleWeights = calculate_weightsforregression(Y[:,i], nbins=30)
                    sampleweightsAll.append(sampleWeights)
                # sampleweightsAll = list(np.array(sampleweightsAll).reshape(1, -1))
                # print('debug shape', sampleweightsAll.shape())

                # model.compile(loss='mean_squared_error', optimizer=optimizer)
                model.compile(loss='mean_squared_error', optimizer=optimizer, sample_weight_mode=['samplewise' for i in range(nPredictions)])

                tensorboard_callback = TensorBoard(log_dir=pthTensorboard)
                earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, 
                    verbose=0, mode='min')
                checkpoint = ModelCheckpoint(join(wrkspc, 'dev', 'modelCheckpoints',
                    modelName) + '.h5',
                    verbose=0, monitor='val_loss', save_best_only=True,
                    mode='auto', restore_best_weights=True)

                results = model.fit(X, [Y[:, i] for i in range(nPredictions)], sample_weight=sampleweightsAll,
                    batch_size=batch_size, epochs=epochs, validation_split=0.5, shuffle=True,
                    callbacks=[tensorboard_callback, earlyStopping, checkpoint],
                    verbose=1)
                # print('time taken', t0 - time.time())

                vallossMin = min(results.history['val_loss'])
                with open(join(wrkspc, 'dev', 'modelCheckpoints', modelName + 'History_val_loss' + f'{vallossMin:03f}' + '.p'), 'wb') as f:
                    dump(results.history, f, protocol=4)

                t0 = time.time()
                # print(prediction)
                # print('time taken', t0 - time.time())
                # print('example sim', 60*(t0 - time.time()))

                model.save(join(wrkspc, 'dev', modelName + '.h5'))

            if not reTrain:
                model = load_model(join(wrkspc, 'dev', modelName + '.h5'))


            # going through test set
            if recompileDatasetTest:
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
                    for i in range(len(states)-1):
                        inputCurrent = np.asarray(states[i][:-4] + stresses[i] + stresses[i-1][-3:])
                        stateNext = np.asarray(states[i+1][:-4])
                        rewardCurrent = np.asarray(rewards[i])
                        rewardNext = np.asarray(rewards[i+1])
                        X.append(inputCurrent)
                        Y.append(stateNext)

                Xtest = np.asarray(X)
                Ytest = np.asarray(Y)

                filehandler = open(join(wrkspc, 'dev', 'surrogateXtest.p'), 'wb')
                dump(Xtest, filehandler, protocol=4)
                filehandler.close()
                filehandler = open(join(wrkspc, 'dev', 'surrogateYtest.p'), 'wb')
                dump(Ytest, filehandler, protocol=4)
                filehandler.close()

            if not recompileDatasetTest:
                filehandler = open(join(wrkspc, 'dev', 'surrogateXtest.p'), 'rb')
                Xtest = load(filehandler)
                filehandler.close()
                filehandler = open(join(wrkspc, 'dev', 'surrogateYtest.p'), 'rb')
                Ytest = load(filehandler)
                filehandler.close()


            if reTest:

                if testSamplesLimit != None:
                    Xtest = Xtest[0:testSamplesLimit]
                    Ytest = Ytest[0:testSamplesLimit]
                lenInput = len(Xtest[0])
                nPredictions = len(Ytest[0])
                print('big debug nPredictions', nPredictions)

                predsAll, simsAll = [], []
                for it in range(nPredictions):
                    predsAll.append([])
                    simsAll.append([])

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

                # iterating through test samples
                for i in range(1, len(Xtest)-1):
                    # actually no need to predict wellQ and wellCoords
                    tinit = time.time()
                    t0 = time.time()
                    prediction = model.predict(Xtest[i, :].reshape(1, -1), batch_size=1)
                    tpredict = time.time() - t0
                    # print('time predict', tpredict)
                    t0 = time.time()
                    predictionFlatten = np.array(prediction).flatten()
                    # print('time flatten', time.time() - t0)
                    t0 = time.time()
                    prediction = observationsVectorToDict(predictionFlatten)
                    # print('big debug prediction1', prediction)
                    # print('time to dict', time.time() - t0)
                    t0 = time.time()
                    prediction = env.unnormalize(prediction)
                    # print('big debug prediction2', prediction)
                    # print('time unnormalize', time.time() - t0)
                    t0 = time.time()
                    prediction = np.array(observationsDictToVector(prediction))
                    # print('time to vector', time.time() - t0)

                    simulated = observationsVectorToDict(Ytest[i, :])
                    simulated = env.unnormalize(simulated)
                    simulated = np.array(observationsDictToVector(simulated)).flatten()

                    for k in range(nPredictions):
                        predsAll[k].append(prediction[k])
                        simsAll[k].append(simulated[k])

                    if (i % 1000 == 0):
                        # print('predict', i, len(Xtest), 'prediction', prediction, 'simulated', simulated)
                        print('predict', i, len(Xtest))

                for i in range(nPredictions):
                    plt.figure(1)
                    plt.subplot(211)
                    sim = simsAll[i]
                    pred = predsAll[i]
                    plt.scatter(sim, pred, s=0.4, lw=0., marker='.')
                    if i > 2:
                        plt.xlim(left=0, right=10.5)
                        plt.ylim(bottom=0, top=10.5)
                    else:
                        dx = sqrt((max(sim)-min(sim))**2)
                        plt.xlim(left=min(sim)-(dx*0.1), right=max(sim)+dx*0.1)
                        plt.ylim(bottom=min(sim)-(dx*0.1), top=max(sim)+dx*0.1)
                    plt.subplot(212)
                    plt.hist(sim, bins=30)
                    if i > 2:
                        plt.xlim(left=0, right=10.5)
                    else:
                        dx = sqrt((max(sim)-min(sim))**2)
                        plt.xlim(left=min(sim)-(dx*0.1), right=max(sim)+dx*0.1)
                    plt.savefig(join(wrkspc, 'dev', modelName + 'pred' + str(i+1).zfill(2) + '.png'), dpi=1000)
                    plt.close('all')