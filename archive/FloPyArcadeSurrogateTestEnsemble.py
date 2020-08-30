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
from numpy import add, arange, argmin, array, divide, mean, multiply, ones, shape, sort, square, sqrt, zeros
from numpy.random import uniform, randint
from os import remove
from os.path import join, sep
from pickle import dump, load
from psutil import cpu_count
from random import choice
from sklearn import preprocessing
import time

from itertools import combinations
from numpy import linspace

from FloPyArcade import FloPyEnv, FloPyAgent

wrkspc = 'C:\\FloPyArcade'
flagOnInitialState = False
# flagWeight = False
# useTensorboard = False
predictBatchSize = 16384
testEnsemble = True
minbestEnsembleMembers = 2
maxbestEnsembleMembers = 3
nEnsembleCandidates = 3
everyNWeight = 10
varyWeightsPerPrediction = False

maxTasksPerWorker = 20


# suffix = 'UnweightedParticle'
suffix = 'UnweightedHeads'
# suffix = 'UnweightedInitial'
# suffix = 'Unweighted'


def testEnsemble_(ensemble, wrkspc, suffix, Xtest, Ytest, nPredictions, predictionSelectIdx,
    envDefinitions, weightsPerPrediction=None,
    weightsSameAllPredictions=None, plotRegression=False):

    t0 = time.time()
    if weightsPerPrediction == None and weightsSameAllPredictions == None:
        weights = [divide(ones(len(ensemble)), len(ensemble)) for i in range(nPredictions)]
    if weightsPerPrediction != None:
        weights = [[weightsPerPrediction[k] for k in range(len(weightsPerPrediction))] for i in range(nPredictions)]
    if weightsSameAllPredictions != None:
        weights = weightsSameAllPredictions

    simsAll = [[] for i in range(np.shape(Ytest)[0])]
    simsNorm = [[] for i in range(nPredictions)]
    for i in range(np.shape(Ytest)[0]):
        for k in range(nPredictions):
            simsNorm[k].append(Ytest[i, :][k])
        if flagOnInitialState:
            simulated = {'heads': list(Ytest[i, :])}
        elif not flagOnInitialState:
            simulated = observationsVectorToDict(Ytest[i, :])
        simulated = unnormalizeFromEnvDefinitions(simulated, envDefinitions)
        if flagOnInitialState:
            simulated = np.array(simulated['heads'])
        elif not flagOnInitialState:
            simulated = np.array(observationsDictToVector(simulated))
        for k in range(nPredictions):
            simsAll[k].append(simulated[k])

    predsAllEnsemble = [list(zeros(np.shape(Ytest)[0])) for i in range(nPredictions)]
    predsNormEnsemble = [list(zeros(np.shape(Ytest)[0])) for i in range(nPredictions)]
    msesEnsemble = []
    for modelIdx in range(len(ensemble)):
        # print('debug modelIdx',  modelIdx)
        with open(ensemble[modelIdx]) as json_file:
            json_config = json_file.read()
        modelLoaded = model_from_json(json_config)
        modelLoaded.load_weights(ensemble[modelIdx].replace('.json', 'Weights.h5'))

        t0 = time.time()
        # iterating through test samples
        # actually no need to predict wellQ and wellCoords
        predictionBatch = modelLoaded.predict(Xtest, use_multiprocessing=True, batch_size=predictBatchSize)

        predsAll = [[] for i in range(nPredictions)]
        predsNorm = [[] for i in range(nPredictions)]
        for i in range(np.shape(Ytest)[0]):
            predictionFlatten = np.array(predictionBatch[i, :])
            for k in range(nPredictions):
                predsNorm[k].append(predictionFlatten[k])
            if flagOnInitialState:
                # for initial state prediction, only heads are predicted
                prediction = {'heads': predictionFlatten}
            elif not flagOnInitialState:
                prediction = observationsVectorToDict(predictionFlatten)
            prediction = unnormalizeFromEnvDefinitions(prediction, envDefinitions)
            if flagOnInitialState:
                prediction = np.array(prediction['heads'])
            elif not flagOnInitialState:
                prediction = np.array(observationsDictToVector(prediction))

            for k in range(nPredictions):
                predsAll[k].append(prediction[k])

        for predIdx in range(nPredictions):
            predsAllEnsemble[predIdx] = add(predsAllEnsemble[predIdx], multiply(predsAll[predIdx], weights[predIdx][modelIdx]))
            predsNormEnsemble[predIdx] = add(predsNormEnsemble[predIdx], multiply(predsNorm[predIdx], weights[predIdx][modelIdx]))

    # calculating mean squared error
    msesAll = []
    for i in range(nPredictions):
        sim = simsNorm[i]
        # these are weighted
        pred = predsNormEnsemble[i]
        msePred = (square(array(sim) - array(pred))).mean(axis=None)
        msesAll.append(msePred)
    mseEnsemble = mean(msesAll)

    # print('debug mseIdvEnsemble', mseEnsemble)
    # print('--------------------')
    # print('debug mse ensemble-weighted', mseIdvEnsemble)
    # print('time batch test preditions', time.time() - t0)

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

    # print('time spent MSE', t0 - time.time())
    # ca. 16 h for 500000 with 16 cpus

    return mseEnsemble

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

def unnormalizeFromEnvDefinitions(data, envDefinitions):
    from numpy import multiply

    keys = data.keys()
    if 'particleCoords' in keys:
        data['particleCoords'] = multiply(data['particleCoords'],
            envDefinitions['minX'] + envDefinitions['extentX'])
    if 'heads' in keys:
        data['heads'] = multiply(data['heads'],
            envDefinitions['maxH'])
    if 'wellQ' in keys:
        data['wellQ'] = multiply(data['wellQ'], envDefinitions['minQ'])
    if 'wellCoords' in keys:
        data['wellCoords'] = multiply(data['wellCoords'], envDefinitions['minX'] + envDefinitions['extentX'])
    return data

def listExistingModelsAndLosses(suffix, sortedbyloss=True):
    filesJSON = glob(join(wrkspc, 'dev', 'modelCheckpoints', '*' + suffix + '.json'))
    i, filesLoss = 0, []
    for fileJSON in filesJSON:
        # print('debug fileJSON', fileJSON)
        filesHistory = glob(fileJSON.replace('.json', '') + 'History*.p')
        if len(filesHistory) > 0:
            i, fileHistory = 0, filesHistory[0]
            # removing redudant history files
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

    return filesJSON, filesLoss

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

def returnWeightCombinations(nEnsembles):
    # https://stackoverflow.com/questions/22841469/all-possible-combinations-of-portfolio-weights/22841756#22841756

    input_list = np.linspace(0.0, 1.0, num=10) # 26
    valid_combinations = []
    for comb_length in range(1,len(input_list)+1):
        possible_combinations = combinations(input_list,comb_length)
        for comb in possible_combinations:
            if sum(comb) ==1:
                comb = list(comb)
                if len(comb) == nEnsembles:
                    valid_combinations.append(comb)
    return valid_combinations

def generateAllWeightCombinations(nEnsembles):
    """
    Generate all partitions of range(n) into k pieces.
    Inspired by: https://stackoverflow.com/questions/51584223/python3-what-is-the-most-efficient-way-to-calculate-all-permutations-of-two-lis
    """
    import itertools

    n = 100
    for c in itertools.combinations(range(n+nEnsembles-1), nEnsembles-1):
        yield list(divide(tuple(y-x-1 for x, y in zip((-1,) + c, c + (n+nEnsembles-1,))), 100))

def listEnsembleCombinations(suffix, nEnsembleCandidates, bestEnsembleMembers):

    ensembleCandidates, ensembleCandidatesLosses = listExistingModelsAndLosses(suffix, sortedbyloss=True)
    ensembleCandidates = ensembleCandidates[:nEnsembleCandidates]
    # print('DEBUG ensembleMembersList', ensembleCandidates)
    ensembleCandidatesLosses = ensembleCandidatesLosses[:nEnsembleCandidates]
    ensembleCombinations_ = combinations(ensembleCandidates, bestEnsembleMembers)
    ensembleCombinations = [ensemble for ensemble in ensembleCombinations_]

    print('DEBUG nEnsembleCandidates', nEnsembleCandidates)
    # print('DEBUG ensembleMembersList', ensembleCandidates)

    return ensembleCombinations, ensembleCandidates, ensembleCandidatesLosses

def createWeightSamples(minbestEnsembleMembers, maxbestEnsembleMembers,
    everyNWeight, nEnsembleCandidates):

    # generating weight combinations and determining number of weight checks
    ensembleMembersList = arange(minbestEnsembleMembers, maxbestEnsembleMembers+1)
    weightCombinationsPernMembers, countWeights = [], 0
    for bestEnsembleMembers in ensembleMembersList:
        weightCombinations = []
        weightCombinations += [weights for weights in generateAllWeightCombinations(bestEnsembleMembers)][0::everyNWeight]
        weightCombinationsPernMembers.append(weightCombinations)
        ensembleCombinations, ensembleCandidates, ensembleCandidatesLosses = listEnsembleCombinations(suffix, nEnsembleCandidates, bestEnsembleMembers)
        countWeights += len(weightCombinations) * len(ensembleCombinations)
        
        # print('weights combination', weightCombinations)
        # print('bestEnsembleMembers', bestEnsembleMembers)
        # print('len(weightCombinations)', len(weightCombinations))
        # print('len(ensembleCombinations)', len(ensembleCombinations))

    return weightCombinationsPernMembers, ensembleMembersList, countWeights

def evaluatePredictionWeights(Xtest, Ytest, predictionSelected, ensembleMembersList, weightCombinationsPernMembers,
    suffix, nEnsembleCandidates, envDefinitions, countWeights, countWeightsChecked, cpuCount,
    varyWeightsPerPrediction=False):

    lenInput, nPredictions = len(Xtest[0]), len(Ytest[0])
    countnMembersChecked = 0
    mses, weightCombinations, ensembles = [], [], []

    for bestEnsembleMembers in ensembleMembersList:
        ensembleCombinations, ensembleCandidates, ensembleCandidatesLosses = listEnsembleCombinations(suffix, nEnsembleCandidates, bestEnsembleMembers)

        for ensemble in ensembleCombinations:
            argsAll, t0 = [], time.time()
            # generating weights
            for combination in weightCombinationsPernMembers[countnMembersChecked]:

                # variable weights option goes here, right?

                weightsSameAllPredictions = [[combination[k] for k in range(len(combination))] for _ in range(nPredictions)]
                weightCombinations.append(weightsSameAllPredictions)
                ensembles.append(ensemble)
                argsAll.append([ensemble, wrkspc, suffix, Xtest, Ytest, nPredictions, predictionSelected, envDefinitions, None, weightsSameAllPredictions, False])
            # print('time spent assembling weights', t0 - time.time())
            # processing in chunks using multiprocessing
            chunksTotal = FloPyAgent().yieldChunks(argsAll, cpuCount*maxTasksPerWorker)
            msesTemp, t0 = [], time.time()
            for chunk in chunksTotal:
                p = Pool(processes=cpuCount)
                msesTempPool = p.starmap_async(testEnsemble_, chunk)
                msesTempPool.wait()
                msesTempToAdd = msesTempPool.get()
                msesTemp += msesTempToAdd
                countWeightsChecked += len(msesTempToAdd)
                
                # estimate speed from first 100 samples?
                print('progress', countWeightsChecked, '/', countWeights)
                p.close(); p.join(); p.terminate()
            for mse in msesTemp:
                mses.append(mse)
        countnMembersChecked += 1

    return mses, weightCombinations, ensembles, ensembleCandidates, ensembleCandidatesLosses, countWeightsChecked

def main():
    env = FloPyEnv()
    cpuCount = cpu_count()
    cpuCount = 8

    filehandler = open(join(wrkspc, 'dev', 'surrogateXtest' + suffix + '.p'), 'rb')
    Xtest = load(filehandler)
    filehandler.close()
    filehandler = open(join(wrkspc, 'dev', 'surrogateYtest' + suffix + '.p'), 'rb')
    Ytest = load(filehandler)
    filehandler.close()

    envDefinitions = {}
    envDefinitions['minX'], envDefinitions['extentX'] = env.minX, env.extentX
    envDefinitions['maxH'], envDefinitions['minQ'] = env.maxH, env.minQ

    weightCombinationsPernMembers, ensembleMembersList, countWeights = createWeightSamples(
        minbestEnsembleMembers, maxbestEnsembleMembers, everyNWeight, nEnsembleCandidates)
    if varyWeightsPerPrediction:
        nPredictions = len(Ytest[0])
        countWeights *= nPredictions
    print('# weights to check', countWeights)
    # print(weightCombinationsPernMembers)

    countWeightsChecked = 0
    if not varyWeightsPerPrediction:
        # varying weights per ensemble model, keeping weights constant per prediction
        mses, weightCombinations, ensembles, ensembleCandidates, ensembleCandidatesLosses, _ = evaluatePredictionWeights(
            Xtest, Ytest, 'all', ensembleMembersList, weightCombinationsPernMembers, suffix,
            nEnsembleCandidates, envDefinitions, countWeights, countWeightsChecked, cpuCount,
            varyWeightsPerPrediction)
        print('mses', mses)
        mseLowestIdx = argmin(mses)

        with open(ensembleCandidates[0]) as json_file:
            json_config = json_file.read()
        modelBest = model_from_json(json_config)
        modelBest.load_weights(ensembleCandidates[0].replace('.json', 'Weights.h5'))
        mse_bestCandidateAlone = test_(modelBest, wrkspc, suffix, Xtest, Ytest, nPredictions=len(Ytest[0]), env=env, plotRegression=False, predictInBatches=True)

        print('best ensemble member weights', weightCombinations[mseLowestIdx])
        print('best ensemble member test loss', mses[mseLowestIdx])
        # print('ensembleCandidates', ensembleCandidates)
        # print('ensembleCandidatesLosses', ensembleCandidatesLosses)
        print('best single-model loss', mse_bestCandidateAlone)
        filehandler = open(join(wrkspc, 'dev', suffix + 'BestEnsembleWeights.p'), 'wb')
        dump(weightCombinations[mseLowestIdx], filehandler)
        filehandler.close()
        filehandler = open(join(wrkspc, 'dev', suffix + 'BestTestLossEnsembleWeighted.p'), 'wb')
        dump(mses[mseLowestIdx], filehandler)
        filehandler.close()
        filehandler = open(join(wrkspc, 'dev', suffix + 'BestEnsembleMembersPaths.p'), 'wb')
        dump(ensembles[mseLowestIdx], filehandler)
        filehandler.close()

    if varyWeightsPerPrediction:
        # varying weights per ensemble model, while varying weights per prediction
        # note: this might be very expensive to compute
        mses = [[0 for _ in range(nPredictions)]]
        weightCombinations = [[[] for _ in range(nPredictions)]]
        ensembles = [[0 for _ in range(nPredictions)]]
        # mses is list of all ensemble mses

        msesPred, weightCombinationsPred, ensemblesPred, ensembleCandidates, ensembleCandidatesLosses, countWeightsChecked = evaluatePredictionWeights(
            Xtest, Ytest, 'all', ensembleMembersList, weightCombinationsPernMembers, suffix,
            nEnsembleCandidates, envDefinitions, countWeights, countWeightsChecked, cpuCount,
            varyWeightsPerPrediction)

        for p in range(nPredictions):
            msesPred, weightCombinationsPred, ensemblesPred, ensembleCandidates, ensembleCandidatesLosses, countWeightsChecked = evaluatePredictionWeights(
                Xtest, Ytest, 'all', ensembleMembersList, weightCombinationsPernMembers, suffix,
                nEnsembleCandidates, envDefinitions, countWeights, countWeightsChecked, cpuCount)
            print('countWeightsChecked', countWeightsChecked)
            # for i in range()
            mses = add(mses, msesPred)
            # mses += msesPred
            # # mses is wrong?
            # weightCombinations += weightCombinationsPred[p]
            # ensembles += ensemblesPred
            print('lowest mse so far', min(mses))
            print('best single-model loss', ensembleCandidatesLosses[0])
        # eventually average mse and select
        for p in range(nPredictions):
            mseLowestIdx = argmin(mses[p])
            weightCombinationsBest = []
            ensemblesBest = []


            # cant test individually per prediction
            # will have to test for best ensemble with varying prediction weights
            # dont vary ensemble members per prediction


if __name__ == '__main__':
    main()