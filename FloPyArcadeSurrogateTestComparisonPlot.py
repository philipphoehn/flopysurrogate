from __future__ import print_function
from glob import glob
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
import os
from os.path import join
from pickle import dump, load
from sklearn import preprocessing
import time
from time import sleep
from FloPyArcade import FloPyEnv

plt.switch_backend('TKAgg')

import matplotlib
import matplotlib.colors as colors
import numpy as np


nSamples = 1000


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# wrkspc = 'C:\\FloPyArcade'
wrkspc = os.path.dirname(os.path.realpath(__file__))


# times = []
# for i in range(nSamples):
# 	env = FloPyEnv(initWithSolution=True)
# 	times.append(env.timeInitialization)
# plt.hist(times)
# plt.show()

env = FloPyEnv(initWithSolution=True)

def unnormalizeInitial(data, env):
    from numpy import multiply
    data = multiply(data, env.maxH)

    return data


model = load_model(join(wrkspc, 'dev', 'surrogateEnv3_sigmoid' + '.h5'))

Xtest = np.load(join(wrkspc, 'dev', 'surrogateInitialXtrain.npy'))
Ytest = np.load(join(wrkspc, 'dev', 'surrogateInitialYtrain.npy'))


Xtest = Xtest[0:nSamples]
Ytest = Ytest[0:nSamples]

# filehandler = open(join(wrkspc, 'dev', 'surrogateInitialXtrain.p.npy'), 'rb')
# Xtest = load(filehandler)
# filehandler.close()
# filehandler = open(join(wrkspc, 'dev', 'surrogateInitialYtrain.p.npy'), 'rb')
# Ytest = load(filehandler)
# filehandler.close()


# collecting runtimes of model
# times = []
# for i in range(nSamples):
# 	t0 = time.time()
# 	pred = model.predict(Xtest[i].reshape(1, -1))
# 	pred = unnormalizeInitial(pred[0], env)
# 	pred = pred.reshape(100,100)
# 	times.append(time.time()-t0)
# plt.hist(times)
# plt.show()


for exampleIdx in range(nSamples):

	print('input', Xtest[exampleIdx])

	t0 = time.time()
	pred = model.predict(Xtest[exampleIdx].reshape(1, -1))
	print('predict time', time.time()-t0)

	pred = unnormalizeInitial(pred[0], env)
	pred = pred.reshape(100,100)

	true = Ytest[exampleIdx]
	true = unnormalizeInitial(true, env)
	true = true.reshape(100,100)

	diff = np.subtract(pred, true)


	print(np.shape(pred))
	print(pred)
	print(min(diff.flatten()), max(diff.flatten()))


	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)
	X, Y = np.meshgrid(X, Y)


	fig, axes = plt.subplots(1,3)

	from mpl_toolkits.axes_grid1 import make_axes_locatable

	predMin, predMax = np.min([list(pred.flatten().flatten()) + list(true.flatten().flatten())]), np.max([list(pred.flatten()) + list(true.flatten())])
	print(predMin, predMax)

	divider0 = make_axes_locatable(axes[0])
	cax0 = divider0.append_axes('right', size='15%', pad=0.05)
	im = axes[0].imshow(pred, cmap='jet', vmin=predMin, vmax=predMax)
	plt.colorbar(mappable=im, cax=cax0, ax=axes[0])

	divider1 = make_axes_locatable(axes[1])
	cax1 = divider1.append_axes('right', size='15%', pad=0.05)
	im = axes[1].imshow(true, cmap='jet', vmin=predMin, vmax=predMax)
	plt.colorbar(mappable=im, cax=cax1, ax=axes[1])

	errorMin, errorMax, errorMid = np.min(diff.flatten()), np.max(diff.flatten()), np.mean(diff.flatten())
	divider2 = make_axes_locatable(axes[2])
	cax2 = divider2.append_axes('right', size='15%', pad=0.05)
	im = axes[2].imshow(diff, cmap='coolwarm',
		clim=(errorMin, errorMax), norm=MidpointNormalize(midpoint=errorMid, vmin=errorMin, vmax=errorMax))
	plt.colorbar(mappable=im, cax=cax2, ax=axes[2])

	# scale colorbar

	# fig.colorbar()
	plt.show()


# run long particle tracks on this field!