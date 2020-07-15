from mnist import MNIST
import numpy as np

class Datagetter:
	def __init__(self):
		return None
	def get_MNIST(self,type,printfirst=False):
		mndata = MNIST('C:/Users/Diana/Documents/Models/Database')
		if type == 'train':
			images, labels = mndata.load_training()
		if type == 'test':
			images, labels = mndata.load_testing()
		if printfirst:
			print(mndata.display(images[0]))
		return np.asarray(images), np.asarray(labels), mndata





