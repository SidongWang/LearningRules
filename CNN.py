import numpy as np
from scipy.signal import convolve
import keyboard
import datetime
#activation functions
def sigmoid(x):
	return 1/(1+np.exp(-x))
def tanh(x):
	return np.tanh(x)
def reLU(x):
	return np.maximum(0,x)
def preLU(x):
	return np.maximum(0.01*x,x)
#activation function gradients
def activation_grad(function,x):
	if function == sigmoid:
		return sigmoid(x)*(1-sigmoid(x))
	elif function == tanh:
		return 1-tanh(x)**2
	elif function == reLU:
		return x>=0
	elif function == preLU:
		return 0.01+(x>=0)*(1-0.01)
def probability(output,y):
	return np.exp(output[y])/np.sum(np.exp(output))
#loss functions
def multiclass_SVM_loss(output,y):
	margins = np.maximum(0,output-output[y]+1)
	margins[y]=0
	return np.sum(margins)
def softmax_loss(output,y):
	return -np.log(probability(output,y))
#loss function gradients
def loss_grad(function,output,y):
	if function == softmax_loss:
		all_exp = np.exp(output)
		res = all_exp/np.sum(all_exp)
		res[y] -= 1
		return res
	if function == multiclass_SVM_loss:
		res = 0*output
		resy = 0
		for i in range(0,len(output)):
			if output[i]-output[y]+1>0:
				res[i]=1
				resy -= 1
		res[y] = resy
		return res
class Trainer():
	def __init__(self,reg_coeff,reg_type,update_type):
		self.reg_coeff = reg_coeff
		self.reg_type = reg_type
		self.update_type = update_type
	def statistics(self,model,precision = 1e4):
		loss = self.loss
		all_W = model.get_W()
		Wmean = np.mean(all_W)
		Wstd = np.std(all_W)
		all_b = model.get_b()
		bmean = np.mean(all_b)
		bstd = np.std(all_b)
		return {'loss':int(precision*loss)/precision,'Wmean':int(precision*Wmean)/precision,'Wstd':int(precision*Wstd)/precision,'bmean':int(precision*bmean)/precision,'bstd':int(precision*bstd)/precision}
	def get_losses(self):
		return self.losses
	def loss_gradient(self,model,loss_func,y):
		grad = [None]*len(model.synapses)
		last_grad = loss_grad(loss_func,model.layers[-1],y)
		for i in range(len(model.synapses)-1,-1,-1):
			grad[i] = model.synapses[i].get_grad(model.layers[i],last_grad)
			last_grad = grad[i]['next_grad']
		return grad
	def update(self,model,wgrad,bgrad,l_rate):
		if self.update_type == 'sgd':
			for i in range(0,len(model.synapses)):
				if self.reg_type == 'L2':
					model.synapses[i].W -= l_rate*(wgrad[i]+self.reg_coeff*2*model.synapses[i].W)
					model.synapses[i].b = model.synapses[i].b-l_rate*bgrad[i]
				if self.reg_type == 'L1':
					reg = model.synapses[i].W/np.abs(model.synapses[i].W)
					reg[np.isnan(reg)]=1
					model.synapses[i].W -= l_rate*(wgrad[i]+self.reg_coeff*reg)
					model.synapses[i].b = model.synapses[i].b-l_rate*bgrad[i]
		return model
	def train(self,X,y,model,num_epochs,batch_size,l_rate,loss_func):
		self.losses = []
		for i in range(0,num_epochs):
			try:  # used try so that if user pressed other than the given key error will not be shown
				if keyboard.is_pressed('q'):  # if key 'q' is pressed 
					print('You Pressed A Key!')
					break  # finishing the loop
			except:
				break
			for j in range(0,int(len(X)/batch_size)):
				try:  # used try so that if user pressed other than the given key error will not be shown
					if keyboard.is_pressed('q'):  # if key 'q' is pressed 
						print('You Pressed A Key!')
						break  # finishing the loop
				except:
					break
				wgrad = [0]*len(model.synapses)
				bgrad = [0]*len(model.synapses)
				interval = np.arange(j*batch_size,(j+1)*batch_size)
				if j%20==0:
					print('epoch ',i,', ',interval[0],':',interval[-1])
				self.loss = 0
				for k in range(0,len(interval)):
					model.feed_fwd(X[interval][k])
					loss = loss_func(model.layers[-1],int(y[interval][k]))
					grad = self.loss_gradient(model,loss_func,int(y[interval][k]))
					for n in range(0,len(model.synapses)):
						wgrad[n]+=grad[n]['wgrad']
						bgrad[n]+=grad[n]['bgrad']
					self.loss += loss
				
				for n in range(0,len(model.synapses)):
					wgrad[n] = wgrad[n]/len(interval)
					bgrad[n] = bgrad[n]/len(interval)
				model = self.update(model,wgrad,bgrad,l_rate)	
				if self.reg_type == 'L1':
					reg = np.sum(np.abs(model.get_W()))
				if self.reg_type == 'L2':
					reg = np.sum(np.square(model.get_W()))
				self.loss = self.loss/len(interval)+self.reg_coeff*reg
				self.losses.append(self.loss)
				if j%20==0:
					print('stats: ',self.statistics(model))
			#interval = np.arange(interval[-1]+1,len(X))
			#if len(interval)>0:
			#	print('rest of batch(not calculated):')
			#	if i==0:
			#		print(interval[0],':',interval[-1])
				
		
class Model():
	def __init__(self,input_size,output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.current_output_size = input_size
		self.synapses = []
		self.layers = [np.zeros(input_size)]
	def add_filters(self,n,size,padding,stride,name,activation,init_W):
		self.synapses.append(Filter(n,size,padding,stride,name,self.current_output_size,activation,init_W))
		self.layers.append(0)
		self.current_output_size = self.synapses[-1].output_size
	def add_FC(self,size,name,activation,init_W):
		self.synapses.append(FC(size,name,self.current_output_size,activation,init_W))
		self.layers.append(0)
		self.current_output_size = size
	def feed_fwd(self,input):
		self.layers[0]=input
		for i in range(0,len(self.synapses)):
			self.layers[i+1]=self.synapses[i].feed_fwd(input)
			input = self.layers[i+1]
	def get_W(self):
		all_W = np.asarray(self.synapses[0].W).flatten()
		for i in range(1,len(self.synapses)):
			all_W = np.concatenate((all_W,np.asarray(self.synapses[i].W).flatten()),axis=None)
		return all_W
	def get_b(self):
		all_b = np.asarray(self.synapses[0].b).flatten()
		for i in range(1,len(self.synapses)):
			all_b = np.concatenate((all_b,np.asarray(self.synapses[i].b).flatten()),axis=None)
		return all_b
	
		
		
class Filter():
	def __init__(self,n,size,padding,stride,name,input_size,activation,init_W,padding_mode = 'constant',padding_cons=0):
		self.number = n
		self.size = size
		self.padding = padding
		self.padding_mode = padding_mode
		self.padding_cons = padding_cons
		self.stride = stride
		self.name = name
		self.activation = activation
		self.W = []
		self.b = []
		for i in range(0,n):
			self.W.append(init_W*np.random.randn(size[0],size[1],input_size[2]))
			self.b.append(0)
		self.W = np.asarray(self.W)
		self.b = np.asarray(self.b)
		out_sizeX = (input_size[0]+2*padding-size[0])/stride+1
		out_sizeY = (input_size[1]+2*padding-size[1])/stride+1
		if (int(out_sizeX)!=out_sizeX):
			print("the stride for filter ",name," does not fit X input size")
		if (int(out_sizeY)!=out_sizeY):
			print("the stride for filter ",name," does not fit Y input size")
		out_sizeX = int(out_sizeX)
		out_sizeY = int(out_sizeY)
		self.output_size = [out_sizeX,out_sizeY,n]
	def feed_fwd(self,input):
		layer = np.zeros(self.output_size)
		self.preact_output = []
		for k in range(0,self.number):
			conv = convolve(input,self.W[k], mode='full')[:,:,input.shape[2]-1]+self.b[k]
			unpadding = [[self.size[0]-self.padding-1,-(self.size[0]-self.padding-1)],[self.size[1]-self.padding-1,-(self.size[1]-self.padding-1)]]
			conv = conv[unpadding[0][0]:(unpadding[0][1] or None):self.stride,unpadding[1][0]:(unpadding[1][1] or None):self.stride]
			self.preact_output.append(conv)
			layer[:,:,k] = self.activation(conv)
		return layer
	def get_grad(self,input,last_grad):
		bgrad = 0*self.b
		wgrad = 0*self.W
		xgrad = 0.0*input
		for k in range(0,self.number):
			conv = self.preact_output[k]
			actigrad = activation_grad(self.activation,conv)
			actigrad_spaces = np.zeros((self.stride*actigrad.shape[0]-self.stride+1,self.stride*actigrad.shape[1]-self.stride+1))
			actigrad_spaces[::self.stride,::self.stride] = last_grad[:,:,k]*actigrad
			input_pad = np.pad(input,((self.padding,self.padding),(self.padding,self.padding),(0,0)),mode = self.padding_mode,
				constant_values = ((self.padding_cons,self.padding_cons),(self.padding_cons,self.padding_cons),(None,None)))
			bgrad[k] = np.sum(last_grad[:,:,k]*actigrad)
			actigrad_spaces_pad = np.pad(actigrad_spaces,((self.size[0]-self.padding-1,self.size[0]-self.padding-1),(self.size[1]-self.padding-1,self.size[1]-self.padding-1)),mode = self.padding_mode,
				constant_values = ((self.padding_cons,self.padding_cons),(self.padding_cons,self.padding_cons)))
			for i in range(0,input.shape[2]):
				wgrad[k][:,:,i] = convolve(input_pad[:,:,i],actigrad_spaces, mode='valid')
				xgrad[:,:,i] += convolve(self.W[k][:,:,i],actigrad_spaces_pad, mode='valid')
			
			
		return {'bgrad':np.asarray(bgrad),'wgrad':np.asarray(wgrad),'next_grad':xgrad}

class FC():
	def __init__(self,size,name,input_size,activation,init_W):
		self.size = size
		self.name = name
		self.activation = activation
		print(input_size)
		if isinstance(input_size, (list, tuple, np.ndarray)):
			streched_input_size = input_size[0]*input_size[1]*input_size[2]
		else:
			streched_input_size = input_size
		print(streched_input_size)
		self.W = init_W*np.random.randn(streched_input_size,size)
		self.b = np.zeros(size)
	def feed_fwd(self,input):
		self.preact_output = np.dot(input.flatten(),self.W)+self.b
		return self.activation(self.preact_output)
	def get_grad(self,input,last_grad):
		actigrad = activation_grad(self.activation,self.preact_output)
		bgrad = last_grad*actigrad
		wgrad = np.outer(input.flatten(),last_grad*actigrad)
		xgrad = np.dot(last_grad*actigrad,self.W.T)
		xgrad = xgrad.reshape(input.shape)
		return {'bgrad':bgrad,'wgrad':wgrad,'next_grad':xgrad}


#model = Model([10,10,3],[1,10])
#model.add_filters(3,[5,5],2,3,'Filter 1',preLU,0.0001)
#model.feed_fwd(np.zeros((10,10,3)))