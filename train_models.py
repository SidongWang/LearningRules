from CNN import *
import NN_database as nd
import model_storage as ms
getter = nd.Datagetter()
X_train,y_train,mndata = getter.get_MNIST('train',printfirst=0)
X_train = X_train.reshape((60000,28,28,1))
#print (y_train)
#print(X_train[1])
model = Model(X_train[0].shape,[1,10])


#model.add_filters(10,[5,5],2,3,'Filter 1',preLU,0.001)
#model.add_filters(3,[2,2],0,2,'Filter 2',preLU,0.0001)
model.add_FC(15000,'FC 0',preLU,0.001)
#model.add_FC(30,'FC 1',preLU,0.01)
#model.add_FC(30,'FC 2',preLU,0.01)
#model.add_FC(30,'FC 3',preLU,0.01)
model.add_FC(10,'final layer',preLU,0.001)

trainer = Trainer(1e-3,'L2','sgd')
#model_name = 'FC 30 30 30 30 preLU 60000x100_again'
#model = ms.get(model_name)
#print(model.synapses[2].W)
size_sample = 60000
num_epochs = 100
trainer.train(X_train[:size_sample],y_train[:size_sample],model,num_epochs = num_epochs,batch_size = 50,l_rate = 0.001,loss_func = softmax_loss)
stats = trainer.statistics(model)
print(stats)
import matplotlib.pyplot as plt
losses = trainer.get_losses()
plt.plot(np.arange(0,len(losses))/len(losses)*num_epochs,losses)
plt.ylabel('loss')
plt.axis([0, num_epochs, 0, 2.5])
plt.show()
#print(model.synapses[2].W)
ms.save(model,type(model.synapses[0]).__name__+' 15000 preLU '+str(size_sample)+'x'+str(num_epochs))
#ms.save(model,'FC 30 30 30 30 preLU 60000x200_again2')
