import plotly
import plotly.graph_objs as go
import model_storage as ms
from plotly import subplots
import numpy as np
model_name = 'Filter 10 5x5 2 3 preLU 60000x10_again'
model = ms.get(model_name)
n_filters = len(model.synapses[0].W)
#create the figure
sq = int(np.sqrt(n_filters))
fig = subplots.make_subplots(rows=sq+1, cols=sq,subplot_titles = np.arange(0,n_filters).astype(str))
print(model.synapses[0].W[0].reshape((5,5)))
for i in range(0,n_filters):
	fig.append_trace(go.Heatmap(
		z=model.synapses[0].W[i].reshape((5,5))
	),int((i-i%sq)/sq)+1,i%sq+1)
fig.layout.update(
				go.Layout(
					autosize = False,
					width = 750,
					height = 1000
				)
			)
plotly.offline.plot(fig,filename = 'C:/Users/Diana/Documents/Models/Reports/'+model_name+'.html',auto_open=True)