import matplotlib.pyplot as plt
import numpy as np
import pickle
from plot.font import get_font

font = get_font()


pred = np.load('../pred.npy')
true = np.load('../true.npy')

lw = 1

fig, ax = plt.subplots(figsize=(7, 4),dpi=300)
a_, = ax.plot(true, linewidth=lw,label='True')   #, color='black'
b_, = ax.plot(pred, linewidth=lw, label='TPGNN')   #, color='#D9B9A4'
# c_, = ax.plot(Pred_LSTM, linewidth=lw, label='LSTM', color='#D9B9A4')
# d_, = ax.plot(Pred_GCNLSTM, linewidth=lw, label='GCNLSTM', color='#6caa89')
# e_, = ax.plot(Pred_GATLSTM, linewidth=lw, label='GATLSTM',color='#e76254')
ax.legend(handles = [a_,b_], loc='upper left', prop=font, fontsize=28, markerscale=5)
plt.savefig('WOKmatrics.png', bbox_inches='tight')
plt.show()

