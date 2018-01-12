from PIL import Image
import scipy.ndimage
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,f1_score,precision_recall_fscore_support
import matplotlib.pyplot as plt

#ground_truth = Image.open("test_500_mask.tif")
#G_t = np.array(ground_truth) 
#X_t = Image.open("Mask_IntensityStructNoMorph.jpg")
#X = np.array(X_t)

Xm = scipy.misc.imread('Result/Mask_IntensityStructNoMorph.tif', flatten=True).astype(np.uint8)
Xm_nohole = scipy.ndimage.binary_fill_holes(Xm, structure=np.ones((6,6))).astype(int)
Xm_opened = scipy.ndimage.grey_opening(Xm_nohole, size=(4,4))
Xm_closed = scipy.ndimage.grey_closing(Xm_opened, size=(4,4))
scipy.misc.imsave('Result/Mask_IntensityStructMorph.tif', Xm_closed)
plt.imshow(Xm_closed)
plt.show()

#Gr = np.resize(G_t, (G_t.shape[0]*G_t.shape[1],1))
#Xr = np.resize(X, (X.shape[0]*X.shape[1],1))
#Xmr = np.resize(Xm, (Xm.shape[0]*Xm.shape[1],1))

#test_metrics1 = precision_recall_fscore_support(Gr, Xr)
#print 'Precision = ' , "%.4f" %test_metrics1[0][0]
#print 'Recall = ' , "%.4f" %test_metrics1[1][0]
#print 'F1 score = ' , "%.4f" %test_metrics1[2][0]
#
#test_metrics2 = precision_recall_fscore_support(Gr, Xmr)
#print 'Precision = ' , "%.4f" %test_metrics2[0][0]
#print 'Recall = ' , "%.4f" %test_metrics2[1][0]
#print 'F1 score = ' , "%.4f" %test_metrics2[2][0]
