
# coding: utf-8

# ## SVD Image Decomposition / Compression

# Implementación del algoritmo SVD para la compresión de una imagen, a medida que se reducen las dimensiones se va perdiendo información y se puede ver como afecta a la calidad de la imagen.

# In[1]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image


# In[34]:


# obtenemos la imagen para poder trabajarla
img = Image.open(r'C:\Users\sbotalla\Desktop\perseocasiopea.png')
imggray = img.convert('LA')
plt.figure(figsize=(9, 6))
plt.imshow(imggray);


# In[35]:


# llevamos la imagen a una representacion matricial con numpy 
# para poder operar
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)

# realizamos el plot a partir de la matriz (no la imagen)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap='gray');


# # SVD

# In[36]:


# realizamos la descomposicion en valores singulares
u, s, v = np.linalg.svd(imgmat)


# In[37]:


imgmat


# In[48]:


# verificamos los valores singulares
s


# # Analisis de los valores singulares

# In[40]:


# realizamos un plot de los valores singulaes# realiz 
# es interesante ir reduciendo el rango para poder 
# verificar algun posible 'codo' para poder considerar 
# la dimensionalidad intrinseca de los datos
#plt.plot(s)
plt.plot(s[:20])


# # Calculo de Energia

# In[41]:


# calculo de energia
singular_values = s
energy_total = sum(singular_values**2)
sv_pow = singular_values**2
energy_percentage = ((sv_pow / energy_total) * 100)
print(energy_percentage)


# In[42]:


# realizamos entonces el calculo de cuanta energia se va
# acumulando con cada uno de los 
acumulated_energy = 0
for i in range(len(energy_percentage)):
    acumulated_energy = acumulated_energy + energy_percentage[i]
    print ('Number of Singular Values ' + str(i+1) + ': ' + str(singular_values[i]) + ' ' + str(round(acumulated_energy,2)) + '%')


# ## Reconstruyendo la imagen a partir de los datos reducidos
# 

# In[43]:


reconstimg = np.matrix(u[:, :1]) * np.diag(s[:1]) * np.matrix(v[:1, :])
plt.figure(figsize=(9,6))
plt.imshow(reconstimg, cmap='gray');


# In[50]:


for i in range(1, 364, 10):
    reconstimg = np.matrix(u[:, :i]) * np.diag(s[:i]) * np.matrix(v[:i, :])
    plt.figure(figsize=(9,6))
    plt.imshow(reconstimg, cmap='gray')
    title = "k = %s" % i
    plt.title(title)
    plt.show()

