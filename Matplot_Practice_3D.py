#!/usr/bin/env python
# coding: utf-8

# In[80]:


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


apl_price=[100.95,113.15,103.05,142.85,160.49]
ms_price=[39.95,50.57,57.05,69.85,94.49]
year=[2014,2015,2016,2017,2018]


# In[71]:


plt.plot(year,apl_price,':k')
plt.plot(year,ms_price,'og')
plt.xlabel('year')
plt.ylabel('stock price')
plt.show()


# In[84]:


fig_1=plt.figure(1,figsize=(20,4.8))
chart_1=fig_1.add_subplot(121)
chart_2=fig_1.add_subplot(122)

chart_1.plot(year,apl_price)
chart_1.xaxis.set_major_locator(MaxNLocator(integer=True))
chart_2.plot(year,ms_price)

plt.show()


# In[102]:



lambda_test=lambda a,b :a+b
lambda_test(2,3)
      

    


# In[104]:


empty_dcit=dict()
empty_dcit['Name']='sadaf'

empty_dcit


# In[3]:


import numpy as np
ar=np.array([1,2,3])
ar1=np.array([1,2,3])
ar=np.hstack([ar,ar1])
matric=np.mat([[1,0],[1,2],[3,0]])
matric


# In[55]:


from scipy import sparse
import numpy as np
matric=np.array([[1,0],[1,2],[3,0]])
matrix_sparse=sparse.csr_matrix(matric)
matric.shape[0]


# In[56]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

xvals=np.linspace(-4,4,10)
yvals=np.linspace(-4,4,7)
xygrid=np.column_stack([[x,y] for x in xvals for y in yvals])
len(str(2))


# In[28]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_radii=8
n_angles=36

radii=np.linspace(0.25,1.0,n_radii)
angles=np.linspace(0,2*np.pi,n_angles,endpoint=False)
angles=np.repeat(angles[...,np.newaxis],n_radii,axis=1)
x=np.append(0,(radii*np.cos(angles)).flatten())
y=np.append(0,(radii*np.sin(angles)).flatten())
z=np.sin(-x*y)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_trisurf(x,y,z,linewidth=0.2,antialiased=True)
plt.show()


# In[29]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.show()


# In[30]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,2,3,4])
plt.show()


# In[31]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,2,3,4],'ro')
plt.show()


# In[45]:


import matplotlib.pyplot as plt
import numpy as np
t=np.arange(0.,5.,0.2)
lin1,line2,line3=plt.plot(t,t,'r--',t,np.cos(2*np.pi*t),'bs',t,t**3,'g>',linewidth=5)

plt.show()


# In[49]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
chart=fig.add_subplot(111,projection='3d')
chart.plot([2,4,8],[1,2,3],[4,5,6])


# In[6]:


from sympy import *

diff(cos(x),x)


# In[9]:


from sympy import *

diff(x**3,x,3)


# In[18]:


import numpy as np
def cost(b):
    return (b-4)**2

def sope(b):
    return 2*(b-4)

b=5
for i in range(20):
    b=.1*sope(b)
    print(b)
    


# In[21]:


import numpy as np
import time

n_hidden=10
n_in=10
n_output=10
n_sample=300

learning_rate=0.01
momento=0.9

np.random.seed(0)
def sigmoid(z):
    return 1/1+np.exp(-z)


# In[ ]:





# In[ ]:




