#!/usr/bin/env python
# coding: utf-8

# In[80]:

#Import needed package
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:

#initialize the lists
apl_price=[93.95,112.15,104.05,144.85,169.49]
ms_price=[39.95,50.57,57.05,69.85,94.49]
year=[2014,2015,2016,2017,2018]


# In[71]:

#Plotting data
plt.plot(year,apl_price,':k')
plt.plot(year,ms_price,'og')
plt.xlabel('year')
plt.ylabel('stock price')
plt.show()


# In[84]:

#adding graphs to figure
fig_1=plt.figure(1,figsize=(10,4.8))
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


# In[111]:


import numpy as np
ar=np.array([1,2,3])
ar1=np.array([1,2,3])
ar=np.hstack([ar,ar1])
ar


# In[ ]:





# In[ ]:




