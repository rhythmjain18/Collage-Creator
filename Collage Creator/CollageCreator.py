#!/usr/bin/env python
# coding: utf-8

# # Assingment1
# 

# ### Task - 
# #### Extract a video’s frames as input and create a collage of frames of that video.Chose six frames from the video randomly and join them into a collage. Sort the frames based on their color/edge information. An example for image placement in collage: image with less variation in color can be in the centre and the ones on the boundary of the collage have more color variation.
# 
# ### Marking –
# #### CollageCreate(AddressofFolder)  function in CollageCreator.py returns and displays a collage image. The input to the function is a relative address of folder containing images from a video. Marks – 8 marks
# #### In the word document Report.doc discuss the method in CollageCreate(). How are the images joined together? Word limit - 100 words – 2 marks Hint: histograms, color and edge information can be used. For extracting frames from a video, you can use ffmpeg library.
# 

# In[26]:


from skimage import io,filters,feature,img_as_ubyte,img_as_float
from skimage.color import rgb2gray
from matplotlib import *
import numpy as np
from skimage.transform import resize
from skimage.viewer import ImageViewer as IV
import glob
import random

def CollageCreate(path):
    #Taking all images from the folder
    TotalList=glob.glob(f"{path}/*.*")
    
    #Picking 6 Random Images
    UpdatedList = random.sample(TotalList, 6)
    ImagesList=[]
    
    #Reading Those Random Images
    for i in UpdatedList:
        ImagesList.append(io.imread(i,as_gray=False))
        
        
    for i in range(6):
        ImagesList[i]=resize(ImagesList[i],(480,640));
    
    #Sort According to Color Variations
    ImagesList.sort(key= lambda i:var_color_edge(i))
    
    
    #For Viewing Output 
    
    Output_Img=merge(ImagesList,100,80)
    
    viewer=IV(Output_Img)
    viewer.show()
    
    #For Saving Image to the specified Location     
    io.imsave(f"{path}/Output_Img.jpg",img_as_ubyte(Output_Img))
    
    #Returning Image     
    return Output_Img

    
#Functions For Merging   
  
def merge(ImagesList,xblock,yblock):
    Row1=merge_h(ImagesList[0],ImagesList[1],xblock)
    Row2=merge_h(ImagesList[2],ImagesList[3],xblock)
    Row3=merge_h(ImagesList[4],ImagesList[5],xblock)
    Col=merge_v(merge_v(Row1,Row2,yblock),Row3,yblock)
    return Col
    

def merge_h(image1,image2,block):

    width1=np.shape(image1)[1]
    height=np.shape(image1)[0]
    temp1=image1[:,width1-block:,:]
    temp2=image2[:,0:block,:]
    temp3=np.zeros((height,block,3))
    
    mid=block//2
    add=block//2
    a=0
    # Merging
    while add>=0:
        temp3[:,mid-add:mid,:]=temp1[:,mid-add:mid,:]*(1-a) + temp2[:,mid-add:mid,:]*a
        temp3[:,mid:mid+add,:]=temp1[:,mid:mid+add,:]*a + temp2[:,mid:mid+add,:]*(1-a)
        add-=1
        a+=(0.5/mid)
        

    final=np.hstack([image1[:,:width1-block,:],temp3,image2[:,block:,:]])
    return final
   
def merge_v(image1,image2,block):
    
    height1=np.shape(image1)[0]
    width=np.shape(image1)[1]
    temp1=image1[height1-block:,:,:]
    temp2=image2[:block,:,:]
    temp3=np.zeros((block,width,3))
    
    mid=block//2
    add=block//2
    a=0
    # Merging
    while add>=0:
        temp3[mid-add:mid,:,:]=temp1[mid-add:mid,:,:]*(1-a) + temp2[mid-add:mid,:,:]*a
        temp3[mid:mid+add,:,:]=temp1[mid:mid+add,:,:]*a + temp2[mid:mid+add,:,:]*(1-a)
        add-=1
        a+=(0.5/mid)
        

    final=np.vstack([image1[:height1-block,:,:],temp3,image2[block:,:,:]])
    return final

#Function For Sort  
def var_color_edge(image):
    
    #Color Intensity variance
    height,width,depth=np.shape(image)
    color=[]
    for c in range (depth):
        temp=0;
        for i in range (height):
            for j in range(width):
                temp+=image[i][j][c]
        color.append(temp/(height*width)*100)



    color=(color[0] + color[1] + color[2])/3

    
    #Edge Variance     
    temp=rgb2gray(image)
    temp=feature.canny(temp,sigma=2)
    
    from skimage.filters import threshold_otsu
    
    temp=img_as_ubyte(temp)

    thresh=threshold_otsu(temp)
    temp= temp<= thresh
    
    black=np.sum(temp==0)
    white=np.sum(temp==1)
    
    edge=(black/(black+white))*100
    
    
    #Sort by 20% color var and 80%Edge Variance
    return 0.8*edge + 0.2*color

    


CollageCreate("images")


# In[ ]:




