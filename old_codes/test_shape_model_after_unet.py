# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 00:52:28 2023

@author: Eedrobup
"""

import config
import os

import numpy as np
from scipy import misc,ndimage
import matplotlib.pyplot as plt
from skimage import io,color
from skimage.util import crop


import torch
import UNet3
from torchvision import transforms
from scipy import ndimage
from skimage.transform import downscale_local_mean
from skimage import transform
import torchvision.transforms as T
from scipy.optimize import minimize

working_directory = config.PersonalConfig.call_wd()
os.chdir(working_directory)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_one = torch.load('./unet_first_layer').to(device)
model_two = torch.load('./unet_second_layer').to(device)
mean_shape = np.genfromtxt("mean_shape.csv",delimiter=',')
#%%



def convert_image_to_tensor(img):
    t_img=transforms.ToTensor()(img)
    t=torch.tensor((),dtype=torch.float32)
    t_in=t.new_ones((1,t_img.shape[0],t_img.shape[1],t_img.shape[2]))
    t_in[0]=t_img
    return t_in

def get_8_points(model,image,device=device):
    # Apply model to image. Display image and peak response point
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    points = [np.unravel_index(np.argmax(result, axis=None), result.shape) for result in y.detach().numpy()[0]]
    pts=np.zeros(16,)
    for i,co in enumerate(points):
        pts[2*i] = co[1]
        pts[2*i+1] = co[0]
    return pts

def find_frog_spine_line(points):
    spine_points = [points[0],
                    tuple(np.average([points[1],points[7]],axis=0)),
                    tuple(np.average([points[0],tuple(np.average([points[1],points[7]],axis=0))],axis=0)),
                    tuple(np.average([np.average([points[0],points[1]],axis=0),np.average([points[0],points[7]],axis=0)],axis=0)),
                    tuple(np.average([points[2],points[6]],axis=0)),
                    tuple(np.average([points[0],points[4]],axis=0)),
                    #tuple(np.average([points[4],tuple(np.average([points[3],points[5]],axis=0))],axis=0)),
                    tuple(np.average([np.average([points[3],points[4]],axis=0),np.average([points[5],points[4]],axis=0)],axis=0)),
                    tuple(np.average([points[3],points[5]],axis=0)),
                    points[4],
                    ]
    n = len(spine_points)
    #y,X = np.transpose(spine_points)
    
    #old code gradient diminish when slope approach verticle
    """xbar = sum(X)/len(X)
    ybar = sum(y)/len(y)
    numer = sum([xi*yi for xi,yi in zip(X, y)]) - n * xbar * ybar
    denumx = sum([xi**2 for xi in X]) - n * xbar**2
    denumy = sum([yi**2 for yi in y]) - n * ybar**2
    bx = numer / denumx
    by = -1*numer/denumy
    ax = ybar - bx * xbar
    ay = xbar - by * ybar"""
    
    #Use PCA to determine the best projection angle
    # === Perform PCA ===
    # First subtract mean from each row
    mean = np.mean(spine_points,axis=0)
    D = spine_points-mean
    
    # Create covariance matrix
    S=D.T @ D/n

    # Compute the eigenvectors and eigenvalues (arbitrary order)
    evals,EVecs = np.linalg.eig(S)

    # Sort by the eigenvalues (largest first)
    idx = np.flip(np.argsort(evals),0)
    evals = evals[idx]
    EVecs = EVecs[:,idx]
    
    dy, dx = EVecs[:,0]
    """
    plt.scatter(np.transpose(spine_points)[1],np.transpose(spine_points)[0],color='blue')
    plt.scatter(np.transpose(points)[1],np.transpose(points)[0],color='green')
    plt.scatter(mean[1],mean[0],color='red')
    near_pt = mean+(np.reshape(np.sqrt(evals[0])*np.array([1,-1]),(2,1))@np.reshape(np.array(EVecs[:,0]),(1,2)))
    plt.scatter(near_pt.T[1],near_pt.T[0],color='orange')
    plt.imshow(transform.rescale(X[8], 1/4, channel_axis=2))
    plt.show()"""
    
    theta = np.rad2deg(np.arctan2(dx,dy))
    points, spine_points = np.array(points), np.array(spine_points)
    vecY, vecX = np.average(spine_points[0:5],axis=0)-np.average(spine_points[6:],axis=0)
    pca_vector = np.array([dy,dx])/(np.sqrt(dy**2+dx**2))
    points_vector = np.array([vecY,vecX])/(np.sqrt(vecY**2+vecX**2))
    direction = np.dot(pca_vector,points_vector)
    print(direction, theta)
    
    return 180-theta if direction>0 else -theta
    """
    if direction>0:
        return 180-theta
    else:
        return -theta
    """
    #return dy, dx, spine_points 

"""def theta_from_slope(dy,dx,points,spine_points):
    theta = np.rad2deg(np.arctan2(-dy,dx)) #-dy since the image invert y axis
    points, spine_points = np.array(points), np.array(spine_points)
    vecY, vecX = np.average(spine_points[0:5],axis=0)-np.average(spine_points[6:],axis=0)
    direction = np.dot([dy,dx],[vecY,vecX])
    print(dy,dx)
    print(direction, theta)
    if direction>0:
        return theta
    else:
        return 180+theta"""
    

def get_upright_image(model,spare_model,image,device=device):
    # Apply model to image and display the result as heat map
    y=model(convert_image_to_tensor(image).to(device)).to('cpu')
    points = [np.unravel_index(np.argmax(result, axis=None), result.shape) for result in y.detach().numpy()[0]]
    theta=find_frog_spine_line(points)
    #dy,dx,spine = find_frog_spine_line(points)
    #theta = theta_from_slope(dy,dx, points,spine)
    print(points,theta)
    return transform.rotate(image, theta,center = np.average(points,axis=0),resize=True)


def new_dim(image):
    x,y,z = image.shape
    #factor_two = np.array([256,512,768,1024,2048,4096])
    #x_dim = np.argmin(abs(x-factor_two))
    #y_dim = np.argmin(abs(y-factor_two))
    return (x-x%4,y-y%4,z)#(factor_two[x_dim],factor_two[y_dim],z)

def plot_points(plt, points, style):
    pts=points.reshape(-1,2)
    plt.plot(pts[:,0],pts[:,1],style)

def affine_transform_cost(params, source_points, target_points):
    a, b, c, d, e, f = params
    x_source, y_source = source_points[:, 0], source_points[:, 1]
    x_target, y_target = target_points[:, 0], target_points[:, 1]

    x_transformed = a * x_source - b * y_source + c
    y_transformed = d * x_source + e * y_source + f

    error_x = x_target - x_transformed
    error_y = y_target - y_transformed

    error = (np.sum(abs(error_x)) + np.sum(abs(error_y)))/2

    return error

def affine_this(params, source_points):
    a, b, c, d, e, f = params
    x_source, y_source = source_points[:, 0], source_points[:, 1]

    x_transformed = a * x_source - b * y_source + c
    y_transformed = d * x_source + e * y_source + f

    print(x_transformed)
    print(y_transformed)

    return np.array([x_transformed,y_transformed])


def optimize_affine_transform(source_points, target_points):
    initial_params = [1, 0, 0, 0, 1, 0]  # Initial guess for the parameters
    result = minimize(affine_transform_cost, initial_params, args=(source_points, target_points), method='BFGS')
    #print(result)
    return result.x

def shape_matching(mean_shape, points):
    """
    mean_shape in (8,2) 
    points in 
    """
    optimal_params = optimize_affine_transform(mean_shape, points)
    pred_points = affine_this(optimal_params,mean_shape) # [[x1,y1],[x2,y2],...,[xn,yn]]
    pred_points = np.round(pred_points,0)
    points = points.T
    dist = ((pred_points[0,:]-points[0,:])**2+(pred_points[1,:]-points[1,:])**2)**(1/2)
    dist_med = np.median(dist)
    dist_sd = np.std(dist)
    dist_z = (dist - dist_med)/dist_sd
    threshold = 1 # how many sd from median is acceptable for unet coordinate
    dist_booleen = [True if abs(x)<threshold else False for x in dist_z]
    final_points = np.zeros((8,2))
    for i,b in enumerate(dist_booleen):
        if b:
            final_points[i] = points.T[i]
        else:
            final_points[i] = pred_points.T[i]
    return final_points
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle in degree but need to transform to radian later within function.
    """
    ox, oy = origin
    px, py = point
    angle = np.deg2rad(angle)

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
#%% loading the pictures
path='./Froggy ID pictures/B1/B1.08/'
X=[]
name = os.listdir(path)
if 'patch' in name:
    name.remove('patch')
for i in name:
    X += [io.imread(path+i)]
#face_im = io.imread(path)
del i, name



#%% finding initial points part



new_X = get_upright_image(model_one,model_two,transform.rescale(X[0], 1/4, channel_axis=2),device)
plt.imshow(new_X)
plt.show()
new_X = transform.resize(new_X,new_dim(new_X))
#print(new_X.shape)
#points = get_8_points(model_one, transform.rescale(X[7], 1/4, channel_axis=2))
points = get_8_points(model_two, new_X)

#%% shape model part
from scipy.optimize import minimize





optimal_params = optimize_affine_transform(mean_shape.reshape(-1,2), points.reshape(-1,2))
pred_points = affine_this(optimal_params,mean_shape.reshape(-1, 2)) # [[x1,x2...],[y1,y2,...]]
pred_points = np.round(pred_points,0)
    
#plot_points(plt,mean_shape,'o')
plot_points(plt,pred_points.T,'+')
plot_points(plt,points,'x')
plt.imshow(new_X)
plt.show()


#%% select the best point from the two sets

#find the euclidean distance of model_points and unet_points
points = points.reshape(-1,2).T
dist = ((pred_points[0,:]-points[0,:])**2+(pred_points[1,:]-points[1,:])**2)**(1/2)
dist_med = np.median(dist)
dist_sd = np.std(dist)
dist_z = (dist - dist_med)/dist_sd
threshold = 1 # how many sd from median is acceptable for unet coordinate
dist_booleen = [True if abs(x)<threshold else False for x in dist_z]
final_points = np.zeros((8,2))
for i,b in enumerate(dist_booleen):
    if b:
        final_points[i] = points.T[i]
    else:
        final_points[i] = pred_points.T[i]

plot_points(plt,final_points,'x')
plt.imshow(new_X)
plt.show()

#%% Cropping out the out of picture

min_x = min(final_points[:,0])
max_x = max(final_points[:,0])
min_y = min(final_points[:,1])
max_y = max(final_points[:,1])

final_points[:,0] -= min_x
final_points[:,1] -= min_y

new_X = new_X[int(min_y):int(max_y),int(min_x):int(max_x),:]
#%%
plot_points(plt,final_points,'x')
plt.plot(np.average(final_points[:,0]),np.average(final_points[:,1]),'o')
plt.imshow(new_X)
plt.show()

#%%

plot_points(plt,points,'x')
plt.plot(np.average(points.reshape(-1,2).T[0]),np.average(points.reshape(-1,2).T[1]),'o')
plt.imshow(transform.rescale(X[7], 1/4, channel_axis=2))
plt.show()

#%%
n=7
plot_points(plt,get_8_points(model_one, transform.rescale(X[n],1/4,channel_axis=2)),'x')
plt.imshow(transform.rescale(X[n],1/4,channel_axis=2))
plt.show()
#%%
for n in range(20):
    tmp = get_upright_image(model_one, model_two, transform.rescale(X[n],1/4,channel_axis=2),device)
    tmp = transform.resize(tmp, new_dim(tmp))
    points = get_8_points(model_two,tmp)
    #plot_points(plt,points,'x')
    #plt.imshow(tmp)
    #plt.show()
    optimal_params = optimize_affine_transform(mean_shape.reshape(-1,2), points.reshape(-1,2))
    pred_points = affine_this(optimal_params,mean_shape.reshape(-1, 2)) # [[x1,x2...],[y1,y2,...]]
    pred_points = np.round(pred_points,0)
    points = points.reshape(-1,2).T
    dist = ((pred_points[0,:]-points[0,:])**2+(pred_points[1,:]-points[1,:])**2)**(1/2)
    dist_med = np.median(dist)
    dist_sd = np.std(dist)
    dist_z = (dist - dist_med)/dist_sd
    threshold = 1 # how many sd from median is acceptable for unet coordinate
    dist_booleen = [True if abs(x)<threshold else False for x in dist_z]
    final_points = np.zeros((8,2))
    for i,b in enumerate(dist_booleen):
        if b:
            final_points[i] = points.T[i]
        else:
            final_points[i] = pred_points.T[i]
    
    plot_points(plt,final_points,'x')
    plt.imshow(tmp)
    plt.show()
#%% Final program
import math

path='./Froggy ID pictures/B1/B1.08/'
X=[]
name = os.listdir(path)
if 'patch' in name:
    name.remove('patch')
for i in name:
    X += [io.imread(path+i)]
#face_im = io.imread(path)
del i, name


for n in range(20):
    print('doing image no',n)
    new_X = get_upright_image(model_one,model_two,transform.rescale(X[n], 1/4, channel_axis=2),device)
    new_X = transform.resize(new_X,new_dim(new_X))
    points = get_8_points(model_two, new_X)
    final_points = shape_matching(mean_shape.reshape(-1,2), points.reshape(-1,2))
    #final_points = shape_matching(mean_shape.reshape(-1,2), final_points)
    
    #plot_points(plt,final_points,'x')
    #plt.imshow(new_X)
    #plt.show()
    margin = 70
    min_x = max(min(final_points[:,0])-margin,0)
    max_x = min(max(final_points[:,0])+margin,new_X.shape[1])
    min_y = max(min(final_points[:,1])-margin,0)
    max_y = min(max(final_points[:,1])+margin,new_X.shape[0])
    final_points[:,0] -= min_x
    final_points[:,1] -= min_y
    new_X = new_X[int(min_y):int(max_y),int(min_x):int(max_x),:]
    
    
    new_X = transform.resize(new_X,new_dim(new_X))
    new_X = get_upright_image(model_one,model_two,new_X,device)
    new_X = transform.resize(new_X,new_dim(new_X))
    points = get_8_points(model_two, new_X)
    final_points = shape_matching(mean_shape.reshape(-1,2), points.reshape(-1,2))
    
    pivot = np.round(np.average(final_points,axis=0),0)
    #theta = np.arctan(abs(pivot[0]-final_points[0][0])/abs(pivot[1]-final_points[0][1])) #radian output
    theta = np.arctan2(pivot[0]-final_points[0][0], pivot[1]-final_points[0][1])
    theta = np.rad2deg(theta) #degree
    #if final_points[0][0]<pivot[0]: #since rotation perform counterclockwise if top slightly left compare to center the need clock-wise = 360-counter-degree
    #    theta = 360 - theta
    new_X = transform.rotate(new_X,-theta,center=pivot)
    final_points[:,0],final_points[:,1] = rotate(pivot,(final_points[:,0],final_points[:,1]),theta)
    
    new_X = transform.resize(new_X,new_dim(new_X))
    points = get_8_points(model_two, new_X)
    final_points = shape_matching(mean_shape.reshape(-1,2), points.reshape(-1,2))
    
    margin = 20
    min_x = max(min(final_points[:,0])-margin,0)
    max_x = min(max(final_points[:,0])+margin,new_X.shape[1])
    min_y = max(min(final_points[:,1])-margin,0)
    max_y = min(max(final_points[:,1])+margin,new_X.shape[0])
    final_points[:,0] -= min_x
    final_points[:,1] -= min_y
    new_X = new_X[int(min_y):int(max_y),int(min_x):int(max_x),:]
    
    center = np.round(np.average(final_points,axis=0),0)
    x_size = 30
    y_size = 100
    patch = new_X[int(center[1]-y_size*0.7):int(center[1]+y_size*1.3),int(center[0]-x_size*1.7):int(center[1]+x_size*0.3)]
    
    plot_points(plt,final_points,'x')
    plt.plot(np.average(final_points[:,0]),np.average(final_points[:,1]),'+')
    plt.imshow(new_X)
    plt.show()
    
    io.imsave(path+'patch/'+str(n)+'.png',(patch * 255).astype(np.uint8))
    #%%
    #plt.scatter(points.T[0],points.T[1])
    plt.scatter(final_points.T[0],final_points.T[1])
    #plt.scatter(fnp.T[0],fnp.T[1])
    plt.imshow(new_X)
    plt.show()
#%% make the main code a function
    
def patch_extraction(image, model_one, model_two, device):
    print('Extracting a patch')
    if max(image.shape)>3000:
        new_X = transform.rescale(image, 1/4, channel_axis=2)
    elif max(image.shape)>=2000:
        new_X = transform.rescale(image, 1/2, channel_axis=2)
    new_X = transform.resize(image,new_dim(new_X))
    new_X = get_upright_image(model_one,model_two,new_X,device)
    new_X = transform.resize(new_X,new_dim(new_X))
    points = get_8_points(model_two, new_X)
    final_points = shape_matching(mean_shape.reshape(-1,2), points.reshape(-1,2))
    #final_points = shape_matching(mean_shape.reshape(-1,2), final_points)
    
    #plot_points(plt,final_points,'x')
    #plt.imshow(new_X)
    #plt.show()
    margin = 70
    min_x = max(min(final_points[:,0])-margin,0)
    max_x = min(max(final_points[:,0])+margin,new_X.shape[1])
    min_y = max(min(final_points[:,1])-margin,0)
    max_y = min(max(final_points[:,1])+margin,new_X.shape[0])
    final_points[:,0] -= min_x
    final_points[:,1] -= min_y
    new_X = new_X[int(min_y):int(max_y),int(min_x):int(max_x),:]
    
    
    new_X = transform.resize(new_X,new_dim(new_X))
    new_X = get_upright_image(model_one,model_two,new_X,device)
    new_X = transform.resize(new_X,new_dim(new_X))
    points = get_8_points(model_two, new_X)
    final_points = shape_matching(mean_shape.reshape(-1,2), points.reshape(-1,2))
    
    pivot = np.round(np.average(final_points,axis=0),0)
    theta = np.arctan2(pivot[0]-final_points[0][0], pivot[1]-final_points[0][1])
    theta = np.rad2deg(theta) #degree
    new_X = transform.rotate(new_X,-theta,center=pivot)
    final_points[:,0],final_points[:,1] = rotate(pivot,(final_points[:,0],final_points[:,1]),theta)
    
    new_X = transform.resize(new_X,new_dim(new_X))
    points = get_8_points(model_two, new_X)
    final_points = shape_matching(mean_shape.reshape(-1,2), points.reshape(-1,2))
    
    margin = 20
    min_x = max(min(final_points[:,0])-margin,0)
    max_x = min(max(final_points[:,0])+margin,new_X.shape[1])
    min_y = max(min(final_points[:,1])-margin,0)
    max_y = min(max(final_points[:,1])+margin,new_X.shape[0])
    final_points[:,0] -= min_x
    final_points[:,1] -= min_y
    new_X = new_X[int(min_y):int(max_y),int(min_x):int(max_x),:]
    
    center = np.round(np.average(final_points,axis=0),0)
    x_size = 30
    y_size = 100
    patch = new_X[int(center[1]-y_size*0.7):int(center[1]+y_size*1.3),int(center[0]-x_size*1.7):int(center[1]+x_size*0.3)]
    
    plot_points(plt,final_points,'x')
    plt.plot(np.average(final_points[:,0]),np.average(final_points[:,1]),'+')
    plt.imshow(new_X)
    plt.show()
    
    return patch, new_X
#%%
from PIL import Image
import pillow_heif
path='./Froggy ID pictures/'
done = ['B1','B2','B3', 'B5', 'E4', 'E5']
tanks = [p for p in os.listdir(path) if p not in done]
pillow_heif.register_heif_opener()
for tank in tanks:
    frogs = os.listdir(path+tank+'/')
    for frog in frogs:
        if frog in ['E6.01','E6.02', 'E6.03','E6.04','E6.05']:
            continue
        X=[]
        name = os.listdir(path+tank+'/'+frog+'/')
        if 'patch' in name:
            name.remove('patch')
        else:
            os.makedirs(path+tank+'/'+frog+'/patch/')
        for i in name:
            if i.split('.')[-1] != "HEIC":
                X += [io.imread(path+tank+'/'+frog+'/'+i)]
            else:
                tmp = Image.open(path+tank+'/'+frog+'/'+i)
                tmp.save(path+tank+'/'+frog+'/'+'.'.join(i.split('.')[:-1]+['jpg']))
                X += [io.imread(path+tank+'/'+frog+'/'+'.'.join(i.split('.')[:-1]+['jpg']))]
                os.remove(path+tank+'/'+frog+'/'+i)
                del tmp
        del i
        for i,p in enumerate(X):
            print('At',tank,frog,i)
            patch, new_X = patch_extraction(p, model_one, model_two, device)
            try:
                io.imsave(path+tank+'/'+frog+'/'+'patch/'+'.'.join(str(name[i]).split('.')[:-1])+'.png',(patch * 255).astype(np.uint8))
            except:
                print('patch error/nsaving reduced image instead')
                io.imsave(path+tank+'/'+frog+'/'+'patch/'+'.'.join(str(name[i]).split('.')[:-1])+'.png',(new_X * 255).astype(np.uint8))
        del i, name
        
#%%
path='./Froggy ID pictures/'
tanks = os.listdir(path)
tanks.remove('B1')
for tank in tanks:
    frogs = os.listdir(path+tank+'/')
    for frog in frogs:
        X=[]
        name = os.listdir(path+tank+'/'+frog+'/')
        for s in name:
            s = s.split('.')
            if s[-1]=="HEIC":
                
#%%
