# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 08:32:26 2023

@author: Eedrobup
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io,color
from scipy.stats import norm

import config

working_directory = config.PersonalConfig.call_wd()
os.chdir(working_directory)

#evaluation function
class data:
    def __init__(self,data):
        """
        According to central limit theorem
        the samples are normally distributed
        to handle some noise I use median and IQR to approximate mean, std of normal curve
        """
        median = np.median(data)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        est_std = IQR / 1.349
        self.data = data
        self.dist = norm(median,est_std)
        #self.dist = norm(np.mean(data),np.std(data))
    def get_pdf(self, num_bin=10, start = 0, end = 1):
        x, dx = np.linspace(start, end, num=num_bin, retstep=True)
        pdf_list = np.zeros((2,len(x)))
        pdf_list[0,:] = x
        pdf_list[1,:] = self.dist.pdf(x)*dx
        return pdf_list
    def get_cdf(self, num_bin=10, start = 0, end = 1):
        x = np.linspace(start, end, num=num_bin)
        cdf_list = np.zeros((2,len(x)))
        cdf_list[0,:] = x
        cdf_list[1,:] = self.dist.cdf(x)
        return cdf_list
    def get_mean_std(self):
        """return mean, std of normal dist"""
        return self.dist.mean(), self.dist.std()
    
def mod_b_coef(x1,x2):
    """
    Bhattacharyya Coefficient, but 1 is dissimilar and 0 is similar
    x1: comparative class
    x2: same class
    """
    p = data(x2)
    q = data(x1)
    p_mean, p_std = p.get_mean_std()
    num_bin = 1000
    p_pdf = p.get_pdf(num_bin, start = p_mean-4*p_std, end = p_mean+4*p_std)
    q_pdf = q.get_pdf(num_bin, start = p_mean-4*p_std, end = p_mean+4*p_std)
    return 1-np.sum(np.sqrt(p_pdf[1,:]*q_pdf[1,:]))

def H_dist(x1,x2):
    """
    hellinger_distance
    x1: comparative class
    x2: same class
    """
    p = data(x2)
    q = data(x1)
    p_mean, p_std = p.get_mean_std()
    q_mean, q_std = q.get_mean_std()
    num_bin = 1000
    start = min(p_mean-4*p_std,q_mean-4*q_std)
    end = max(p_mean+4*p_std,q_mean+4*q_std)
    p_pdf = p.get_pdf(num_bin, start, end)
    q_pdf = q.get_pdf(num_bin, start, end)
    return np.sqrt(np.sum((np.sqrt(p_pdf[1,:]) - np.sqrt(q_pdf[1,:])) ** 2)) / np.sqrt(2)

def KL_divergence(x1, x2,epsilon = 0.00001):
    """
    x1: comparative class
    x2: same class
    """
    p = data(x2)
    q = data(x1)
    p_mean, p_std = p.get_mean_std()
    num_bin = 1000
    p_pdf = p.get_pdf(num_bin, start = p_mean-4*p_std, end = p_mean+4*p_std)
    q_pdf = q.get_pdf(num_bin, start = p_mean-4*p_std, end = p_mean+4*p_std)
    for i in range(num_bin):
        if q_pdf[1,i] == 0:
            q_pdf[1,i] = epsilon
    #mix_pdf = p_pdf.copy()
    #mix_pdf[1,:] = (mix_pdf[1,:]+q_pdf[1,:])/2
    KL = np.sum(p_pdf[1,:]*(np.log2(p_pdf[1,:])-np.log2(q_pdf[1,:])))
    return KL

def modified_KL(x1,x2):
    KL = KL_divergence(x1, x2)
    return np.arctan(KL)/(2*np.pi)

def JS_divergence(x1, x2):
    """
    x1: comparative class
    x2: same class
    """
    return (KL_divergence(x1, x2) + KL_divergence(x2, x1))/2

def Tim_divergence(x1, x2):
    """
    similar to ROC and AUC
    x1: comparative class
    x2: same class
    """
    p = data(x1)
    q = data(x2)
    p_mean, p_std = p.get_mean_std()
    num_bin = 1000
    p_pdf = p.get_pdf(num_bin, start = p_mean-4*p_std, end = p_mean+4*p_std)
    q_cdf = q.get_cdf(num_bin, start = p_mean-4*p_std, end = p_mean+4*p_std)
    Tim = np.sum(p_pdf[1,:]*(1-q_cdf[1,:]))
    return Tim

def Tim_divergence_2(x1,x2):
    """
    similar to ROC and AUC
    x1: comparative class
    x2: same class
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.sum([1/len(x1)*(1-sum(x2<=i)/len(x2)) for i in x1])

def divergence_report(all_image, div_function, compare_function):
    """
    all_image shape (classes, samples, dimensions) : post_featurized
    div_function: divergence function to compare two distribution
    compare_function: any comparison function with two parameters (comparative_vector,reference_vector)
    *** beware of assymetry output for div_function and compare_function
    
    output: quality of divergence (scalar), divergence report (shape (n_class,n_class))
    """
    n_class = len(all_image)
    inter_div_report = np.zeros((n_class,n_class))
    ref_image_return = []
    total_fit = 0
    intra_return = []
    for i in range(0,n_class): #ref (same class)
        #compute intraclass distribution and select representative vector
        n_sample = len(all_image[i])
        intraclass_matrix = np.zeros((n_sample,n_sample))
        for a in range(0,n_sample):
            for b in range(0,n_sample):
                intraclass_matrix[a,b] = compare_function(all_image[i,a],all_image[i,b])
        select = np.argmax(np.average(intraclass_matrix,axis=0))
        intraclass_dist = intraclass_matrix[select]
        intra_return += [intraclass_dist]
        np.delete(intraclass_dist, np.argwhere(intraclass_dist == max(intraclass_dist)))
        ref_image = all_image[i,select]
        ref_image_return += [ref_image]
        
        #compute interclass distribution
        for j in range(0,n_class): #comparison
            if i==j:
                inter_div_report[i,j] = np.nan
                continue
            interclass_dist = [compare_function(ref_image,com) for com in all_image[j]]
            inter_div_report[i,j] = div_function(interclass_dist, intraclass_dist)
            total_fit += inter_div_report[i,j]
    return total_fit/(n_class*(n_class-1)), inter_div_report, intra_return, ref_image_return

#%% loading the pictures

def crop_pic(X,y_size,x_size):
    if X.shape[1]<x_size:
        x_size=X.shape[1]
    if X.shape[0]<y_size:
        y_size=X.shape[0]
    return X[:y_size,:x_size]

def load_pic(path,y_size,x_size):
    X=[]
    name = os.listdir(path)
    for i in name:
        if os.path.isdir(path+i):
            continue
        P=crop_pic(np.average(io.imread(path+i)/255,axis=2),y_size,x_size)
        if P.shape != (y_size,x_size):
            P = np.pad(P,((0,y_size-P.shape[0]),(0,x_size-P.shape[1])),mode='constant',constant_values=(0))
        X += [P]
    return X

def vec_size(vector):
    """assumed numpy 1d vector"""
    return (np.sum(vector**2))**(1/2)
#%% define local binary pattern function and local histogram function
def lbp_coding(image, radius=1, num_points=8):
    height, width = image.shape
    lbp_coded = np.zeros((height, width))

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            # Extract the neighboring pixels around the central pixel
            pixels = [
                image[y - radius, x - radius],
                image[y - radius, x],
                image[y - radius, x + radius],
                image[y, x + radius],
                image[y + radius, x + radius],
                image[y + radius, x],
                image[y + radius, x - radius],
                image[y, x - radius]
            ]

            # Calculate the LBP value for the central pixel
            center_pixel = image[y, x]
            lbp_value = 0
            for i in range(num_points):
                lbp_value += (pixels[i] >= center_pixel) * (1 << i)

            # Assign the LBP value to the corresponding pixel in the LBP coded image
            lbp_coded[y, x] = lbp_value

    return lbp_coded

def compute_local_histograms(image, num_cells=8, radius=1, num_points=8, num_bins=256):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        if max(image)>1:
            image = np.average(image/255,axis=image.shape.index(3))
        else:
            image = np.average(image,axis=image.shape.index(3))

    height, width = image.shape
    cell_height = height // num_cells
    cell_width = width // num_cells

    local_histograms = []
    for i in range(num_cells):
        for j in range(num_cells):
            # Extract the cell from the image
            cell = image[i*cell_height: (i+1)*cell_height, j*cell_width: (j+1)*cell_width]

            # Compute the LBP coded image for the cell
            lbp_coded_cell = lbp_coding(cell, radius, num_points)

            # Extract the histogram from the LBP coded cell
            histogram = np.histogram(lbp_coded_cell, bins=num_bins, range=(0, num_bins - 1))[0]

            # Normalize the histogram to make it robust against changes in cell size
            histogram = histogram.astype(np.float32) / np.sum(histogram)

            local_histograms.append(histogram)

    return np.concatenate(local_histograms)
#%%
import cv2

def float_to_uint8(img_float):
    if np.max(img_float) <= 1.0:
        return (img_float * 255).astype(np.uint8)
    else:
        return img_float.astype(np.uint8)

def extract_sift_features(image):
    
    # Check proper input
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = float_to_uint8(image)
    
    # Convert the image to grayscale (if it's not already)
    if len(image.shape) == 3 and image.shape[2] == 3: 
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    
    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    return keypoints, descriptors


#%% Main


y_size = 200
x_size = 110

X1 = load_pic('./Froggy ID pictures/B1/B1.01/patch/',y_size,x_size)
X3 = load_pic('./Froggy ID pictures/B1/B1.03/patch/',y_size,x_size)
X5 = load_pic('./Froggy ID pictures/B1/B1.05/patch/',y_size,x_size)
X7 = load_pic('./Froggy ID pictures/B1/B1.07/patch/',y_size,x_size)
X8 = load_pic('./Froggy ID pictures/B1/B1.08/patch/',y_size,x_size)

#plt.imshow(X1[0])
#plt.show()
#%% import all images
from skimage.feature import hog

def vnorm(vector):
    """normalize image vector"""
    x_bar = np.average(vector.reshape(-1))
    u_vector = vector.reshape(-1)-x_bar
    return u_vector/vec_size(u_vector)

def featurize(X, descriptor=None):
    if descriptor != None:
        if descriptor == "hog":
            X = hog(X, orientations=11, pixels_per_cell=(32, 32),
                         cells_per_block=(2, 2), feature_vector=True,channel_axis=None)
        elif descriptor == "LBP_hist":
            X = compute_local_histograms(X,num_cells=3)
    return X


def get_system(path='./Froggy ID pictures/', y_size = 200, x_size = 110):
    subjects = pd.DataFrame(columns=['path','files','count','images','norm_images','lbp_images',
                                     'hog_images','best_hog','best_lbp','best_norm',
                                     'inclass_stat_hog','inclass_stat_lbp','inclass_stat_norm',
                                     'interdist_hog','interdist_lbp','interdist_norm',
                                     'T_score_hog','T_score_lbp','T_score_norm']) 
    for tank in os.listdir(path):
        frogs = os.listdir(path+tank+'/')
        for frog in frogs:
            print('processing',frog)
            subjects.loc[frog,'path'] = path+tank+'/'+frog+'/patch/'
            subjects.loc[frog,'files'] = [x for x in os.listdir(subjects.loc[frog,'path']) if x != 'error']
            subjects.loc[frog,'count'] = len(subjects.loc[frog,'files'])
            subjects.loc[frog,'images'] = np.transpose(np.stack(load_pic(subjects.loc[frog,'path'],y_size,x_size),axis=2),axes=(2,0,1))
            subjects.loc[frog,'norm_images'] = np.stack([vnorm(i) for i in subjects.loc[frog,'images']],axis=0)
            subjects.loc[frog,'lbp_images'] = np.stack([vnorm(featurize(i,descriptor='LBP_hist')) for i in subjects.loc[frog,'images']],axis=0)
            subjects.loc[frog,'hog_images'] = np.stack([vnorm(featurize(i,descriptor='hog')) for i in subjects.loc[frog,'images']],axis=0)

            print('comparing intraclass')
            intraclass_matrix = np.zeros((3,subjects.loc[frog,'count'],subjects.loc[frog,'count']))
            for i in range(subjects.loc[frog,'count']):
                for j in range(subjects.loc[frog,'count']):
                    if i==j:
                        intraclass_matrix[0,i,j], intraclass_matrix[1,i,j] ,intraclass_matrix[2,i,j] = np.nan,np.nan,np.nan
                        continue
                    for k,n in enumerate(['hog','lbp','norm']):
                        intraclass_matrix[k,i,j] = np.dot(subjects.loc[frog, n+'_images'][i],subjects.loc[frog,n+'_images'][j])
            
            print('constructing report stats')
            for i,n in enumerate(['hog','lbp','norm']):
                subjects.loc[frog,'best_'+n] = np.argsort(np.nanmean(intraclass_matrix[i],axis=0))[::-1]
                
                ref = subjects.loc[frog, n+'_images'][subjects.loc[frog,'best_'+n][0]]
                result = [np.dot(ref,p) if i!= subjects.loc[frog,'best_'+n][0] else np.nan
                          for i,p in enumerate(subjects.loc[frog,n+'_images'])]
                subjects.loc[frog,'inclass_stat_'+n] = (np.nanmean(result), np.nanstd(result))
                del ref, result
            del intraclass_matrix
            
            print('comparing interclass distribution')
            for n in ['hog','lbp','norm']:
                for ref_frog in subjects.index:
                    ref = subjects.loc[ref_frog, n+'_images'][subjects.loc[ref_frog,'best_'+n][0]]
                    result = {target_frog:[np.dot(ref,target) for target in subjects.loc[target_frog, n+'_images']] for target_frog in subjects.index if ref_frog!=target_frog}
                    result.update({ref_frog :[np.dot(ref,p) for im_no, p in enumerate(subjects.loc[ref_frog, n+'_images']) if im_no != subjects.loc[ref_frog,'best_'+n][0]]})
                    subjects.loc[ref_frog,'interdist_'+n] = [result]
                    subjects.loc[ref_frog,'T_score_'+n] = [{target_frog:Tim_divergence(subjects.loc[ref_frog,'interdist_'+n][0][target_frog], subjects.loc[ref_frog,'interdist_'+n][0][ref_frog]) for target_frog in subjects.index if ref_frog!=target_frog}]
            del ref, result
    print('completed')
    return subjects
subjects = get_system()
#%% Top 1 3 and 5 scoring system

def get_top_matrix(subjects, feature = 'hog'):
    n = feature
    result =[]
    for frog in subjects.index:
        for im in subjects.loc[frog,n+'_images']:
            compare1 = np.array([[ref_frog,np.dot(im,subjects.loc[ref_frog, n+'_images'][subjects.loc[ref_frog,'best_'+n][0]])] for ref_frog in subjects.index])
            indices = np.argsort(compare1[:,1])[::-1] #in descending order
            match_result = compare1[indices]
            print(match_result[0,1])
            if float(match_result[0,1])<0.44:
                compare2 = np.array([[ref_frog,np.dot(im,subjects.loc[ref_frog, n+'_images'][subjects.loc[ref_frog,'best_'+n][1]])] for ref_frog in subjects.index])
                indices = np.argsort(compare2[:,1])[::-1]
                spare_result = compare2[indices]
                if compare2[0,1]>compare1[0,1]:
                    match_result = spare_result
            #print(frog,compare1[indices][0])
            #result += [compare1[indices][0][1]]
            #print(match_result)
            result += [list(match_result[:,0]).index(frog)+1] #[,float(compare1[indices][0][1])]
    return result

top_hog = get_top_matrix(subjects,'hog')
top_lbp = get_top_matrix(subjects,'lbp')
top_norm = get_top_matrix(subjects,'norm')
#%% general histogram
import random

for n in ['hog','lbp','norm']:
    n='norm'
    formal = {'hog':'Histogram of Oriented Gradients','lbp':'Local Binary Pattern Histogram','norm':'Normalized Template Matching'}
    diff_class = sum([[np.dot(im1,im2) for im1 in subjects.loc[f1,n+'_images'] for im2 in subjects.loc[f2, n+'_images']] for f2 in subjects.index for f1 in subjects.index if f1!=f2],[])
    #same_class = sum([[np.dot(im1,im2) for i,im1 in enumerate(subjects.loc[f1,n+'_images']) for j,im2 in enumerate(subjects.loc[f2, n+'_images']) if i!=j] for f2 in subjects.index for f1 in subjects.index if f1==f2],[])
    ref_same_class = np.array(sum([[np.dot(im1,subjects.loc[f,n+'_images'][subjects.loc[f,'best_'+n][0]]) for i,im1 in enumerate(subjects.loc[f,n+'_images']) if i!=subjects.loc[f,'best_'+n][0]] for f in subjects.index],[]))
    diff_sam = np.array(diff_class)#random.sample(diff_class,len(ref_same_class))
    score = Tim_divergence_2(diff_sam,ref_same_class)
    plt.hist(diff_sam, bins=30,density=True,alpha=0.5)
    plt.hist(ref_same_class, bins=30,density=True,alpha=0.5)
    leg = plt.legend(['different_class','same_class'])
    plt.title('Distribution of different class comparison and same class comparison\nusing '+formal[n])
    plt.xlim([-0.5,1.0])
    #plt.ylabel('Probability density')
    plt.xlabel('Matching score')
    #plt.yticks()
    plt.gca().set_yticklabels(['{:,.2f}'.format(x/100) for x in plt.gca().get_yticks()])
    plt.gca().set_xticklabels(['{:,.2f}'.format(x) for x in plt.gca().get_xticks()])
    plt.draw()

    p = leg.get_frame().get_bbox().bounds
    
    #plt.annotate(f'Criterion score: {score:.2f}', (p[0], p[1]), (p[0]-p[3]/2, p[1]-p[3]/2), 
    #            xycoords='figure pixels', zorder=9)
    plt.text(0.58,4.0,f'Criterion score: {0.70:.2f}')
    #plt.text(0.5,2.8,f'Criterion score: {0.95:.2f}')
    #plt.text(0.27,12.7,f'Criterion score: {0.84:.2f}')
    #plt.hist(same_class, bins=50,alpha=0.5)
    plt.savefig('nomd_3_G_histogram.png',dpi=200)
    
    plt.show()
    
    print(n,':',Tim_divergence_2(random.sample(diff_class,len(ref_same_class)),ref_same_class))
#%% Two dist curve figure drawing

plt.plot(np.arange(-4, 1, 0.1), norm.pdf(np.arange(-4, 1, 0.1), -1.5, 1))
plt.plot(np.arange(-1, 4, 0.1), norm.pdf(np.arange(-1, 4, 0.1), 1.5, 1),color='orange')
plt.plot([-1.5,-1.5],[0.015,0.397],color='blue',alpha=0.7,linewidth=0.7)
plt.text(-1.6,0.405,r'$\mathbb{P}_a({x}_1)$',color='blue')
plt.text(-1.65,0.00,r'$\ {x}_1$')

plt.plot(np.arange(-1, 4, 0.1), norm.pdf(np.arange(-1, 4, 0.1), 1.5, 1))
plt.fill_between(
        x= np.arange(-1, 1, 0.1), 
        y1= norm.pdf(np.arange(-1, 1, 0.1), 1.5, 1),
        y2=0.015,
        color= "orange",
        alpha= 0.2)
plt.text(0.1,0.12,r'$\mathbb{C}_b({x}_2)$',color='chocolate')
plt.text(0.8,0.00,r'$\ {x}_2$')

plt.plot([-4,4],[0.015,0.015],color='black',linewidth=0.7)
plt.plot([-4,-4],[0.015,0.4],color='black',linewidth=0.7)
plt.title('Two Distribution Curves')
plt.axis('off')
plt.legend(['Curve a','Curve b'])
plt.savefig('two_histogram.png',dpi=200)
plt.show()


#%% plot top accuracy
import matplotlib as mpl
mpl.rc('font',family='Arial')

plotnorm = np.array([[top,sum([i<=top for i in top_norm])/1220] for top in range(1,70)])
plotlbph = np.array([[top,sum([i<=top for i in top_lbp])/1220] for top in range(1,70)])
plothog = np.array([[top,sum([i<=top for i in top_hog])/1220] for top in range(1,70)])
gcolor = ['blue','orange','green']

xshift = 1.25
yshift = 0.99
for no,descriptor in enumerate([plotnorm,plotlbph,plothog]):
    plt.plot(descriptor.T[0],descriptor.T[1],color = gcolor[no])
plt.legend(['Template_matching','Local_Binary_Pattern_Histogram','Histogram_of_Gradient'])

for no,descriptor in enumerate([plotnorm,plotlbph,plothog]):
    for i in [1,3,5]:
        i-=1
        plt.scatter(i+1, descriptor.T[1][i], s= 20,color = gcolor[no])
        if i==2 and no!=2:
            plt.text(i+2.2, descriptor.T[1][i]*yshift-0.011,f'Top{i+1}: {descriptor.T[1][i]:.0%}',color = gcolor[no])
            continue
        if i==4:
            plt.text(i+3, descriptor.T[1][i]*yshift,f'Top{i+1}: {descriptor.T[1][i]:.0%}',color = gcolor[no])
            continue
        if i==0 and no==2:
            plt.text(i+2, descriptor.T[1][i]*yshift+0.03,f'Top{i+1}: {descriptor.T[1][i]:.0%}',color = gcolor[no])
            continue
        if i==0 and no!=2:
            plt.text(i+2, descriptor.T[1][i]*yshift,f'Top{i+1}: {descriptor.T[1][i]:.0%}',color = gcolor[no])
            continue
        plt.text(i+2.3, descriptor.T[1][i]*yshift,f'Top{i+1}: {descriptor.T[1][i]:.0%}',color = gcolor[no])

plt.title('Recognition Accuracy Per Descriptor')
plt.rcParams["font.family"] = "cursive"
plt.ylabel('Accuracy of recognition in the top N elements')
plt.xlabel('N')
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.savefig('recognition_accuracy.png', dpi=200)
plt.show()




#%% Try template matching

in_list = [X1,X3,X5,X7,X8]

#pre-processing for template matching
vector = np.zeros((len(in_list),len(in_list[0]),y_size*x_size))
for i,X in enumerate(in_list):
    for j,P in enumerate(X):
        if P.shape != (y_size,x_size):
            P = np.pad(P,((0,y_size-P.shape[0]),(0,x_size-P.shape[1])),mode='constant',constant_values=(0))
        vector[i,j,:] = (P.reshape(-1,y_size*x_size)-np.average(P.reshape(-1,y_size*x_size)))/vec_size(P.reshape(-1,y_size*x_size)-np.average(P.reshape(-1,y_size*x_size)))
        
co_matrix = np.zeros((len(in_list),len(in_list[0]),len(in_list[0])))

for i in range(0,len(vector)):
    for j in range(0,len(vector[i])):
        for k in range(0,len(vector[i])):
            co_matrix[i,j,k] = np.dot(vector[i,j,:],vector[i,k,:])

tmp_diff = []
for j in range(0,20):
    tmp_diff += [np.dot(vector[0,17],vector[1,j])]
        
tmp_same = []
for i in range(0,20):
    for j in range(0,20):
        tmp_same += [np.dot(vector[0,i],vector[0,j])]


select = co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))]
np.delete(co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))], np.argwhere(co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))] == max(co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))])))


plt.hist(np.delete(co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))], np.argwhere(co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))] == max(co_matrix[0,np.argmax(np.average(co_matrix[0],axis=0))]))),color='blue',alpha=0.5)
plt.hist(tmp_diff,color='orange',alpha=0.5)
plt.legend(loc='upper right')
plt.show()

#report
q, table = divergence_report(vector, Tim_divergence, np.dot)
print(divergence_report(vector, mod_b_coef, np.dot))
print(divergence_report(vector, H_dist, np.dot))
#%% local binary pattern and local binary histogram main code

lbp_vector = []
for i,X in enumerate(subjects.loc[:,'images']):
    tmp = []
    for j,p in enumerate(X):
        #if P.shape != (y_size,x_size):
        #    P = np.pad(P,((0,y_size-P.shape[0]),(0,x_size-P.shape[1])),mode='constant',constant_values=(0))
        v = compute_local_histograms(p,num_cells=6, radius=1, num_points=8)
        v = (v-np.mean(v))/vec_size(v)
        tmp += [v]
    lbp_vector += [tmp]
del i,j,tmp, v
lbp_vector = np.array(lbp_vector)
print(divergence_report(lbp_vector, Tim_divergence, np.dot))
print(divergence_report(lbp_vector, mod_b_coef, np.dot))
print(divergence_report(lbp_vector, H_dist, np.dot))
#%% Histogram of oriented gradient main code
from skimage.feature import hog
hog_vector = []
for i,X in enumerate(in_list):
    tmp = []
    for j,P in enumerate(X):
        if P.shape != (y_size,x_size):
            P = np.pad(P,((0,y_size-P.shape[0]),(0,x_size-P.shape[1])),mode='constant',constant_values=(0))
        v = hog(P, orientations=8, pixels_per_cell=(32, 32),
                     cells_per_block=(2, 2), feature_vector=True,channel_axis=None)
        v = (v-np.mean(v))/vec_size(v)
        tmp += [v]
    hog_vector += [tmp]
del i,j,tmp,v
hog_vector = np.array(hog_vector)
print(divergence_report(hog_vector, Tim_divergence, np.dot))
print(divergence_report(hog_vector, mod_b_coef, np.dot))
print(divergence_report(hog_vector, H_dist, np.dot))

#%%
def get_hyperparam_hog(subjects):
    minn = 2
    lim = 1+5
    step = 1
    df = pd.DataFrame(columns=[n+str(i) for n in ['hog_','best_'] for i in range(minn,lim,step)])
    print('re-encode vectors')
    for frog in subjects.index:
        for n in range(minn,lim,step):
            print('computing',n)
            df.loc[frog,'hog_'+str(n)] = np.stack([vnorm(
                #hog(i, orientations=11, pixels_per_cell=(n, n),
                     #cells_per_block=(2, 2), feature_vector=True, channel_axis=None)
                     compute_local_histograms(i,num_cells=n, radius=1, num_points=4)
                                                         )
                                                   for i in subjects.loc[frog,'images']],axis=0)
    print('intraclass')
    for frog in subjects.index:
        intraclass_matrix = np.zeros((len(range(minn,lim,step)),subjects.loc[frog,'count'],subjects.loc[frog,'count']))
        for i in range(subjects.loc[frog,'count']):
            for j in range(subjects.loc[frog,'count']):
                if i==j:
                    for n in range(0,len(range(minn,lim,step))):
                        intraclass_matrix[n,i,j] = np.nan
                    continue
                for k,n in enumerate([n+str(i) for n in ['hog_'] for i in range(minn,lim,step)]):
                    intraclass_matrix[k,i,j] = np.dot(df.loc[frog, n][i],df.loc[frog, n][j])
        
        for i,n in enumerate([str(a) for a in range(minn,lim,step)]):
            df.loc[frog,'best_'+n] = np.argsort(np.nanmean(intraclass_matrix[i],axis=0))[::-1]
        del intraclass_matrix
    print('check interclass')
    dic = {}
    for n in [str(a) for a in range(minn,lim,step)]:
        tmp = []
        for ref_frog in df.index:    
            ref = df.loc[ref_frog, 'hog_'+n][df.loc[ref_frog,'best_'+n][0]]
            result = {target_frog:[np.dot(ref,target) for target in df.loc[target_frog, 'hog_'+n]] for target_frog in df.index if ref_frog!=target_frog}
            result.update({ref_frog :[np.dot(ref,p) for im_no, p in enumerate(df.loc[ref_frog, 'hog_'+n]) if im_no != df.loc[ref_frog,'best_'+n][0]]})
            tmp += [Tim_divergence(result[target_frog], result[ref_frog]) for target_frog in df.index if ref_frog!=target_frog]
        dic[n]=np.mean(tmp)
    return dic

df_lbp_pixel14 = get_hyperparam_hog(subjects)
#%%
mpl.rc('font',family='Arial')
#df_lbp_pixel14 = np.array([[int(key),value]for key,value in df_lbp_pixel14.items()])
#lbph = lbph[np.argsort(lbph[:,0])]
plt.plot(df_hog_pixel[:,0],df_hog_pixel[:,1])
plt.plot(df_hog_pixel[:,0],df_hog_pixel[:,1],'o',color='blue')
#plt.plot(df_lbp_pixel14[:,0],df_lbp_pixel14[:,1])
#plt.plot(df_lbph[:,0],df_lbph[:,1])
#plt.plot(df_lbp_pixel[:,0],df_lbp_pixel[:,1])
#plt.legend(['Radius 1, neighbor 4', 'Radius 1, neighbor 8','Radius 2, neighbor 8'],loc='upper right')
#plt.plot(df_hog[:,0],df_hog[:,1],'o',color='blue')
#plt.plot(df_hog_pixel[:,0],df_hog_pixel[:,1])
#plt.plot(df_hog_pixel[:,0],df_hog_pixel[:,1],'o',color='blue')
#plt.plot(df_lbp_pixel14[:,0],df_lbp_pixel14[:,1],'o',color='blue')
#plt.plot(df_lbph[:,0],df_lbph[:,1],'o',color='orange')
#plt.plot(df_lbp_pixel[:,0],df_lbp_pixel[:,1],'o',color='green')

plt.plot([32,32],[0.84,0.948],color='blue',alpha = 0.4,linestyle='dashed')
#plt.title('Number of histogram cells tuning for Local Binary Pattern Histogram')
plt.title('Number of pixels per cell tuning for Histogram of Oriented Gradients')
plt.ylabel('Criterion score')
#plt.xlabel('Number of cells')
plt.xlabel('Number of pixels per cell')
plt.xticks(np.arange(16, 37+1, 2))
plt.savefig('HOGd_Pix_tuning.png',dpi=200)
plt.show()

#%%
tmp_diff=[]
for j in range(0,20):
    tmp_diff += [np.dot(ref[0],hog_vector[1,j])]

q, table, intra, ref = divergence_report(hog_vector, Tim_divergence, np.dot)
plt.hist(intra[0],color='blue',alpha=0.5)
plt.hist(tmp_diff,color='orange',alpha=0.5)
plt.legend(loc='upper right')
plt.show()
#%%

plt.imshow(example_frog_pic,cmap='gray',alpha=0.5)
li = []
for i in range(0,8):
    #li += [make_gauss_blob(example_frog_pic.shape[:2], int(example_frog_points[i,1]), int(example_frog_points[i,0]),10)]
    plt.imshow(example_frog_pic,cmap='gray',alpha=0.5)
    plt.imshow(make_gauss_blob(example_frog_pic.shape[:2], int(example_frog_points[i,1]), int(example_frog_points[i,0]),10),cmap='plasma',alpha=0.5)
    #plt.text(example_frog_points[i,0],example_frog_points[i,1],i)
    plt.axis('off')
    plt.savefig('blobed_frog'+str(i)+'.png',dpi=200)
    plt.show()
li = np.sum(np.array(li),axis=0)
plt.imshow(li,cmap='plasma',alpha=0.5)
plt.axis('off')
plt.savefig('blobed_frog'+i+'.png',dpi=200)
plt.show()
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores):
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute the equal error rate and the associated threshold
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[eer_index]
    eer = fpr[eer_index]
    eer_tpr = tpr[eer_index]
    alpha005_index = np.nanargmin(np.absolute(fpr-0.05))
    alpha005 = thresholds[alpha005_index]
    alpha005_fpr = fpr[alpha005_index]
    alpha005_tpr = tpr[alpha005_index]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (Area Under the Curve = %0.2f)' % roc_auc)
    plt.scatter([eer], [eer_tpr], color='red', s=100, label=f'Equal Error Rate (Threshold = {eer_threshold:.2f})')  # plot point
    plt.scatter([alpha005_fpr],[alpha005_tpr], s=100, label=f'Type I error at 0.05 (Threshold = {alpha005:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Histogram of Oriented Gradients Descriptor:\nReceiver Operating Characteristic (ROC) Curve', fontsize=26)
    plt.legend(loc="lower right", fontsize=18)
    plt.savefig('HOGROC.png',dpi=200)
    plt.show()
    
plot_roc_curve(np.concatenate([np.ones_like(ref_same_class), np.zeros_like(diff_sam)]),np.concatenate([ref_same_class, diff_sam]))

threshold = 0.19
np.around(confusion_matrix(np.concatenate([np.ones_like(ref_same_class), np.zeros_like(diff_sam)]), np.concatenate([ref_same_class>=threshold, diff_sam>=threshold]),normalize='true'),2)
