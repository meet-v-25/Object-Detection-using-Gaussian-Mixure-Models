################################################################## Imports ####################################################################

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

###################################################### Makes object for training set #######################################################

class Make_Object():
    def __init__(self, img, K, alpha, mu, sigma, weight, T):
        self.k=K; self.alpha=alpha; self.T=T; 
        self.TotalPixels = img.shape[0]*img.shape[1]
        self.MoG_pixels = np.zeros([1, self.TotalPixels], int)
        self.mu = np.full([self.k, self.TotalPixels], mu)
        self.sigma = np.full([self.k, self.TotalPixels], sigma)
        self.wt = np.full([self.k, self.TotalPixels], weight)

################################################# To load video frames into program ########################################################

def Load_Dataset(root):
    images=[]; files_names=os.listdir(root); 
    for i in files_names:
        file_path=os.path.join(root,i)
        images.append(cv.imread(file_path))
    return images

######################################################### To Train the Training frames ######################################################


def MoG_Train_Channels(model, image, K, alpha, mu, input_sigma, weight):

    channels = cv.split(image);         # Dividing into channels
    rows = channels[0].shape[0]; cols = channels[0].shape[1]; 
    for i in range(3):                  # Iterating over channels
        Channel_Model=model[i]; channel=channels[i]; 

        for iii in range(rows):
            for jjj in range(cols):
                # For each pixel, initially, it has no match
                matched = False
                for kkk in range (Channel_Model.MoG_pixels[0, iii*cols+jjj]):
                    
                    # For each MoG Distribution pixel, take params
                    this_pixel = channel[iii, jjj]
                    this_alpha = Channel_Model.alpha
                    this_mu = Channel_Model.mu[kkk,iii*cols+jjj]
                    this_sigma = Channel_Model.sigma[kkk,iii*cols+jjj]
                    this_weight = Channel_Model.wt[kkk,iii*cols+jjj]

                    difference=abs(this_pixel-this_mu); 
                    if(difference>(2.5*this_sigma)):           # No match found. Update weight only. 
                        Channel_Model.wt[kkk,iii*cols+jjj] = (1-this_alpha) * this_weight; 
                    else:                                   # If diff<=2.5*sigma, we have a match. Update params. 
                        matched=True
                        prob = (1/((2*np.pi)**0.5)*this_sigma) * np.exp(-( (this_pixel-this_mu)**2 / (2*(this_sigma**2)) ))     # Gaussian Prob.
                        rho = this_alpha*prob       # Second Learning rate
                        
                        this_mu = Channel_Model.mu[kkk,iii*cols+jjj] = (1-rho)*this_mu + rho*this_pixel         # Update mu
                        this_weight = Channel_Model.wt[kkk,iii*cols+jjj] = (1-this_alpha)*this_weight + this_alpha  # Update weight
                        
                        if(this_sigma<input_sigma/2): Channel_Model.sigma[kkk,iii*cols+jjj] = input_sigma/2
                        else: Channel_Model.sigma[kkk,iii*cols+jjj] = ( (1-rho)*(this_sigma**2) + rho*(difference**2) )**0.5
                        break
                    
                    # Sorting and Rearranging the model in decreasing order of (weight/sigma) ratio
                    CM=Channel_Model; index=iii*cols+jjj; 
                    ratio = CM.wt[:, index] / CM.sigma[:, index]
                    xx,Channel_Model.wt[:, index] = zip(*sorted(zip(ratio, CM.wt[:, index]), reverse=True))
                    xx,Channel_Model.mu[:, index] = zip(*sorted(zip(ratio, CM.mu[:, index]), reverse=True))
                    xx,Channel_Model.sigma[:, index] = zip(*sorted(zip(ratio, CM.sigma[:, index]), reverse=True))
                
                if(matched==False):         # If not matched, we need to add a new distribution
                    this_MoG_pixels = Channel_Model.MoG_pixels[0, iii*cols+jjj]

                    # Case 1 : Total distributions are less than K, just add a new distri. 
                    if(this_MoG_pixels<Channel_Model.k):
                        Channel_Model.wt [this_MoG_pixels, iii*cols+jjj] = weight
                        Channel_Model.mu [this_MoG_pixels, iii*cols+jjj] = channel[iii,jjj]
                        Channel_Model.sigma [this_MoG_pixels, iii*cols+jjj] = input_sigma
                        Channel_Model.MoG_pixels[0,iii*cols+jjj] = this_MoG_pixels + 1
                    
                    # case 2 : There are K disri. Replace the lowest disribution (at index K-1)
                    else:
                        Channel_Model.wt [Channel_Model.k-1, iii*cols+jjj] = weight
                        Channel_Model.mu [Channel_Model.k-1, iii*cols+jjj] = channel[iii,jjj]
                        Channel_Model.sigma [Channel_Model.k-1, iii*cols+jjj] = input_sigma
                
                # Normalise the weights
                total_weight = sum(Channel_Model.wt[:, iii*cols+jjj])
                if(total_weight!=0): Channel_Model.wt[:, iii*cols+jjj] = Channel_Model.wt[:, iii*cols+jjj] / total_weight


################################################# To process BG using trained model #######################################################

def background_subtract(Channel_Model, img):
    Test_Channels=cv.split(img); 
    rows=Test_Channels[0].shape[0]; cols=Test_Channels[0].shape[1]; 
    
    # Finding parameter B for each pixel
    for model in Channel_Model:             # For each channel's model in MODEL
        for pixel_index in range(rows*cols):    # For each pixel of model
            total_weight = 0                    # to store sum of weights
            for k in range(model.MoG_pixels[0, pixel_index]):
                total_weight = total_weight + model.wt[k, pixel_index]
                # If summation(weights)>T condition is satisfied, store it as argmin B. 
                if(total_weight>model.T): model.MoG_pixels[0,pixel_index]=k+1; break
    
    result_img=np.full([rows, cols],255,np.uint8)
    for iii in range(rows):
        for jjj in range(cols):
            hits=0
            for kkk in range(len(Channel_Model)):
                for lll in range(Channel_Model[kkk].MoG_pixels[0, iii*cols+jjj]):
                    difference = abs(Test_Channels[kkk][iii,jjj] - Channel_Model[kkk].mu[lll, iii*cols+jjj])
                    if(difference<=(2.5*Channel_Model[kkk].sigma[lll, iii*cols+jjj])): hits+=1; break; 
            # If all the three channels says it is a backgroung pixel, make it black
            if(hits==3): result_img[iii,jjj]=0;       # 3 = no of channels
    return result_img

################################################### To do connected components method ######################################################

def Do_Connected_Components(img):
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (2,2)); kernel2 = cv.getStructuringElement(cv.MORPH_ERODE, (3,3)); 
    final_parsed = cv.morphologyEx(cv.morphologyEx(img, cv.MORPH_OPEN, kernel1, iterations=2), cv.MORPH_CLOSE, kernel2, iterations=2)
    return final_parsed

###################################################### Generates final img frames ##########################################################

def Generate_Results(images, models):
    subtracted_bg_results = [] ; connected_component_results = [] ; 
    for i in range(len(images)):
        subtracted_bg_results.append(background_subtract(models, images[i]));         
        connected_component_results.append(Do_Connected_Components(subtracted_bg_results[-1])); 
    return subtracted_bg_results, connected_component_results

######################################################### To plot the results ################################################################

def plot_results(org_imgs, subtr_imgs, conn_comp_imgs, delay):    
    plt.ion()
    for i in range(len(org_imgs)):
        plt.subplot(131); plt.title('Original'); plt.imshow(org_imgs[i]); 
        plt.subplot(132); plt.title('Results'); plt.imshow(subtr_imgs[i], cmap='gray'); 
        plt.subplot(133); plt.title('Improved'); plt.imshow(conn_comp_imgs[i], cmap='gray'); 
        plt.pause(delay); plt.clf(); 
    plt.ioff()

########################################################## Initializing Parameters ##########################################################

K = 5
alpha = 0.0001
init_mu = 0
init_sigma = 25
init_weight = 0.05
T = 0.7
output_video_delay = 0.5      # in seconds

############################################################# The Main Code ################################################################


print("\nInitiating the process.")

initiating_image = cv.imread('Train/Img (00).png')
R,G,B = cv.split(initiating_image)
R_MoG = Make_Object(R,K,alpha,init_mu,init_sigma,init_weight,T)
G_MoG = Make_Object(G,K,alpha,init_mu,init_sigma,init_weight,T)
B_MoG = Make_Object(B,K,alpha,init_mu,init_sigma,init_weight,T)
MoG_models = R_MoG, G_MoG, B_MoG

train_path='Train/'; test_path='Test/'; 
training_images=Load_Dataset(train_path)
len_training_images=len(training_images)

print("\nCorresponding frames are being/done training : ", end='')
for i in range(len_training_images):
    MoG_Train_Channels (MoG_models, training_images[i], K, alpha, init_mu, init_sigma, init_weight)
    print(str(i), end=' ')

print('\n\nTraining Done. Now testing. It will show the video-type pyplot.')

testing_images=Load_Dataset(test_path)
subtracted_bg_results, connected_component_results = Generate_Results(testing_images, MoG_models)
plot_results(testing_images, subtracted_bg_results, connected_component_results, output_video_delay)

print("\nCompleted Execution.\n")


##############################################################################################################################################
############################################################# THE END ########################################################################
##############################################################################################################################################
