import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def normal_vector(pixels) :
    return np.multiply(pixels, 2) - np.ones(3)
    #2 * pixels - 1

def convert_to_array(img) :
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            pixel = img[i,j]/255
        return pixel

def loss_fct(matrix_ov_reg1, matrix_ov_reg2) :
    sum = 0
    for i in range(len(matrix_ov_reg1)) :
        n1 = normal_vector(convert_to_array(matrix_ov_reg1[i]))
        n2 = normal_vector(convert_to_array(matrix_ov_reg2[i]))
        error = np.linalg.norm(n1-n2)
        sum += error
    
    return sum/(len(matrix_ov_reg1)) # len(matrix_ov_reg1) = # of pixels (same as reg2)

def sweep_and_compare(tex1, tex2, crop1, crop2) :
    """
    min_loss_fct = loss_fct(overlap1_a, overlap2_a)
    min_overlap1 = overlap1_a
    min_overlap2 = overlap2_a

    height1 = overlap1_a.shape[0]
    width1 = overlap1_a.shape[1]
    height2 = overlap2_a.shape[0]
    width2 = overlap2_a.shape[1]
    
    while (width1 > 0) :
        
        overlap1_sw = overlap1_a[height1:height1, width1:width1-50]
        overlap2_sw = overlap2_a[height2:height2, width2:width2-50]
        
        curr_loss_fct = loss_fct(overlap1_sw, overlap2_sw)
        
        if (curr_loss_fct < min_loss_fct) :
            min_loss_fct = curr_loss_fct
            min_overlap1 = overlap1_sw #or matrix_ov_reg1
            min_overlap2 = overlap2_sw #or matrix_ov_reg2
        
        overlap1_a = overlap1_sw
        overlap2_a = overlap2_sw
    
    return cv2.imshow('/Users/syrineenneifer/Desktop/best_overlap00' ,min_overlap1), cv2.imshow('/Users/syrineenneifer/Desktop/best_overlap01' ,min_overlap2) #or just min_overlap1
    """

    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    overlap1 = tex1[0:height1, crop1:width1].copy()  #here, crop1 = 4150
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    
    height2 = tex2.shape[0]
    width2 = tex2.shape[1]
    overlap2 = tex2[0:height2, 0:crop2].copy() #here, crop2 = 1874

    min_loss_fct = loss_fct(overlap1, overlap2)
    best_overlap1 = overlap1
    best_overlap2 = overlap2

    p = 0
    shift_array = []
    shift = -1
    loss_fct_array = []
    
    while (1874 - p > 0) :
        overlap1_sw = tex1[0:height1, crop1 + p:width1].copy()
        overlap2_sw = tex2[0:height2, 0:crop2 - p].copy()

        curr_loss_fct = loss_fct(overlap1_sw, overlap2_sw)
        loss_fct_array.append(curr_loss_fct)
        
        shift = shift + 1
        shift_array.append(shift)
        
        plt.xlabel('Shift number')
        plt.ylabel('Loss function value')
        
        plt.plot(shift_array, loss_fct_array)
        plt.show()

        if (curr_loss_fct < min_loss_fct) :
            min_loss_fct = curr_loss_fct
            best_overlap1 = overlap1_sw
            best_overlap2 = overlap2_sw
        
        p = p + 100
    
    cv2.imshow('C:\\Users\\mocap\\Desktop\\best_overlap00', best_overlap1)
    cv2.imshow('C:\\Users\\mocap\\Desktop\\best_overlap01', best_overlap2)
    #or just min_overlap1
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    # the problem is with the reducing the width and wight 
    # TROUVE UNE AUTRE FACON DE SWEEP


#img1 = Image.open("/Users/syrineenneifer/Desktop/oak_back_00_heightmap_nrm.png")
#overlap_reg1 = img1.crop((4150, 0, 6024, 4022))
#overlap_reg1.save('/Users/syrineenneifer/Desktop/cropped_overlap00.png')

#overlap_reg1_a = cv2.imread('/Users/syrineenneifer/Desktop/cropped_overlap00.png')

img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak_back_00_heightmap_nrm.png")

#img2 = Image.open("/Users/syrineenneifer/Desktop/oak_back_01_heightmap_nrm.png")
#overlap_reg2 = img2.crop((0, 0, 1874, 4022))
#overlap_reg2.save('/Users/syrineenneifer/Desktop/cropped_overlap01.png')

#overlap_reg2_a = cv2.imread('/Users/syrineenneifer/Desktop/cropped_overlap01.png')

img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak_back_00_heightmap_nrm.png")

sweep_and_compare(img1, img2, 4150, 1874)
