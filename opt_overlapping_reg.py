import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

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
        
        # FIRST METHOD : 
        #error = np.linalg.norm(n1 - n2) 
        
        # SECOND METHOD :
        error = 1 - np.dot(n1, n2)
        
        # THIRD METHOD :
        #error = mae(n1, n2)
            
        sum += error
    
    # FOR FIRST AND THIRD METHOD
    return sum/(len(matrix_ov_reg1)) # len(matrix_ov_reg1) = # of pixels (same as reg2)
   
    # FOR SECOND METHOD
    #return sum


#FIRST METHOD OF ALIGNING

"""
#ONLY FOR HORIZONTAL OVERLAPPING
def sweep_and_compare(tex1, tex2, crop1, crop2) :
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    overlap1 = tex1[0:height1, crop1:width1].copy()  #here, crop1 = 4150
    
    height2 = tex2.shape[0]
    overlap2 = tex2[0:height2, 0:crop2].copy() #here, crop2 = 1874
    
    min_loss_fct = loss_fct(overlap1, overlap2)
    best_overlap1 = overlap1
    best_overlap2 = overlap2

    p = 0
    shift_array = []
    shift = -1
    loss_fct_array = []
    min_p = p
    
    while (crop1 + p < width1) :
        overlap1_sw = tex1[0:height1, crop1 + p:width1].copy()
        overlap2_sw = tex2[0:height2, p:crop2].copy()
                    
        curr_loss_fct = loss_fct(overlap1_sw, overlap2_sw)
        loss_fct_array.append(curr_loss_fct)
        
        shift = shift + 5
        shift_array.append(shift)

        if (curr_loss_fct < min_loss_fct) :
            min_loss_fct = curr_loss_fct
            best_overlap1 = overlap1_sw
            best_overlap2 = overlap2_sw
            min_p = p
                    
        # This is just to get pics of the sweeped areas
        #wd = os.getcwd()
        #cv2.imshow(wd + "\sweeped_reg", overlap1_sw)
        #cv2.waitKey(0)

        p = p + 5
    plt.xlabel('Shift number')
    plt.ylabel('Loss function value')
        
    plt.plot(shift_array, loss_fct_array)
    plt.show()
            
    return best_overlap1, best_overlap2

def align_images_hor(tex1, tex2, overlap1, overlap2) :
    #We'll blend both overlaps together
    blended = cv2.addWeighted(overlap1, 0.5, overlap2, 0.5, 0.0)
    
    #Now we paste tex1 without the overlap to blended
    width_over = overlap1.shape[1]  # = overlap1.shape[1]
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    tex1_wo_overlap = tex1[0:height1, 0:width1 - width_over].copy()
    
    #axis=1 is to concatenate them horizontally
    partial_img = np.concatenate((tex1_wo_overlap, blended), axis=1)

    #Now paste it to tex2 without the overlap to get the final image
    height2 = tex2.shape[0]
    width2 = tex2.shape[1]
    tex2_wo_overlap = tex2[0:height2, width_over:width2].copy()
    
    final_img = np.concatenate((partial_img, tex2_wo_overlap), axis=1)
     
    return final_img

def align_images_ver(tex1, tex2, overlap1, overlap2) :
    #We'll blend both overlaps together
    blended = cv2.addWeighted(overlap1, 0.75, overlap2, 0.25, 0.0)
    
    #Now we paste tex1 without the overlap to blended
    height_over = overlap1.shape[0]  # = overlap1.shape[0]
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    tex1_wo_overlap = tex1[height_over:height1, 0:width1].copy()

    #axis=0 is to concatenate them vertically
    partial_img = np.concatenate((tex1_wo_overlap, blended), axis=0)

    #Now paste it to tex2 without the overlap to get the final image
    height2 = tex2.shape[0]
    width2 = tex2.shape[1]
    tex2_wo_overlap = tex2[0:height2 - height_over, 0:width2].copy()
    
    final_img = np.concatenate((partial_img, tex2_wo_overlap), axis=0)
     
    return final_img


def align_right_and_down(img, img_right, img_down, overlap_right1, overlap_right2, overlap_down1, overlap_down2) :
    #blended_right = cv2.addWeighted(overlap_right1, 0.75, overlap_right2, 0.25, 0.0)
    #blended_down = cv2.addWeighted(overlap_down1, 0.75, overlap_down2, 0.25, 0.0)
    
    final_right = align_images_hor(img, img_right, overlap_right1, overlap_right2)
    final_down = align_images_ver(img, img_down, overlap_down1, overlap_down2)
    return final_right, final_down  

def align_row(img1, img2, img3, img4, img5, img6, img7,
              overlap1, overlap2_1, overlap2_3, overlap3_2,
              overlap3_4, overlap4_3, overlap4_5, overlap5_4,
              overlap5_6, overlap6_5, overlap6_7, overlap7) :
    #For a whole row 
     
    #Find overlap between the 2 pics, blend the two same overlaps
    #Remove it from the original scans then concatenate them to the overlap
    partial1 = align_images_hor(img1, img2, overlap1, overlap2_1)
    partial2 = align_images_hor(partial1, img3, overlap2_3, overlap3_2)
    partial3 = align_images_hor(partial2, img4, overlap3_4, overlap4_3)
    partial4 = align_images_hor(partial3, img5, overlap4_5, overlap5_4)
    partial5 = align_images_hor(partial4, img6, overlap5_6, overlap6_5)
    final = align_images_hor(partial5, img7, overlap6_7, overlap7)

    #Too long???? 
        
    #Then return the last partial = final
    
    return final
"""

#SECOND METHOD OF ALIGNING

def sweep_and_compare_hor(tex1, tex2, crop1, crop2) :
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    overlap1 = tex1[0:height1, crop1:width1].copy()
    
    height2 = tex2.shape[0] 
    width2 = tex2.shape[1]
    overlap2 = tex2[0:height2, 0:crop2].copy()
    
    min_loss_fct = loss_fct(overlap1, overlap2)
    best_overlap1 = overlap1
    best_overlap2 = overlap2

    p = 0
    shift_array = []
    shift = -1
    loss_fct_array = []
    min_p = p
    
    while (crop1 + p < width1) :
        overlap1_sw = tex1[0:height1, crop1 + p:width1].copy()
        overlap2_sw = tex2[0:height2, p:crop2].copy()
                    
        curr_loss_fct = loss_fct(overlap1_sw, overlap2_sw)
        loss_fct_array.append(curr_loss_fct)
        
        shift = shift + 1
        shift_array.append(shift)
       
        if (curr_loss_fct < min_loss_fct) :
            min_loss_fct = curr_loss_fct
            best_overlap1 = overlap1_sw
            best_overlap2 = overlap2_sw
            min_p = p
                    
        p = p + 1
        
    plt.xlabel('Shift number')
    plt.ylabel('Loss function value for horizontal alignment')
        
    plt.plot(shift_array, loss_fct_array)
    plt.show()
            
    return crop1 + min_p #or just min_p????

def align_images_hor(tex1, tex2, offset_hor) :
    height1 = tex1.shape[0]
    tex1_wo_overlap = tex1[0:height1, 0:offset_hor].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=1)
     
    return final_img

def sweep_and_compare_ver(tex1, tex2, crop1, crop2) :
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    overlap1 = tex1[crop1:height1, 0:width1].copy()
    
    height2 = tex2.shape[0] 
    width2 = tex2.shape[1]
    overlap2 = tex2[0:crop2, 0:width2].copy()
    
    min_loss_fct = loss_fct(overlap1, overlap2)
    best_overlap1 = overlap1
    best_overlap2 = overlap2

    p = 0
    shift_array = []
    shift = -1
    loss_fct_array = []
    min_p = p
    
    while (crop1 + p < height1 - 50) :
        overlap1_sw = tex1[crop1 + p:height1, 0:width1].copy()
        overlap2_sw = tex2[p:crop2, 0:width2].copy()
                    
        curr_loss_fct = loss_fct(overlap1_sw, overlap2_sw)
        loss_fct_array.append(curr_loss_fct)
        
        shift = shift + 1
        shift_array.append(shift)
       
        if (curr_loss_fct < min_loss_fct) :
            min_loss_fct = curr_loss_fct
            best_overlap1 = overlap1_sw
            best_overlap2 = overlap2_sw
            min_p = p
                    
        p = p + 1
   
    plt.xlabel('Shift number')
    plt.ylabel('Loss function value for vertical alignment')
        
    plt.plot(shift_array, loss_fct_array)
    plt.show()
    
    return crop1 + min_p #or just min_p????

def align_images_ver(tex1, tex2, offset_ver) :
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    tex1_wo_overlap = tex1[0:offset_ver, 0:width1].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=0)
     
    return final_img


def align_right_and_down(img, img_right, img_down, offset_hor, offset_ver) :
    #blended_right = cv2.addWeighted(overlap_right1, 0.75, overlap_right2, 0.25, 0.0)
    #blended_down = cv2.addWeighted(overlap_down1, 0.75, overlap_down2, 0.25, 0.0)
    
    final_right = align_images_hor(img, img_right, offset_hor)
    final_down = align_images_ver(img, img_down, offset_ver)
    return final_right, final_down   
    

"""
def align_row(img1, img2, img3, img4, img5, img6, img7,
              offset1, offset2, offset3, offset4, offset5, offset6) :
    #For a whole row 
     
    #Find overlap between the 2 pics, blend the two same overlaps
    #Remove it from the original scans then concatenate them to the overlap
    partial1 = align_images_hor(img1, img2, offset2)
    partial2 = align_images_hor(partial1, img3, offset3)
    partial3 = align_images_hor(partial2, img4, offset4)
    partial4 = align_images_hor(partial3, img5, offset5)
    partial5 = align_images_hor(partial4, img6, offset6)
    final = align_images_hor(partial5, img7, offset6)

    #Too long???? 
        
    #Then return the last partial = final
    
    return final

"""
def align_row(img1, img2, img3, img4, img5, img6,
               offset2, offset3, offset4, offset5, offset6) :
   
    height = img1.shape[0]
    img1_wo_overlap = img1[0:height, 0:offset2].copy()
    img2_wo_overlap = img2[0:height, 0:offset3].copy()
    img3_wo_overlap = img3[0:height, 0:offset4].copy()
    img4_wo_overlap = img4[0:height, 0:offset5].copy()
    img5_wo_overlap = img5[0:height, 0:offset6].copy()

    partial1 = np.concatenate((img1_wo_overlap, img2_wo_overlap), axis=1)
    partial2 = np.concatenate((partial1, img3_wo_overlap), axis=1)
    partial3 = np.concatenate((partial2, img4_wo_overlap), axis=1)
    partial4 = np.concatenate((partial3, img5_wo_overlap), axis=1)
    final = np.concatenate((partial4, img6), axis=1)

    return final


# scaled acrylic

img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_00_heightmap_nrm_scaled.png")

img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_01_heightmap_nrm_scaled.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_02_heightmap_nrm_scaled.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_03_heightmap_nrm_scaled.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_04_heightmap_nrm_scaled.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_05_heightmap_nrm_scaled.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_06_heightmap_nrm_scaled.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_07_heightmap_nrm_scaled.png")

img9 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_08_heightmap_nrm_scaled.png")

img10 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_09_heightmap_nrm_scaled.png")

img11 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_10_heightmap_nrm_scaled.png")

img12 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_11_heightmap_nrm_scaled.png")

img13 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_12_heightmap_nrm_scaled.png")

img14 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_13_heightmap_nrm_scaled.png")

img15 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_14_heightmap_nrm_scaled.png")


#THIS IS FOR FIRST METHOD
"""
opt_overlap1, opt_overlap2_1 = sweep_and_compare(img1, img2, 4700, 1324) 
opt_overlap2_3, opt_overlap3_2 = sweep_and_compare(img2, img3, 4700, 1324)
opt_overlap3_4, opt_overlap4_3 = sweep_and_compare(img3, img4, 4700, 1324)
opt_overlap4_5, opt_overlap5_4 = sweep_and_compare(img4, img5, 4700, 1324)
opt_overlap5_6, opt_overlap6_5 = sweep_and_compare(img5, img6, 4700, 1324)
opt_overlap6_7, opt_overlap7 = sweep_and_compare(img6, img7, 4700, 1324)

final_row = align_row(img1, img2, img3, img4, img5, img6, img7, 
               opt_overlap1, opt_overlap2_1, opt_overlap2_3, opt_overlap3_2,
               opt_overlap3_4, opt_overlap4_3, opt_overlap4_5, opt_overlap5_4,
               opt_overlap5_6, opt_overlap6_5, opt_overlap6_7, opt_overlap7)

cv2.imshow("final_row_brass", final_row)
cv2.imwrite("final_row_brass.png", final_row)
cv2.waitKey()
"""

#THIS IS FOR SECOND METHOD

#THIS IS FOR NON-SCALED IMAGES
"""
offset_hor1 = sweep_and_compare_hor(img1, img2, 4700, 1324) 
offset_hor2 = sweep_and_compare_hor(img2, img3, 4700, 1324)
offset_hor3 = sweep_and_compare_hor(img3, img4, 4700, 1324)
offset_hor4 = sweep_and_compare_hor(img4, img5, 4700, 1324)
offset_hor5 = sweep_and_compare_hor(img5, img6, 4700, 1324)
offset_hor6 = sweep_and_compare_hor(img6, img7, 4700, 1324)
"""
#THIS IS FOR SCALED IMAGES

offset_hor1 = sweep_and_compare_hor(img1, img2, 2350, 662) 
offset_hor2 = sweep_and_compare_hor(img2, img3, 2350, 662)
offset_hor3 = sweep_and_compare_hor(img3, img4, 2350, 662)
offset_hor4 = sweep_and_compare_hor(img4, img5, 2350, 662)
offset_hor5 = sweep_and_compare_hor(img5, img6, 2350, 662)
offset_hor6 = sweep_and_compare_hor(img6, img7, 2350, 662)

#offset_hor1 = sweep_and_compare_hor(img1, img2, 2350, 662) 

"""
img = align_images_hor(img2, img3, offset_hor2)
cv2.imshow("2_3", img)
cv2.imwrite("2_3.png", img)
cv2.waitKey()
"""
#THIS IS FOR NON_SCALED IMAGES

#offset_ver1 = sweep_and_compare_ver(img1, img8, 2000, 2022)

#THIS IS FOR SCALED IMAGES

offset_ver1 = sweep_and_compare_ver(img1, img8, 1000, 1011)
offset_ver2 = sweep_and_compare_ver(img2, img9, 1000, 1011)
offset_ver3 = sweep_and_compare_ver(img3, img10, 1000, 1011)
offset_ver4 = sweep_and_compare_ver(img4, img11, 1000, 1011)
offset_ver5 = sweep_and_compare_ver(img5, img12, 1000, 1011)
offset_ver6 = sweep_and_compare_ver(img6, img13, 1000, 1011)
offset_ver7 = sweep_and_compare_ver(img7, img14, 1000, 1011)

img_right1, img_down1 = align_right_and_down(img1, img2, img8, offset_hor1, offset_ver1)
img_right2, img_down2 = align_right_and_down(img2, img3, img9, offset_hor2, offset_ver2)
img_right3, img_down3 = align_right_and_down(img3, img4, img10, offset_hor3, offset_ver3)
img_right4, img_down4 = align_right_and_down(img4, img5, img11, offset_hor4, offset_ver4)
img_right5, img_down5 = align_right_and_down(img5, img6, img12, offset_hor5, offset_ver5)
img_right6, img_down6 = align_right_and_down(img6, img7, img13, offset_hor6, offset_ver6)
img_down7 = align_images_ver(img7, img14, offset_ver7)


#alignment of one row

final_img = align_row(img_right1, img_right2, img_right3, img_right4, img_right5, img_right6, offset_hor2, offset_hor3, offset_hor4, offset_hor5, offset_hor6)
"""
cv2.imshow("acry_right", img_right3)
cv2.imwrite("acry_right.png", img_right3)
cv2.waitKey()

cv2.imshow("acry_down", img_down7)
cv2.imwrite("acry_down.png", img_down7)
cv2.waitKey()
"""
cv2.imshow("final", final_img)
cv2.imwrite("final.png", final_img)
cv2.waitKey()
