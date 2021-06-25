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
    #return sum/(len(matrix_ov_reg1)) # len(matrix_ov_reg1) = # of pixels (same as reg2)
   
    # FOR SECOND METHOD    
    return sum


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
            min_p = p
                    
        # This is just to get pics of the sweeped areas
        #wd = os.getcwd()
        #cv2.imshow(wd + "\sweeped_reg", overlap1_sw)
        #cv2.waitKey(0)

        p = p + 50
            
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
        
        plt.xlabel('Shift number')
        plt.ylabel('Loss function value')
        
        plt.plot(shift_array, loss_fct_array)
        plt.show()

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
            
    return crop1 + min_p #or just min_p????

def align_images_hor(tex1, tex2, offset_hor) :
    height1 = tex1.shape[0]
    tex1_wo_overlap = tex1[0:height1, 0:offset_hor].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=1)
     
    return final_img

def sweep_and_compare_ver(tex1, tex2, crop1, crop2) :
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    overlap1 = tex1[0:crop1, 0:width1].copy()
    
    height2 = tex2.shape[0]
    width2 = tex2.shape[1]
    overlap2 = tex2[crop2:height2, 0:width2].copy()
    
    min_loss_fct = loss_fct(overlap1, overlap2)
    best_overlap1 = overlap1
    best_overlap2 = overlap2

    p = 0
    shift_array = []
    shift = -1
    loss_fct_array = []
    min_p = p
    
    while (crop2 - p > 0) :
        overlap1_sw = tex1[0:crop1 - p, 0:width1].copy()
        overlap2_sw = tex2[crop2:height2 - p, 0:width2].copy()
                    
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
            min_p = p
                    
        # This is just to get pics of the sweeped areas
        #cv2.imshow("\sweeped_reg1", overlap1_sw)
        #cv2.waitKey(0)

        #cv2.imshow("\sweeped_reg2", overlap2_sw)
        #cv2.waitKey(0)

        p = p + 5
    print(crop1 - min_p)       
    return crop1 - min_p #or just min_p????

def align_images_ver(tex1, tex2, offset_ver) :
    height1 = tex1.shape[0]
    width1 = tex1.shape[1]
    tex1_wo_overlap = tex1[offset_ver:height1, 0:width1].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=0)
     
    return final_img


def align_right_and_down(img, img_right, img_down, offset_hor, offset_ver) :
    #blended_right = cv2.addWeighted(overlap_right1, 0.75, overlap_right2, 0.25, 0.0)
    #blended_down = cv2.addWeighted(overlap_down1, 0.75, overlap_down2, 0.25, 0.0)
    
    final_right = align_images_hor(img, img_right, offset_hor)
    final_down = align_images_ver(img, img_down, offset_ver)
    return final_right, final_down   
    

def align_row(img1, img2, img3, img4, img5, img6, img7,
              offset1, offset2, offset3, offset4, offset5, offset6) :
    #For a whole row 
     
    #Find overlap between the 2 pics, blend the two same overlaps
    #Remove it from the original scans then concatenate them to the overlap
    partial1 = align_images_hor(img1, img2, offset1)
    partial2 = align_images_hor(partial1, img3, offset2)
    partial3 = align_images_hor(partial2, img4, offset3)
    partial4 = align_images_hor(partial3, img5, offset4)
    partial5 = align_images_hor(partial4, img6, offset5)
    final = align_images_hor(partial5, img7, offset6)

    #Too long???? 
        
    #Then return the last partial = final
    
    return final

wd = os.getcwd()

# acrylic 
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_00_heightmap_nrm.png")

img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\acrylic\\acrylic_06_heightmap_nrm.png")
"""

# pine
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_00_heightmap_nrm.png")

img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\pine\\pine_06_heightmap_nrm.png")
"""

# nylon
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_00_heightmap_nrm.png")

img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\nylon\\nylon_06_heightmap_nrm.png")
"""

# steel
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\steel\\steel_06_heightmap_nrm.png")
"""

# FOR OAK : crop1 = 4600, crop2 = 1424
# FOR ACRYLIC : crop1 = 4800, crop2 = 1224

# rosewood
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\rosewood\\rosewood_06_heightmap_nrm.png")
"""

# aluminium
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_06_heightmap_nrm.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\alu\\alu_07_heightmap_nrm.png")
"""


# pvc

img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_06_heightmap_nrm.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_07_heightmap_nrm.png")


# brass
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_06_heightmap_nrm.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\brass\\brass_07_heightmap_nrm.png")
"""

# copper
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_06_heightmap_nrm.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\copper\\copper_07_heightmap_nrm.png")
"""

# oak
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_00_heightmap_nrm.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_01_heightmap_nrm.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_02_heightmap_nrm.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_03_heightmap_nrm.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_04_heightmap_nrm.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_05_heightmap_nrm.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_06_heightmap_nrm.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_07_heightmap_nrm.png")
"""


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

"""

#THIS IS FOR SECOND METHOD

offset_hor1 = sweep_and_compare_hor(img1, img2, 4700, 1324) 
offset_hor2 = sweep_and_compare_hor(img2, img3, 4700, 1324)
offset_hor3 = sweep_and_compare_hor(img3, img4, 4700, 1324)
offset_hor4 = sweep_and_compare_hor(img4, img5, 4700, 1324)
offset_hor5 = sweep_and_compare_hor(img5, img6, 4700, 1324)
offset_hor6 = sweep_and_compare_hor(img6, img7, 4700, 1324)
"""
final_row = final_row = align_row(img1, img2, img3, img4, img5, img6, img7, offset_hor1,
                                  offset_hor2, offset_hor3, offset_hor4, offset_hor5, offset_hor6)


cv2.imshow("final_row_pvc", final_row)
cv2.imwrite("final_row_pvc.png", final_row)
cv2.waitKey()
"""
offset_ver1 = sweep_and_compare_ver(img1, img8, 2000, 2022)
print(offset_ver1)

img_right, img_down = align_right_and_down(img1, img2, img8, offset_hor1, offset_ver1)
cv2.imshow("pvc_right", img_right)
cv2.imwrite("pvc_right.png", img_right)
cv2.waitKey()

cv2.imshow("pvc_down", img_down)
cv2.imwrite("pvc_down.png", img_down)
cv2.waitKey()
