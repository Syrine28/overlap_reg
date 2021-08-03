import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
import cma

def normal_vector(pixels) :
    return np.multiply(pixels, 2) - np.ones(3)
    #2 * pixels - 1

def mean_normals(img) :
    normals = []
    for i in range(0, img.shape[0]) :
        for j in range(0, img.shape[1]) :
            pixel = img[i, j]/255
            normals.append(normal_vector(pixel))
    return np.mean(normals)

def convert_to_array_hor_al(crop, img) :
    col_pixel = []
    for i in range(0, img.shape[0]):
        pixel = img[i, crop]/255
        col_pixel.append(pixel)
    return col_pixel 
    

def convert_to_array_ver_al(crop, img) :
    row_pixel = []
    for i in range(0, img.shape[1]):
        pixel = img[crop, i]/255
        row_pixel.append(pixel)
    return row_pixel 

#SECOND METHOD OF ALIGNING

def cma_hor(img1, img2, crop1) :

    width = img1.shape[1]

    def f(x) :
        
        offset = x[0] + crop1
        
        for i in range(int(offset), width) :
                       
            n1 = normal_vector(convert_to_array_hor_al(i, img1))
            n2 = normal_vector(convert_to_array_hor_al(0, img2))
            error = np.linalg.norm(n1 - n2)
                        
            return error
    
    sigma0 = 0.15 * width
    fc_min = cma.fmin(f, [0, 0], sigma0)
    
    print(crop1 + int(fc_min[0][0]))
    return (crop1 + int(fc_min[0][0]))

    
def sweep_and_compare_hor(img1, img2, crop1)  :
    
    results = []
    shifts = []
    shift = -1  
    
    for i in range(crop1, img1.shape[1]) :
        
        n1 = normal_vector(convert_to_array_hor_al(i, img1))
        n2 = normal_vector(convert_to_array_hor_al(0, img2))
        
        #FIRST METHOD :
        error = np.linalg.norm(n1 - n2)
       
        # SECOND METHOD :
        #error = 1 - np.dot(n1, n2)
                
        results.append(error)
        
        shift += 1
        shifts.append(shift)

    offset = np.argmin(results) + crop1
    print(offset)
    plt.xlabel('Shift number')
    plt.ylabel('Loss function value for horizontal alignment')
        
    plt.plot(shifts, results)
    plt.show()
        
    return offset
        
def align_images_hor(tex1, tex2, offset_hor) :
    
    height1 = tex1.shape[0]
    tex1_wo_overlap = tex1[0:height1, 0:offset_hor].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=1)
    
    return final_img

def sweep_and_compare_ver(img1, img2, crop1) :
    
    results = []
    shifts = []
    shift = -1
    
    for i in range(crop1, img1.shape[0]) :
        
        n1 = normal_vector(convert_to_array_ver_al(i, img1))
        n2 = normal_vector(convert_to_array_ver_al(0, img2))
        
        # FIRST METHOD : 
        error = np.linalg.norm(n1 - n2) 
        
        # SECOND METHOD :
        #error = 1 - np.dot(n1, n2)
            
        results.append(error)
        
        shift += 1
        shifts.append(shift)
        
    offset = np.argmin(results) + crop1
    
    plt.xlabel('Shift number')
    plt.ylabel('Loss function value for vertical alignment')
        
    plt.plot(shifts, results)
    plt.show()
        
    return offset

def align_images_ver(tex1, tex2, offset_ver) :

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



# scaled pvc

img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_00_heightmap_nrm_scaled.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_01_heightmap_nrm_scaled.png")

img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_02_heightmap_nrm_scaled.png")

img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_03_heightmap_nrm_scaled.png")

img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_04_heightmap_nrm_scaled.png")

img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_05_heightmap_nrm_scaled.png")

img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_06_heightmap_nrm_scaled.png")

img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_07_heightmap_nrm_scaled.png")

img9 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_08_heightmap_nrm_scaled.png")
 
img10 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_09_heightmap_nrm_scaled.png")

img11 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_10_heightmap_nrm_scaled.png")

img12 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_11_heightmap_nrm_scaled.png")

img13 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_12_heightmap_nrm_scaled.png")

img14 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_13_heightmap_nrm_scaled.png")

img15 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_14_heightmap_nrm_scaled.png")


# scaled oak
"""
img1 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_00_heightmap_nrm_scaled.png")
 
img2 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_01_heightmap_nrm_scaled.png")
img3 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_02_heightmap_nrm_scaled.png")
img4 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_03_heightmap_nrm_scaled.png")
img5 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_04_heightmap_nrm_scaled.png")
img6 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_05_heightmap_nrm_scaled.png")
img7 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_06_heightmap_nrm_scaled.png")
img8 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_07_heightmap_nrm_scaled.png")
img9 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_08_heightmap_nrm_scaled.png")
 
img10 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_09_heightmap_nrm_scaled.png")
img11 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_10_heightmap_nrm_scaled.png")
img12 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_11_heightmap_nrm_scaled.png")
img13 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_12_heightmap_nrm_scaled.png")
img14 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_13_heightmap_nrm_scaled.png")
img15 = cv2.imread("C:\\Users\\mocap\\Desktop\\oak\\oak_14_heightmap_nrm_scaled.png")
"""

#THIS IS FOR SECOND METHOD

#THIS IS FOR NON-SCALED IMAGES
"""
offset_hor1 = sweep_and_compare_hor(img1, img2, 4700) 
offset_hor2 = sweep_and_compare_hor(img2, img3, 4700)
offset_hor3 = sweep_and_compare_hor(img3, img4, 4700)
offset_hor4 = sweep_and_compare_hor(img4, img5, 4700)
offset_hor5 = sweep_and_compare_hor(img5, img6, 4700)
offset_hor6 = sweep_and_compare_hor(img6, img7, 4700)

offset_ver1 = sweep_and_compare_ver(img1, img8, 2000)
"""
#THIS IS FOR SCALED IMAGES
"""
offset_hor1 = sweep_and_compare_hor(img1, img2, 2000)
offset_hor2 = sweep_and_compare_hor(img2, img3, 2000)
offset_hor3 = sweep_and_compare_hor(img3, img4, 2000)
offset_hor4 = sweep_and_compare_hor(img4, img5, 2000)
offset_hor5 = sweep_and_compare_hor(img5, img6, 2000)
offset_hor6 = sweep_and_compare_hor(img6, img7, 2000)

offset_ver1 = sweep_and_compare_ver(img1, img8, 1000)
offset_ver2 = sweep_and_compare_ver(img2, img9, 1000)
offset_ver3 = sweep_and_compare_ver(img3, img10, 1000)
offset_ver4 = sweep_and_compare_ver(img4, img11, 1000)
offset_ver5 = sweep_and_compare_ver(img5, img12, 1000)
offset_ver6 = sweep_and_compare_ver(img6, img13, 1000)
offset_ver7 = sweep_and_compare_ver(img7, img14, 1000)


#Alignment right and down

img_right1, img_down1 = align_right_and_down(img1, img2, img8, offset_hor1, offset_ver1)
img_right2, img_down2 = align_right_and_down(img2, img3, img9, offset_hor2, offset_ver2)
img_right3, img_down3 = align_right_and_down(img3, img4, img10, offset_hor3, offset_ver3)
img_right4, img_down4 = align_right_and_down(img4, img5, img11, offset_hor4, offset_ver4)
img_right5, img_down5 = align_right_and_down(img5, img6, img12, offset_hor5, offset_ver5)
img_right6, img_down6 = align_right_and_down(img6, img7, img13, offset_hor6, offset_ver6)
img_down7 = align_images_ver(img7, img14, offset_ver7)

cv2.imshow("pvc_down7", img_down7)
cv2.imwrite("pvc_down7.png", img_down7)
cv2.waitKey()

cv2.imshow("pvc_right1", img_right1)
cv2.imwrite("pvc_right1.png", img_right1)
cv2.waitKey()
"""
#Alignment of one row
"""
final_img = align_row(img_right1, img_right2, img_right3, img_right4, img_right5, img_right6, offset_hor2, offset_hor3, offset_hor4, offset_hor5, offset_hor6)

cv2.imshow("final_oak_scaled", final_img)
cv2.imwrite("final_oak_scaled.png", final_img)
cv2.waitKey()
"""


#Using CMA 

offset_hor1 = sweep_and_compare_hor(img1, img2, 2000)

offset_hor1 = sweep_and_compare_hor(img1, img2, 2000) 

offset_hor1_cma = cma_hor(img1, img2, 2000)

img_right1_cma = align_images_hor(img1, img2, offset_hor1_cma)

cv2.imshow("pvc_right1_cma", img_right1_cma)
cv2.imwrite("pvc_right1_cma.png", img_right1_cma)
cv2.waitKey()
