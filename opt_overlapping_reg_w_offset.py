import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
import cma

def normal_vector(pixels) :
    #2 * pixels - 1
    return np.multiply(pixels, 2) - np.ones(3) 

#Returns the mean of the normals of an image
def mean_normals(img) :
    normals = []
    for i in range(0, img.shape[0]) :
        for j in range(0, img.shape[1]) :
            pixel = img[i, j]/255
            normals.append(normal_vector(pixel))
    return np.mean(normals)

#Converts an image into columns of pixels
def convert_to_array_hor_al(crop, img) :
    col_pixel = []
    for i in range(0, img.shape[0]):
        pixel = img[i, crop]/255
        col_pixel.append(pixel)
    return col_pixel 
    
#Converts an image into rows of pixels
def convert_to_array_ver_al(crop, img) :
    row_pixel = []
    for i in range(0, img.shape[1]):
        pixel = img[crop, i]/255
        row_pixel.append(pixel)
    return row_pixel 

#CMA alignment
def cma_hor(img1, img2, crop1) :

    width = img1.shape[1]

    def f(x) : #objective function
        
        offset = x[0] + crop1
        
        for i in range(int(offset), width) :
                       
            n1 = normal_vector(convert_to_array_hor_al(i, img1))
            n2 = normal_vector(convert_to_array_hor_al(0, img2))
            error = np.linalg.norm(n1 - n2)
                        
            return error
    
    sigma0 = 0.15 * width
    fc_min = cma.fmin(f, [0, 0], sigma0, noise_handler=cma.NoiseHandler(6), incpopsize=6)
    
    return (crop1 + int(fc_min[0][0]))

def cma_ver(img1, img2, crop1) :

    length = img1.shape[0]

    def f(x) : #objective function
        
        offset = x[0] + crop1
        
        for i in range(int(offset), length) :
                       
            n1 = normal_vector(convert_to_array_ver_al(i, img1))
            n2 = normal_vector(convert_to_array_ver_al(0, img2))
            error = np.linalg.norm(n1 - n2)
                        
            return error
    
    sigma0 = 0.06 * length
    fc_min = cma.fmin(f, [0, 0], sigma0, noise_handler=cma.NoiseHandler(6), incpopsize=6)
    
    return (crop1 + int(fc_min[0][0]))

#Returns the optimal horizontal offset based on the minimum error computed with the normals
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
    plt.xlabel('Shift number')
    plt.ylabel('Loss function value for horizontal alignment')
        
    plt.plot(shifts, results)
    plt.show()
        
    return offset
        
#Moves the second image on top of the first one horizontally
def align_images_hor(tex1, tex2, offset_hor) :
    
    height1 = tex1.shape[0]
    tex1_wo_overlap = tex1[0:height1, 0:offset_hor].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=1)
    
    return final_img

#Returns the optimal vertical offset based on the minimum error computed with the normals
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

#Moves the second image on top of the first one vertically
def align_images_ver(tex1, tex2, offset_ver) :

    width1 = tex1.shape[1]
    tex1_wo_overlap = tex1[0:offset_ver, 0:width1].copy()
    
    final_img = np.concatenate((tex1_wo_overlap, tex2), axis=0)
     
    return final_img

#Aligns an image to the one to its right and the one below it
def align_right_and_down(img, img_right, img_down, offset_hor, offset_ver) :
    #blended_right = cv2.addWeighted(overlap_right1, 0.75, overlap_right2, 0.25, 0.0)
    #blended_down = cv2.addWeighted(overlap_down1, 0.75, overlap_down2, 0.25, 0.0)
    
    final_right = align_images_hor(img, img_right, offset_hor)
    final_down = align_images_ver(img, img_down, offset_ver)
    return final_right, final_down   
    
#Aligns a whole row
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
img16 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_15_heightmap_nrm_scaled.png") 
img17 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_16_heightmap_nrm_scaled.png")
img18 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_17_heightmap_nrm_scaled.png")
img19 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_18_heightmap_nrm_scaled.png")
img20 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_19_heightmap_nrm_scaled.png")
img21 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_20_heightmap_nrm_scaled.png")
img22 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_21_heightmap_nrm_scaled.png")
img23 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_22_heightmap_nrm_scaled.png")
img24 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_23_heightmap_nrm_scaled.png")
img25 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_24_heightmap_nrm_scaled.png")
img26 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_25_heightmap_nrm_scaled.png")
img27 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_26_heightmap_nrm_scaled.png")
img28 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_27_heightmap_nrm_scaled.png")
img29 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_28_heightmap_nrm_scaled.png")
img30 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_29_heightmap_nrm_scaled.png")
img31 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_30_heightmap_nrm_scaled.png") 
img32 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_31_heightmap_nrm_scaled.png")
img33 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_32_heightmap_nrm_scaled.png")
img34 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_33_heightmap_nrm_scaled.png")
img35 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_34_heightmap_nrm_scaled.png")
img36 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_35_heightmap_nrm_scaled.png")
img37 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_36_heightmap_nrm_scaled.png")
img38 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_37_heightmap_nrm_scaled.png")
img39 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_38_heightmap_nrm_scaled.png") 
img40 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_39_heightmap_nrm_scaled.png")
img41 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_40_heightmap_nrm_scaled.png")
img42 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_41_heightmap_nrm_scaled.png")
img43 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_42_heightmap_nrm_scaled.png")
img44 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_43_heightmap_nrm_scaled.png")
img45 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_44_heightmap_nrm_scaled.png")
img46 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_45_heightmap_nrm_scaled.png") 
img47 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_46_heightmap_nrm_scaled.png")
img48 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_47_heightmap_nrm_scaled.png")
img49 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_48_heightmap_nrm_scaled.png")
img50 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_49_heightmap_nrm_scaled.png")
img51 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_50_heightmap_nrm_scaled.png")
img52 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_51_heightmap_nrm_scaled.png")
img53 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_52_heightmap_nrm_scaled.png")
img54 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_53_heightmap_nrm_scaled.png") 
img55 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_54_heightmap_nrm_scaled.png")
img56 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_55_heightmap_nrm_scaled.png")
img57 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_56_heightmap_nrm_scaled.png")
img58 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_57_heightmap_nrm_scaled.png")
img59 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_58_heightmap_nrm_scaled.png")
img60 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_59_heightmap_nrm_scaled.png")
img61 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_60_heightmap_nrm_scaled.png")
img62 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_61_heightmap_nrm_scaled.png")
img63 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_62_heightmap_nrm_scaled.png")
img64 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_63_heightmap_nrm_scaled.png")
img65 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_64_heightmap_nrm_scaled.png")
img66 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_65_heightmap_nrm_scaled.png")
img67 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_66_heightmap_nrm_scaled.png")
img68 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_67_heightmap_nrm_scaled.png")
img69 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_68_heightmap_nrm_scaled.png")
img70 = cv2.imread("C:\\Users\\mocap\\Desktop\\pvc\\pvc_69_heightmap_nrm_scaled.png")

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

#THIS IS FOR SCALED IMAGES
"""
offset_hor1 = sweep_and_compare_hor(img1, img2, 2000)
offset_hor2 = sweep_and_compare_hor(img2, img3, 2000)
offset_hor3 = sweep_and_compare_hor(img3, img4, 2000)
offset_hor4 = sweep_and_compare_hor(img4, img5, 2000)
offset_hor5 = sweep_and_compare_hor(img5, img6, 2000)
offset_hor6 = sweep_and_compare_hor(img6, img7, 2000)

offset_ver1 = sweep_and_compare_ver(img1, img8, 1500)
offset_ver2 = sweep_and_compare_ver(img2, img9, 1500)
offset_ver3 = sweep_and_compare_ver(img3, img10, 1500)
offset_ver4 = sweep_and_compare_ver(img4, img11, 1500)
offset_ver5 = sweep_and_compare_ver(img5, img12, 1500)
offset_ver6 = sweep_and_compare_ver(img6, img13, 1500)
offset_ver7 = sweep_and_compare_ver(img7, img14, 1500)


#Alignment right and down

img_right1, img_down1 = align_right_and_down(img1, img2, img8, offset_hor1, offset_ver1)
img_right2, img_down2 = align_right_and_down(img2, img3, img9, offset_hor2, offset_ver2)
img_right3, img_down3 = align_right_and_down(img3, img4, img10, offset_hor3, offset_ver3)
img_right4, img_down4 = align_right_and_down(img4, img5, img11, offset_hor4, offset_ver4)
img_right5, img_down5 = align_right_and_down(img5, img6, img12, offset_hor5, offset_ver5)
img_right6, img_down6 = align_right_and_down(img6, img7, img13, offset_hor6, offset_ver6)
img_down7 = align_images_ver(img7, img14, offset_ver7)

cv2.imshow("down7", img_down7)
cv2.imwrite("down7.png", img_down7)
cv2.waitKey()

cv2.imshow("right6", img_right6)
cv2.imwrite("right6.png", img_right6)
cv2.waitKey()
"""
#Alignment of one row

#First row
#final_img = align_row(img_right1, img_right2, img_right3, img_right4, img_right5, img_right6, offset_hor2, offset_hor3, offset_hor4, offset_hor5, offset_hor6)
#Second row
"""
offset_hor8 = sweep_and_compare_hor(img8, img9, 2000)
offset_hor9 = sweep_and_compare_hor(img9, img10, 2000)
offset_hor10 = sweep_and_compare_hor(img10, img11, 2000)
offset_hor11 = sweep_and_compare_hor(img11, img12, 2000)
offset_hor12 = sweep_and_compare_hor(img12, img13, 2000)
offset_hor13 = sweep_and_compare_hor(img13, img14, 2000)


img_right8 = align_images_hor(img8, img9, offset_hor8)
img_right9 = align_images_hor(img9, img10, offset_hor9)
img_right10 = align_images_hor(img10, img11, offset_hor10)
img_right11 = align_images_hor(img11, img12, offset_hor11)
img_right12 = align_images_hor(img12, img13, offset_hor12)
img_right13 = align_images_hor(img13, img14, offset_hor13)

final_img = align_row(img_right8, img_right9, img_right10, img_right11, img_right12, img_right13, offset_hor9, offset_hor10, offset_hor11, offset_hor12, offset_hor13)

cv2.imshow("final_pvc_scaled", final_img)
cv2.imwrite("final_pvc_scaled.png", final_img)
cv2.waitKey()
"""
#Using CMA 
"""
offset_hor = sweep_and_compare_hor(img3, img4, 2000) 

offset_hor_cma = cma_hor(img3, img4, 2800)
"""

offset_ver = sweep_and_compare_ver(img7, img14, 1500)
print(offset_ver)

offset_ver_cma = cma_ver(img7, img14, 1600)
print(offset_ver_cma)

cma_ver = align_images_ver(img7, img14, offset_ver_cma)

cv2.imshow("cma_ver", cma_ver)
cv2.imwrite("cma_ver.png", cma_ver)
cv2.waitKey()

#Printing all offsets using cma
"""
offh1 = cma_hor(img1, img2, 2800)
offh2 = cma_hor(img2, img3, 2800)
offh3 = cma_hor(img3, img4, 2800)
offh4 = cma_hor(img4, img5, 2800)
offh5 = cma_hor(img5, img6, 2800)
offh6 = cma_hor(img6, img7, 2800)

offh8 = cma_hor(img8, img9, 2800)
offh9 = cma_hor(img9, img10, 2800)
offh10 = cma_hor(img10, img11, 2800)
offh11 = cma_hor(img11, img12, 2800)
offh12 = cma_hor(img12, img13, 2800)
offh13 = cma_hor(img13, img14, 2800)

offh15 = cma_hor(img15, img16, 2800)
offh16 = cma_hor(img16, img17, 2800)
offh17 = cma_hor(img17, img18, 2800)
offh18 = cma_hor(img18, img19, 2800)
offh19 = cma_hor(img19, img20, 2800)
offh20 = cma_hor(img20, img21, 2800)

offh22 = cma_hor(img22, img23, 2800)
offh23 = cma_hor(img23, img24, 2800)
offh24 = cma_hor(img24, img25, 2800)
offh25 = cma_hor(img25, img26, 2800)
offh26 = cma_hor(img26, img27, 2800)
offh27 = cma_hor(img27, img28, 2800)

offh29 = cma_hor(img29, img30, 2800)
offh30 = cma_hor(img30, img31, 2800)
offh31 = cma_hor(img31, img32, 2800)
offh32 = cma_hor(img32, img33, 2800)
offh33 = cma_hor(img33, img34, 2800)
offh34 = cma_hor(img34, img35, 2800)

offh36 = cma_hor(img36, img37, 2800)
offh37 = cma_hor(img37, img38, 2800)
offh38 = cma_hor(img38, img39, 2800)
offh39 = cma_hor(img39, img40, 2800)
offh40 = cma_hor(img40, img41, 2800)
offh41 = cma_hor(img41, img42, 2800)

offh43 = cma_hor(img43, img44, 2800)
offh44 = cma_hor(img44, img45, 2800)
offh45 = cma_hor(img45, img46, 2800)
offh46 = cma_hor(img46, img47, 2800)
offh47 = cma_hor(img47, img48, 2800)
offh48 = cma_hor(img48, img49, 2800)

offh50 = cma_hor(img50, img51, 2800)
offh51 = cma_hor(img51, img52, 2800)
offh52 = cma_hor(img52, img53, 2800)
offh53 = cma_hor(img53, img54, 2800)
offh54 = cma_hor(img54, img55, 2800)
offh55 = cma_hor(img55, img56, 2800)

offh57 = cma_hor(img57, img58, 2800)
offh58 = cma_hor(img58, img59, 2800)
offh59 = cma_hor(img59, img60, 2800)
offh60 = cma_hor(img60, img61, 2800)
offh61 = cma_hor(img61, img62, 2800)
offh62 = cma_hor(img62, img63, 2800)

offh64 = cma_hor(img64, img65, 2800)
offh65 = cma_hor(img65, img66, 2800)
offh66 = cma_hor(img66, img67, 2800)
offh67 = cma_hor(img67, img68, 2800)
offh68 = cma_hor(img68, img69, 2800)
offh69 = cma_hor(img69, img70, 2800)
"""
