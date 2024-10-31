import numpy as np
from PIL import Image
import os
import copy
import sys

# Given radius, number of neighbors, and the pixel, returns the neighbor pixels 
# İt would be interesting to run the program for different radius and number of neighbors 
# for different pixels of the same image; with a strategy to select the varying parameters for
# the selected radius and number of neighbors
def lbp_descriptor(radius, neighbors, center_pixel, height, width):
    x, y = center_pixel

    corners = [(0,0),(width,0),(0,-height),(width,-height)]
    edges = [(x,0),(x,-height),(0,y),(width,y)]
    # Decide whether the center_pixel is on one of the edges (given the radius) or on one of the corners.
    if (x, y) in corners:  # for all radiuses, neighbors is 3
        if (x, y) == (0, 0):
            pixel_list = [(x + radius, y), (x + radius, y + radius), (x, y + radius)]
            return pixel_list
        elif (x, y) == (width, 0):
            pixel_list = [(x - radius, y), (x - radius, y + radius), (x, y + radius)]
            return pixel_list
        elif (x, y) == (0, -height):
            pixel_list = [(x + radius, y), (x + radius, y - radius), (x, y - radius)]
            return pixel_list
        elif (x, y) == (width, -height):
            pixel_list = [(x - radius, y), (x - radius, y - radius), (x, y - radius)]
            return pixel_list
    elif (x, y) in edges: # for all radiuses, neighbors is 5	
        if (x,y) == (x, 0) & x != 0 & x >= radius & x <= width - radius: # upper edge                                                                                                                     -:  # upper edge
            pixel_list = [(x-radius, 0), (x - radius,-radius), (x, -radius), (x+radius, -radius), (x+radius, 0)]
            return pixel_list
        elif (x,y) == (x, -height) & x != 0 & x >= radius & x <= width - radius  :  # bottom edge
            pixel_list = [(x-radius, -height), (x - radius,-height+radius), (x, -height+radius), (x+radius, -height+radius), (x+radius, -height)]
            return pixel_list
        elif (x,y) == (0, y) & y != 0 & y >= radius & y <= height - radius:   # left edge
            pixel_list = [(0, y+radius), (x+radius, y+radius), (x+radius, y), (x+radius, y+radius), (0, y+radius)]
            return pixel_list
        elif (x,y) == (width, y) & y != 0 & y >= radius & y <= height - radius:  # right edge
            pixel_list = [(width, y+radius), (x-radius, y+radius), (x-radius, y), (x-radius, y+radius), (width, y+radius)]
            return pixel_list
    elif  radius == 1 and neighbors == 8: # can be reduced to one line but for now it is kept as is
            pixel_list = [(x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]
            return pixel_list    
    elif radius == 2 and neighbors == 8:
        pixel_list = [(x+2,y), (x+2,y+2), (x,y+2), (x-2,y+1), (x-2,y), (x-2,y-2), (x,y-2), (x+2,y-2)]
        return pixel_list
    elif radius == 3 and neighbors == 8:
        pixel_list = [(x+3,y), (x+3,y+3), (x,y+3), (x-3,y+3), (x-3,y), (x-3,y-3), (x,y-3), (x+3,y-3)]
        return pixel_list
    elif radius == 4 and neighbors == 8:
        pixel_list = [(x+4,y), (x+4,y+4), (x,y+4), (x-4,y+4), (x-4,y), (x-4,y-4), (x,y-4), (x+4,y-4)]
        return pixel_list
    elif radius == 5 and neighbors == 8:
        pixel_list = [(x+5,y), (x+5,y+5), (x,y+5), (x-5,y+5), (x-5,y), (x-5,y-5), (x,y-5), (x+5,y-5)]
        return pixel_list
    elif radius == 6 and neighbors == 8:
        pixel_list = [(x+6,y), (x+6,y+6), (x,y+6), (x-6,y+6), (x-6,y), (x-6,y-6), (x,y-6), (x+6,y-6)]
        return pixel_list
    elif radius == 7 and neighbors == 16:
        pixel_list = [(x+radius, y), (x, y+radius), (x-radius, y), (x, y-radius),
                       (x+radius, y+2), (x+radius, y+3), (x+radius-1, y+2), (x+radius-1, y+3),
                       (x+radius-2, y+4), (x+radius-2, y+5), (x+radius-3, y+4), (x+radius-3, y+5),
                       (x+radius-4, y+6), (x+radius-4, y+7), (x+radius-5, y+6), (x+radius-5, y+7),
                       (x-2, y+radius), (x-2, y+radius-1), (x-3, y+radius), (x-4, y+radius-1),
                       (x-5, y+radius-2), (x-5, y+radius-3), (x-6, y+radius-2), (x-6, y+radius-3),
                       (x-7, y+radius-4), (x-7, y+radius-5), (x-8, y+radius-4), (x-8, y+radius-5),
                       (x-radius, y-2), (x-radius, y-3), (x-radius+1, y-2), (x-radius+1, y-3),
                       (x-radius+2, y-4), (x-radius+2, y-5), (x-radius+3, y-4), (x-radius+3, y-5),
                       (x-radius+4, y-6), (x-radius+4, y-7), (x-radius+5, y-6), (x-radius+5, y-7),
                       (x+2, y-radius), (x+2, y-radius+1), (x+3, y-radius), (x+3, y-radius+1),
                       (x+4, y-radius+2), (x+4, y-radius+3), (x+5, y-radius+2), (x+5, y-radius+3),
                       (x+6, y-radius+4), (x+6, y-radius+5), (x+7, y-radius+4), (x+7, y-radius+5)]
        return pixel_list
    
    
# this function populates the mslbp_to_grayCode from the win_template 
# should be updated to encode the pixels obtained from aco of img_array
def win_template_to_grayCode(img_array, win_template, mslbp_to_grayCode):
    # Get image dimensions
    height, width = img_array.shape
    # index i corresponds to i+1 radius
    radius_gray_coding = [0, 1, 3, 2, 5]
    # iterate over the win_template and generate mslbp_to_grayCode 
    for i in range(0, height - 1):
            for j in range(0, width - 1):
                mslbp_to_grayCode[i][j] = radius_gray_coding[(win_template[i][j][0][0] - 1)]
    # return the mslbp_matrix
    return mslbp_to_grayCode

def compute_mslbp(img_array, win_template, mslbp_template):
    # Initialize current_template
    current_template = [[((i, j), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]
    # Define key value pairs of radius and neighbs
    RADIUS_NEIGHBS = [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 16)]
    radius_neighbs = RADIUS_NEIGHBS
    # Get image dimensions
    height, width = img_array.shape
    # initialize the current_template with the radius and neighbors set to the fixed values
    current_template = [[((radius_neighbs[0][0], radius_neighbs[0][1]), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]
    # initialize the win_template with the radius and neighbors set to the fixed values
    win_template = [[((radius_neighbs[0][0], radius_neighbs[0][1]), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]

    # Iterate over the radius_neighbs
    for radius, neighbors in radius_neighbs[:5]:
        # iterate over pixels determined by the mslbp_template; ignore the edges for now
        for i, j in mslbp_template:
            # after each iteration temp_lbp should be zeroed out
            temp_lbp = 0
            # each computation of the lbp_descriptor will be compared and assigned to the winning template
            # Get the pixel value
            pixel = (i, j)
            # Get the dimensions of the maslbp_template
            # height, width = mslbp_template.shape
            # Get the neighborhood pixels
            neighborhood_pixels = lbp_descriptor(radius, neighbors, pixel, height, width)
            # Compute the linear binary pattern descriptor over the neighborhood pixels
            for k in range(len(neighborhood_pixels)):
                nx, ny = neighborhood_pixels[k]
                if 0 <= nx < width and 0 <= ny < height and img_array[i, j] <= img_array[nx, ny]:
                    temp_lbp = temp_lbp + 2**k
                else:
                    temp_lbp = temp_lbp + 0
                # contains elements from each template; will be used to choose max as winning element
                # the corresponding radius and neighbors is stored in the radius_neighbs list
                current_template[i][j] = ((radius, neighbors), temp_lbp)
                win_template[i][j] = max(current_template[i][j], win_template[i][j], key=lambda x: x[1])
    return win_template

def main():
    # Directory containing the image
    image_directory = r'C:\Users\ozgul\Documents\GitHub\Palm_Veins_Identity\images'
    image_filename = 'test.png'
    # Full path to the image
    image_path = os.path.join(image_directory, image_filename)
    # Read the image
    img = Image.open(image_path)    
    # Convert to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    # Convert image to numpy array as type float
    img_array = np.array(img, dtype=np.float32)  
    # Get image dimensions
    height, width = img_array.shape 
    aco_image_filename = 'aco_image.bmp'
    aco_image_path = os.path.join(image_directory, aco_image_filename)
    aco_img = Image.open(aco_image_path)
    # Convert to grayscale if it's not already
    if aco_img.mode != 'L':
        aco_img = aco_img.convert('L')
    # Convert image to numpy array; row, column should be accessed as y, x, since this is a numpy array
    aco_img_array = np.array(aco_img, dtype=np.uint8)
    # Extract black pixels (value 0)
    black_pixels = [(x, y) for y in range(aco_img_array.shape[0]) for x in range(aco_img_array.shape[1]) if aco_img_array[y, x] == 0]
    # identifies which pixels to iterate to compute the winning template from the original image
    mslbp_template = black_pixels
    # Initialize win_template
    win_template = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=[('x', 'i8'), ('y', 'i8'), ('z', 'i8')])
    # Create a copy to send to the function
    win_template_copy = copy.deepcopy(win_template)
    # Create the winning template; is a numpy array; each element is a tuple of (radius, neighbors) and the lbp value
    win_template = compute_mslbp(img_array, win_template_copy, mslbp_template)
    # Initialize and populate from the image the multi scale local binary pattern matrix
    mslbp_to_grayCode = np.zeros_like(img_array)
    # Create a copy to send to the function
    mslbp_to_grayCode_copy = copy.deepcopy(mslbp_to_grayCode)
    mslbp_to_grayCode = win_template_to_grayCode(img_array, win_template, mslbp_to_grayCode_copy)

    # Convert matrix to an image
    mslbp_to_grayCode_image = Image.fromarray(mslbp_to_grayCode, mode='L')
    # Save the image
    output_image_path = os.path.join(image_directory, 'mslbp_to_grayCode_image.png')
    mslbp_to_grayCode_image.save(output_image_path)
    print(f"Multi-scale local binary pattern encoded with Gray Code saved to {output_image_path}")
    # Write the matrix to a file
    output_path = os.path.join(image_directory, 'mslbp_to_grayCode')
    np.save(output_path, mslbp_to_grayCode)
    print(f"Multi-scale local binary pattern matrix saved to {output_path}")
if __name__ == "__main__":
    main()