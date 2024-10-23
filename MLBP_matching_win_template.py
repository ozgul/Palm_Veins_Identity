import numpy as np
from PIL import Image
import os
import copy

# Given radius, number of neighbors, and the pixel, returns the neighbor pixels 
# İt would be interesting to run the program for different radius and number of neighbors 
# for different pixels of the same image; with a strategy to select the varying parameters for
# the selected radius and number of neighbors
def lbp_descriptor(radius, neighbors, center_pixel):
    x, y = center_pixel
    if radius == 1 and neighbors == 8:
        pixel_list = [(x+1,y), (x+1,y+1), (x,y+1), (x-1,y+1), (x-1,y), (x-1,y-1), (x,y-1), (x+1,y-1)]
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
    
    # Write another list for the half circle and the quarter circle use existing lists

# converts a matrix to an image and prints it
def matrix_to_image(mslbp_matrix, output_path):
    # Normalize the mslbp_matrix to the range [0, 255]
    # If the radius is 16 then normalize to [0, 65535]
    mslbp_matrix_normalized = (255 * (mslbp_matrix - np.min(mslbp_matrix)) / (np.max(mslbp_matrix) - np.min(mslbp_matrix))).astype(np.uint8)
    
    # Convert the normalized matrix to an image
    mslbp_image = Image.fromarray(mslbp_matrix_normalized)
    # Save the image
    output_image_path = os.path.join(output_path, 'mslbp_image.png')
    mslbp_image.save(output_image_path)
    print(f"Multi-scale local binary pattern image saved to {output_image_path}")
    # Write the mslbp_matrix to a file
    output_path = os.path.join(output_path, 'mslbp.npy')
    np.save(output_path, mslbp_matrix)
    print(f"Multi-scale local binary pattern matrix saved to {output_path}")
    return None


# this function populates the mslbp_matrix from the win_template applied on img_array
def print_win_template(img_array, win_template, mslbp_matrix):
    # Get image dimensions
    height, width = img_array.shape
    edges = 7
    # iterate over the win_template and populate the mslbp_matrix; ignore edges for now 
    for i in range(edges, height - edges):
            for j in range(edges, width - edges):
                radius, neighbors = win_template[i][j][0]
                pixel = (i, j)
                neighborhood_pixels = lbp_descriptor(radius, neighbors, pixel)
                temp_lbp = 0
                for k in range(len(neighborhood_pixels)):
                    if img_array[i, j] <= img_array[neighborhood_pixels[k]]:
                        temp_lbp += 2**k
                    else:
                        temp_lbp += 0
            mslbp_matrix[i][j] = temp_lbp
    # return the mslbp_matrix
    return mslbp_matrix

def compute_mslbp(img_array, win_template):
    # Initialize current_template
    current_template = [[((i, j), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]
    # Define key value pairs of radius and neighbs
    RADIUS_NEIGHBS = [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 16)]
    radius_neighbs = RADIUS_NEIGHBS
 #   length = len(neighborhood_pixels)
    # Get image dimensions
    height, width = img_array.shape
    # initialize the current_template with the radius and neighbors set to the fixed values
    current_template = [[((radius_neighbs[0][0], radius_neighbs[0][1]), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]
    # initialize the win_template with the radius and neighbors set to the fixed values
    win_template = [[((radius_neighbs[0][0], radius_neighbs[0][1]), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]
    # Iterate over the radius_neighbs
    for radius, neighbors in radius_neighbs:
        
        # iterate over the image array; ignore the edges for now
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                # after each iteration temp_lbp should be zeroed out
                temp_lbp = 0
                # each computation of the lbp_descriptor will be compared and assigned to the winning template
                # Get the pixel value
                pixel = (i, j)
                # Get the neighborhood pixels
                neighborhood_pixels = lbp_descriptor(radius, neighbors, pixel)
                # Compute the linear binary pattern descriptor over the neighborhood pixels
                for k in range(len(neighborhood_pixels)):
                    if img_array[i, j] <= img_array[neighborhood_pixels[k]]:
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
    image_filename = 'camera128.bmp'

    # Full path to the image
    image_path = os.path.join(image_directory, image_filename)
    # Read the image
    img = Image.open(image_path)    
    # Convert to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    # Convert image to numpy array as type float
    img_array = np.array(img, dtype=np.float32) 
    
    # Initialize win_template
    win_template = [[((i, j), 0) for j in range(img_array.shape[1])] for i in range(img_array.shape[0])]
    # Create a copy to send to the function
    win_template_copy = copy.deepcopy(win_template)
    # Create the winning template 
    win_template = compute_mslbp(img_array,win_template_copy)
    # Initialize and populate from the image the multi scale local binary pattern matrix
    mslbp_matrix = np.zeros_like(img_array)
    # Create a copy to send to the function
    mslbp_matrix_copy = copy.deepcopy(mslbp_matrix)
    mslbp_matrix = print_win_template(img_array, win_template, mslbp_matrix_copy)

    matrix_to_image(mslbp_matrix, image_directory)

if __name__ == "__main__":
    main()