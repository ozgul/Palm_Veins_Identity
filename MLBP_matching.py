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
    else:
        raise ValueError("Invalid radius or number of neighbors")
    # Write another list for the half circle and the quarter circle use existing lists
    
def initialization(img_array, mslbp_matrix):

    # mslbp_matrix = np.full_like(img_array, 0.1, dtype=np.float32)
    # Define a list of radius values
    radius_list = [1, 2, 3, 4, 5, 7]
    neighbors = 8 # 8 or 16 neighbors
    # Get image dimensions
    height, width = img_array.shape
    #iterate over the image array 
    index = radius_list[3]
    for i in range(index, height - index):
        for j in range(index, width - index):
            # Get the pixel value
            pixel = (i, j)
            # Get the neighborhood pixels
            neighborhood_pixels = lbp_descriptor(index, neighbors, pixel) # adjust the radius and number of neighbors
            for k in range(0, len(neighborhood_pixels)-1):
                if img_array[i, j] <= img_array[neighborhood_pixels[k]]:
                    mslbp_matrix[i, j] = mslbp_matrix[i, j] + 2**k
                else:
                    mslbp_matrix[i, j] = mslbp_matrix[i, j] + 0
    return mslbp_matrix

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
    
    # Initialize mslbp_matrix
    mslbp_matrix = np.full_like(img_array, 0.0, dtype=np.float32)
    # Create a copy to send to the function
    mslbp_matrix_copy = copy.deepcopy(mslbp_matrix)
    # Create, Initialize and populate from the image the multi scale local binary pattern matrix
    mslbp_matrix = initialization(img_array,mslbp_matrix_copy)

    # Normalize the mslbp_matrix to the range [0, 255]
    # If the radius is 16 then normalize to [0, 65535]
    mslbp_matrix_normalized = (255 * (mslbp_matrix - np.min(mslbp_matrix)) / (np.max(mslbp_matrix) - np.min(mslbp_matrix))).astype(np.uint8)
    
    # Convert the normalized matrix to an image
    mslbp_image = Image.fromarray(mslbp_matrix_normalized)
    
    # Save the image
    output_image_path = os.path.join(image_directory, 'mslbp_image.png')
    mslbp_image.save(output_image_path)
    print(f"Multi-scale local binary pattern image saved to {output_image_path}")
    
    # Write the mslbp_matrix to a file
    output_path = os.path.join(image_directory, 'mslbp.npy')
    np.save(output_path, mslbp_matrix)
    print(f"Multi-scale local binary pattern matrix saved to {output_path}")
    

if __name__ == "__main__":
    main()