import numpy as np
from PIL import Image
import os
import copy

# Function to update the pheromone values globally
def global_pheromone_update(pheromone_matrix):
    decay_coefficient = 0.05
    # Iterate over the pheromone_matrix to apply decay
    for i in range(pheromone_matrix.shape[0]):
        for j in range(pheromone_matrix.shape[1]):
            pheromone_matrix[i, j] *= (1 - decay_coefficient)
    return pheromone_matrix


# Function to move the ant; populates the visited_pixels dictionary and pheromone_matrix
# will be called by L many times; process over the ants located at pixels of the image
def move_ant(ant_position, pheromone_matrix, heuristic_matrix):
    # Parameters for the ant movement
    alpha = 1.0 # Pheromone weight
    beta = 0.1  # Heuristic weight
    decay_coefficient   = 0.05
    initial_pheromone = 0.1
    rho = 0.1 # Evaporation rate

    # Get the first key with nonzero flag from ant_position
    # that contains pixels with at least one ant
    height, width = pheromone_matrix.shape
    # Iterate over the ant_position to find all keys with a flag of 1
    nonzero_flags = [(x, y) for (x, y), flag in ant_position.items() if flag == 1]
    for x, y in nonzero_flags:
       # x, y = pixel  # Unpack the pixel tuple into x and y
            # break  # Exit the loop once the first non-zero flag (1) is found
    # if the ant is on the corners of the image the possible movements are limited, add an if statement to handle this case
    # Find out the (0,0) point of the image depending on the height and width of the image to find the coordinates of the
    # edges and corners of the image to write specific possible movements array
     
       # movements = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        if x == 0 and y == 0:  # Top left corner
            movements = [(0, 1), (0, -1), (-1, 1)]
        elif x == 0 and y == width - 1:  # Top right corner
            movements = [(0, -1), (-1, 0), (-1, -1)]
        elif x == 0 and y == -height + 1 :  # Bottom left corner
            movements = [(0, 1), (1, 0), (1, -1)]
        elif x == width - 1 and y == -height + 1:  # Bottom right corner
            movements = [(0, 1), (-1, 0), (-1, 1)]
        elif x == 0:  # Left edge
            movements = [(0, 1), (0, -1), (1, 1), (1, -1), (1, 0)]
        elif x == width - 1:  # Right edge
            movements = [(0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
        elif y == 0:  # Top edge
            movements = [(-1, 0), (1, 0), (0, -1), (1, -1), (1, -1)]
        elif y == width - 1:  # Bottom edge
            movements = [(-1, 0), (1, 0), (0, -1), (1, -1), (-1, -1)]
        else:
            movements = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # if the ant is on the corners of the image the possible movements are limited, add an if statement to handle this case 
    # Compute probabilities for each possible movement and store in array probabilities
    # Each element is a tuple of probability and coordinates
        probabilities = []
        for dx, dy in movements:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < height and 0 <= new_y < width:
                pheromone = pheromone_matrix[new_x, new_y]
                heuristic = heuristic_matrix[new_x, new_y]
                probability = (pheromone ** alpha) * (heuristic ** beta)
                probabilities.append((probability, (new_x, new_y)))

    # Find the maximum probability and assign the corresponding (x, y) to new_x, new_y
    # if all the prob are the same or there are same values then how should ant behave
    # rewrite the code; if probabilities are the same then the ant should move randomly;
    # if there is no where to move; probability is 0; the ant should not move
        if probabilities:
            max_probability, (new_x, new_y) = max(probabilities, key=lambda item: item[0])
        else:
            continue  # Skip to the next iteration if no valid movements are found
        max_probability = float(max_probability)  # Ensure max_probability is of type float

    # the flag of the new position of the ant should be 1 
    # Update the specific key (x, y) with the value 1
    # add a new variable moves and if max is found move the ant otherwise dont move
        ant_position[(new_x,new_y)] = 1
    # the flag of the previous position of the ant should be 0
        ant_position[(x,y)] = 0 

    # Computation of normalized values of probabilites for each possible movement
        total = sum(prob for prob, _ in probabilities)
        probabilities = [(prob / total, pos) for prob, pos in probabilities]
    # Find the maximum value in probabilities
        max_probability = float(max(probabilities, key=lambda item: item[0])[0])
    # max of probabilities will be used in the global pheromone update
    # Update the local pheromone value where the ant moves
        update_pheromone_at = new_x, new_y
    # Store current pheromone value
        current_pheromone = pheromone_matrix[update_pheromone_at]
    
    # Write sample values of the local pheromone update equation 
    # Update local pheromone value
        new_pheromone = (rho * heuristic_matrix[new_x, new_y] + 
                        (1 - rho) * current_pheromone)
        pheromone_matrix[update_pheromone_at] = new_pheromone  
    
    return ant_position, pheromone_matrix

def initialization(img_array,heuristic_matrix):
    # Get image dimensions
    height, width = img_array.shape
    # Compute initial intensity variation V_c(i,j)
    for i in range(2, height-2): # Skip the first and last two rows
        for j in range(2, width-2): # Skip the first and last two columns 
            V_c = abs(img_array[i-2, j-1] - img_array[i+2, j+1]) + \
                  abs(img_array[i-2, j+1] - img_array[i+2, j-1]) + \
                  abs(img_array[i-1, j-2] - img_array[i+1, j+2]) + \
                  abs(img_array[i-1, j+2] - img_array[i+1, j-2]) + \
                  abs(img_array[i-1, j-1] - img_array[i+1, j+1]) + \
                  abs(img_array[i-1, j+1] - img_array[i+1, j-1]) + \
                  abs(img_array[i-1, j] - img_array[i+1, j]) + \
                  abs(img_array[i, j-1] - img_array[i, j+1])
            
            heuristic_matrix[i, j] = V_c
      
    # Compute V_max
    V_max = np.sum(heuristic_matrix) 
    # Normalize heuristic_matrix
    if V_max != 0:
        heuristic_matrix /= V_max  
  #  print(f"Sample of heuristic matrix (created once and constant):\n{intensity_matrix[:10, :10]}") 
    return heuristic_matrix

def main():
# Stores the pixels that contains at least one ant
    ant_position = {}
# Tracks accumulated pixels and their data 
# to be used for global update of pheromone values is a dict of dict
    visited_pixels = {}
    heuristic_matrix = None
    pheromone_matrix = None
    # Directory containing the image
    image_directory = r'C:\Users\ozgul\Documents\GitHub\Palm_Veins_Identity\images'
    image_filename = 'CroppedNoBGLightClose.png'
    # Full path to the image
    image_path = os.path.join(image_directory, image_filename)
    # Read the image
    img = Image.open(image_path)    
    # Convert to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    # Convert image to numpy array as type float; also normalize pixel values between 0 and 1 
    img_array = np.array(img, dtype=np.float32) 
    # Get image dimensions
    height, width = img_array.shape
    # Create pheromone_matrix initialized with 0.0001 for all pixels in the image
    pheromone_matrix = np.full((height, width), 0.0001, dtype=np.float32)
    # Initialize ant_position with 1 for all pixels in the image; start with an ant at each pixel.
    ant_position = {(x, y): 1 for x in range(width) for y in range(height)}

    # Print the image in numpy format
    print(f"image in numpy :\n{img_array}")
    # Initialize heuristic_matrix
    heuristic_matrix = np.full_like(img_array, 0.1, dtype=np.float32)
    # Create, Initialize and populate heuristic_matrix from the image; this is done only once
    heuristic_matrix_copy = copy.deepcopy(heuristic_matrix)
    heuristic_matrix = initialization(img_array,heuristic_matrix_copy)

    # Move the ants on the image for ants_pixels_count many times and update the pheromone values
    # We test it initially for one iteration; can be repeated 2 or 3 many times with an outer loop
    # Global Pheromone Update is done after all ants have moved once for each iteration
    # Each function call will move all the ants to a new pixel
    # Each ant can in theory move over the all pixels of the image; 
    # define a variable to control the number of steps; ants_pixels_move = height * width
    ants_pixels_move = 80
    # We process number of ants equal to number of pixels in move_ant() function
    for _ in range(ants_pixels_move):
        # Create deep copies of the variables before passing them to the function
        ant_position_copy = copy.deepcopy(ant_position)
        pheromone_matrix_copy = copy.deepcopy(pheromone_matrix)
        # Move the ants
        # Pass deep copies to the function
        updated_ant_position, updated_pheromone_matrix = move_ant(ant_position_copy, pheromone_matrix_copy, heuristic_matrix)
        # Assign returned updated values to the original variables
        ant_position = updated_ant_position
        pheromone_matrix = updated_pheromone_matrix
        
        
    # Create a deep copy before global update
    pheromone_matrix_copy = copy.deepcopy(pheromone_matrix)
    # Update the pheromone values globally
    # Update the pheromone matrix globally and reassign the original
    pheromone_matrix = global_pheromone_update(pheromone_matrix_copy)

    print(f"Image shape: {img_array.shape}")
    print(f"Pheromone matrix shape: {pheromone_matrix.shape}")
    print(f"Heuristic matrix shape: {heuristic_matrix.shape}")
    print(f"Sample of pheromone matrix:\n{pheromone_matrix[:10, :10]}")
    print(f"Sample of heuristic matrix (created once and constant):\n{heuristic_matrix[:10, :10]}")
    # Write the pheromone_matrix to a file
    output_path = os.path.join(image_directory, 'pheromone_matrix.npy')
    np.save(output_path, pheromone_matrix)
    print(f"Pheromone matrix saved to {output_path}")

if __name__ == "__main__":
    main()
