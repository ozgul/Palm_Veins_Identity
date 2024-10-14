import numpy as np
from PIL import Image
import os
import copy

# Function to update the pheromone values globally
def global_pheromone_update(pheromone_matrix, visited_pixels):

    # Iterate over the visited_pixels dictionary
    for pixel, data in visited_pixels.items():
        x, y = pixel  # Unpack the pixel tuple into x and y

        # Compute the new pheromone value
        probability = visited_pixels[pixel]['probability']
        new_pheromone = ((1 - probability) * pheromone_matrix[x, y] + probability * visited_pixels[pixel]['pheromone'])

        # Update the pheromone_matrix with the new pheromone value
        pheromone_matrix[x, y] = new_pheromone

    return pheromone_matrix


# Function to move the ant; populates the visited_pixels dictionary and pheromone_matrix
# will be called by L many times; process over the ants located at pixels of the image
def move_ant(ant_position, pheromone_matrix, heuristic_matrix, visited_pixels):
    
    alpha = 1
    beta = 2
    decay_coefficient   = 0.1
    initial_pheromone = 0.1

    # Get the first key with nonzero flag from ant_position
    # that contains pixels with at least one ant

    # Iterate over the ant_position to find the first key with a flag of 1
    for pixel, flag in list(ant_position.items()):
        if flag == 1 :  # Check if the flag is 1 and ant is not on the edge of the image
            x, y = pixel  # Unpack the pixel tuple into x and y
            # break  # Exit the loop once the first non-zero flag (1) is found
        height, width = pheromone_matrix.shape

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
        if probabilities:
            max_probability, (new_x, new_y) = max(probabilities, key=lambda item: item[0])
        else:
            continue  # Skip to the next iteration if no valid movements are found
        max_probability = float(max_probability)  # Ensure max_probability is of type float

    # the flag of the new position of the ant should be 1 
    # Update the specific key (x, y) with the value 1
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
        new_pheromone = (decay_coefficient * initial_pheromone + 
                        (1 - decay_coefficient) * current_pheromone)
        pheromone_matrix[update_pheromone_at] = new_pheromone
    
    

    # Populate the visited_pixels dictionary with the new values
    # Ensure the key exists in visited_pixels

        # Update the visited_pixels dictionary with the new values
        if (new_x, new_y) not in visited_pixels:
            visited_pixels[(new_x, new_y)] = {'probability': 0.0, 'pheromone': 1.0, 'flag': 0}
    # Update only the 'probability' field with the maximum value
        visited_pixels[(new_x, new_y)]['probability'] = max(visited_pixels[(new_x, new_y)]['probability'], max_probability)
    # Add the new pheromone difference to the 'pheromone_differences' field
        visited_pixels[(new_x, new_y)]['pheromone'] += pheromone_matrix[update_pheromone_at]
    # Update the 'flag' field to 1, I dont think this is necessary
        visited_pixels[(new_x, new_y)]['flag'] = 1
    
    return ant_position, pheromone_matrix, visited_pixels

def initialization(img_array,heuristic_matrix):
    # Get image dimensions
    height, width = img_array.shape
    # Compute initial intensity variation V_c(i,j)
    for i in range(2, height-2):
        for j in range(2, width-2):
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
# Stores the pixels visited by at least one ant
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
    
    # Convert image to numpy array as type float
    img_array = np.array(img, dtype=np.float32)
    
    # Get image dimensions
    height, width = img_array.shape
    
    #global pheromone_matrix  # Declare pheromone_matrix as global   
    # Create pheromone_matrix with random values < 10; should be set to 0
    pheromone_matrix = np.random.rand(height, width).astype(np.float32)

    #global ant_position  # Declare ant_position as global
    # Initialize ant_position with 1 for all pixels in the image
    ant_position = {(x, y): 1 for x in range(width) for y in range(height)}

    #global visited_pixels  # Declare visited_pixels as global
    # Initialize visited_pixels with 0 for all pixels in the image
    visited_pixels = {(x, y): {'probability': 0.0, 'pheromone': 1.0, 'flag': 0} for x in range(width) for y in range(height)}
    print(f"image in numpy :\n{img_array}")
    # Initialize heuristic_matrix
    heuristic_matrix = np.full_like(img_array, 0.1, dtype=np.float32)
    # Create, Initialize and populate heuristic_matrix from the image
    heuristic_matrix_copy = copy.deepcopy(heuristic_matrix)
    heuristic_matrix = initialization(img_array,heuristic_matrix_copy)

    # Move the ants on the image for L many times (iterations)
    # We test it initially for one iteration
    # Each function call will move all the ants to a new pixel
    # Construction of L steps of the ant movement
    for _ in range(2):
        # Create deep copies of the variables before passing them to the function
        ant_position_copy = copy.deepcopy(ant_position)
        pheromone_matrix_copy = copy.deepcopy(pheromone_matrix)
        visited_pixels_copy = copy.deepcopy(visited_pixels)
        # Move the ants
        # Pass deep copies to the function
        updated_ant_position, updated_pheromone_matrix, updated_visited_pixels = move_ant(
            ant_position_copy, pheromone_matrix_copy, heuristic_matrix, visited_pixels_copy
        )
        # Assign returned updated values to the original variables
        ant_position = updated_ant_position
        pheromone_matrix = updated_pheromone_matrix
        visited_pixels = updated_visited_pixels
        # Create a deep copy before global update
        pheromone_matrix_copy = copy.deepcopy(pheromone_matrix)
        # Update the pheromone values globally
        # Update the pheromone matrix globally and reassign the original
        pheromone_matrix = global_pheromone_update(pheromone_matrix_copy, updated_visited_pixels)
    
    print(f"Image shape: {img_array.shape}")
    print(f"Pheromone matrix shape: {pheromone_matrix.shape}")
    print(f"Heuristic matrix shape: {heuristic_matrix.shape}")
    print(f"Sample of pheromone matrix:\n{pheromone_matrix[:10, :10]}")
    print(f"Sample of heuristic matrix (created once and constant):\n{heuristic_matrix[:10, :10]}")


if __name__ == "__main__":
    main()
