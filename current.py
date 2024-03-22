import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseButton
from PIL import Image
from numpy.linalg import norm

fig, ax = plt.subplots()
fig.subplots_adjust(0,0,1,1)

checked_pixels = []
pixel_queue = []
reddened_pixel_count = 0

chosen_image = str(input('File name: ')) 

while True:
    similarity_variable = float(input('Similarity variable (0.1 to 1): '))
    
    while similarity_variable < 0.1 or similarity_variable > 1:
        print('Similarity variable has an incorrect value, try again.')
        similarity_variable = float(input('Similarity variable (0.1 to 1): '))
    
    break

with Image.open(chosen_image) as im: # open the image in read (rb) mode
    image_pixel_width, image_pixel_height = im.size
    print(f"Image size in pixels is {im.size}, image mode is {im.mode}")
    im_array = np.asarray(im)
    writable_im_array = im_array.copy() # to circumvent the editing disabled by default
    im_axesimage = ax.imshow(im_array) # wczytuje koordowy array i go pokazuje
    im_pixels_array = im.load()
    
def cos_similarity(vector1, vector2):
    cosine = np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))
    return cosine

def return_neighbors(x, y): 
    up = [x, y+1]
    upleft = [x-1, y+1]
    upright = [x+1, y+1]
    down = [x, y-1]
    downleft = [x-1, y-1]
    downright = [x+1, y-1]
    left = [x-1, y]
    right = [x+1, y]
    return up, upright, upleft, down, downleft, downright, left, right
    
def redden_pixel(x, y):
    global reddened_pixel_count
    print(f'Pixel {(x, y)} passed the check and is turning red')
    writable_im_array[y, x] = [255, 0, 0]
    reddened_pixel_count += 1

def pixel_picker(event):
    
    if event.button is MouseButton.LEFT and isinstance(im_axesimage, AxesImage):
        #try:  
            x = int(event.xdata) #column 
            y = int(event.ydata) #row 
            pixel_values = im_pixels_array[x, y] # zgodnie z indeksowaniem pixelowym, kliknięty pixel to ten pokazywany przez wbudowany 
            #indeksator matplotliba
            print(f"The pixel's RGB values are {pixel_values} and its position is {x, y}") # tu zostawiamy indeksowanie niepixelowe
            # żeby było zgodne z tym, co pokazuje indeksator matplotliba
            writable_im_array[y, x] = (255, 0, 0)

            def in_image(x, y):
                if x + 1 <= image_pixel_width and x - 1 >= 0 and y + 1 <= image_pixel_height and y - 1 >= 0: # to nie jest praca na arrayu tylko na plocie, więc x,y powinno działać poprawnie
                    return True                        
                else:
                    return False
                        
            def check_pixel(x, y):
                 
                if not in_image(x, y):
                    print(f'Neighbor {(x, y)} is not in the image.')
                    return
                    
                if (x, y) not in checked_pixels:
                    checked_pixels.append((x, y))
                else:
                    return
                
                if cos_similarity(im_pixels_array[x, y], pixel_values) < similarity_variable:
                    print(f'Neighbor {(x, y)} did not pass the similarity check')  
                    return
                
                return True                     

            def queue_func(x, y):
                
                neighbors = return_neighbors(x, y) 
                print(f'Neighbors of [{x}, {y}] are {neighbors}')

                for n in neighbors:
                    if check_pixel(n[0], n[1]):
                        redden_pixel(n[0], n[1])
                        pixel_queue.append(n)

            queue_func(x, y)

            while len(pixel_queue) > 0:
                a = pixel_queue[0]
                del pixel_queue[0]
                queue_func(a[0], a[1])

            im = Image.fromarray(writable_im_array)
            split_name = chosen_image.split('.')
            new_name = split_name[0] + ' modified' + '.' + split_name[1]
            im.save(new_name)
            im = Image.open(new_name)
            ax.imshow(writable_im_array)
            plt.show()

plt.connect('button_press_event', pixel_picker)
plt.show()