import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseButton
from PIL import Image
from numpy.linalg import norm

def main() -> None:
    fig, ax = plt.subplots()
    fig.subplots_adjust(0,0,1,1)

    checked_pixels = []
    pixel_queue = []
    reddened_pixel_count = 0

    chosen_image = str(input('File name: ')) 

    while True:
        similarity_variable = float(input('Enter similarity variable (0.1 to 1): '))

        while similarity_variable < 0.1 or similarity_variable > 1:
            print('Similarity variable has an incorrect value, try again.')
            similarity_variable = float(input('Enter similarity variable (0.1 to 1): '))

        break

    with Image.open(chosen_image) as image: # open the image in read (rb) mode
        image_pixel_width, image_pixel_height = image.size
        print(f"Image size in pixels is {image.size}, image mode is {image.mode}")
        uneditable_array = np.asarray(image)
        writable_array = uneditable_array.copy()
        axes_image = ax.imshow(uneditable_array) # imshow displays the array
        pixels_array = image.load()

    def cos_similarity(vector1, vector2) -> float:
        cosine = np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))
        return cosine

    def return_neighbors(x, y) -> list[str]: 
        up = (x, y+1)
        upleft = (x-1, y+1)
        upright = (x+1, y+1)
        down = (x, y-1)
        downleft = (x-1, y-1)
        downright = (x+1, y-1)
        left = (x-1, y)
        right = (x+1, y)
        return up, upright, upleft, down, downleft, downright, left, right

    def redden_pixel(x, y) -> None:
        nonlocal reddened_pixel_count
        print(f'Pixel {(x, y)} passed the check and is turning red')
        writable_array[y, x] = [255, 0, 0]
        reddened_pixel_count += 1

    def pixel_picker(event) -> None:

        if event.button is MouseButton.LEFT and isinstance(axes_image, AxesImage):

            x = int(event.xdata)  
            y = int(event.ydata) 
            pixel_values = pixels_array[x, y] 
            print(f"The pixel's RGB values are {pixel_values} and its position is {x, y}")
            print (writable_array[y, x])
            writable_array[y, x] = [255, 0, 0]

            def in_image(x, y):
                if x + 1 <= image_pixel_width and x - 1 >= 0 and y + 1 <= image_pixel_height and y - 1 >= 0:
                    return True

            def check_pixel(x, y):

                if not in_image(x, y):
                    print(f'Neighbor {(x, y)} is not in the image')
                    return

                if (x, y) in checked_pixels:
                    return

                checked_pixels.append((x, y))

                if cos_similarity(pixels_array[x, y], pixel_values) < similarity_variable:
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

            split_name = chosen_image.split('.')
            new_name = split_name[0] + ' modified' + '.' + split_name[1]

            im = Image.fromarray(writable_array)
            im.save(new_name)
            im = Image.open(new_name)
            ax.imshow(writable_array)
            plt.show()

    plt.connect('button_press_event', pixel_picker)
    plt.show()

if __name__ == "__main__":
    main()