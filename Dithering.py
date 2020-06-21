import sys, os
import numpy as np
from PIL import Image

# ----- Convert to gray scale ----- #
def read_img(path, height):
    """ Open the image at the specified path and resize it
        Resize the image given the height in pixels
        Returns the image as array of 3x8 bit float between 0 and 1 
    """
    img = Image.open(path)
    
    orig_w, orig_h = img.size
    width = int(orig_w * height / orig_h)
    
    img = img.resize((width,height),Image.ANTIALIAS)

    return np.asarray(img,dtype=float)/255 # Array of floats between 0 and 1


# ----- Convert to gray scale ----- #
def grayscale(img_data):
    """ Takes a RGB or RGBa image as input
        Return a gray scale image as float array
    """

    assert(len(img_data.shape)>2 and img_data.shape[2] == 3)

    gray =  (img_data[:,:,0] * 0.21 + \
            img_data[:,:,1] * 0.72 + \
            img_data[:,:,2] * 0.07)
    return gray

# ----- Convert to gray scale ----- #
def floyd_steinberg(image, quantization_factor = 1):
    """ Takes a grayscale image as a 2D float array in input
        Quantization factor define the number of colors 
    
    """
    img = image.copy()

    distribution = np.array([7,3,5,1], dtype=float)/16
    
    for row in range(img.shape[0]-1):
        for col in range(img.shape[1]-1):
            
            # Pixel rounding 
            old = img[row, col]
            img[row, col] = np.round(old*quantization_factor)/quantization_factor #np.round(old)
            
            # Calculaton of the error
            error = old - img[row, col]
            
            # Distribution of the error
            err_dist=(distribution * error)

            img[row,col+1] +=  err_dist[0] 
            img[row+1,col-1] +=  err_dist[1]
            img[row+1,col] +=  err_dist[2]
            img[row+1,col+1] +=  err_dist[3]

        if row % 10 == 0: print_progress(row/img.shape[0])

    print_progress(1.) #Print finished

    # Border of the image set to white
    img[:,0]=img[0,:]=img[:,-1]=img[-1,:] = 1.
    
    return img

# ----- Saving the image on a file ----- #
def export_img(img, name, black_white):
    coverted_im =(img*255).astype("uint8")
    pil_img = Image.fromarray(coverted_im)
    if black_white == True:
        pil_img.convert("1") # Image converted to 1-bit image
    pil_img.save(name+"_dithered.png")


def print_progress(perc):
    print("\r----> [{:>3.0%}]".format(perc), end='')
    if perc == 1: print("\n")

# ----- Timing wrapper ----- #
def Timer(func):
    import time
    def wrap(*args, **kwargs):
        start = time.perf_counter()
        val = func(*args, **kwargs)
        seconds = time.perf_counter() - start
        print("Image processed in {0:.5f} seconds".format(seconds))
        return val
    return wrap

@Timer
def Dithering(path, height, quantization_factor = 1):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    RGB_image = read_img(path, height)
    gray_image = grayscale(RGB_image)
    
    print("The image {} is being processed.".format(base) )
    
    dithered_image = floyd_steinberg(gray_image, quantization_factor)
    
    b_w = True if quantization_factor == 1 else False # If the image is in black and white it can be further compressed
    export_img(dithered_image, name, b_w)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("This software will perform dithering on a image using the Floyd-Steinberg algorithm.\n\nIMAGE PATH: ")
        path = input()
    
    try:
        Dithering(path,400)
    except IOError:
        print("Unable to open/find the file or the file is not in an image format")
    except AssertionError:
        print("Image not in RGB format")
    #except Exception as e:
    #    print(e)
    finally:
        input("Press enter to terminate")
        sys.exit(1)       

    input("Press enter to terminate")


