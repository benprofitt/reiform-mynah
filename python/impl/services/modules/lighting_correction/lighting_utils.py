from impl.services.modules.lighting_correction.lighting_resources import *

def make_bright(im, rand=False):

    factor = 2
    if rand:
        factor = 1.6 + random.random()

    enhancer = ImageEnhance.Brightness(im)
    return enhancer.enhance(factor)

def make_dark(im, rand=False):

    min_val = 64
    factor = 0.6
    
    if rand:
        p = random.random()
        min_val = (76 * p)//1
        factor = p * 0.6 + 0.1

    extrema = Image.Image.getextrema(im)
    new_pixel = [e[0] + min_val for e in extrema]
    new_pixel = np.array(new_pixel)

    min_image = np.full([im.size[1], im.size[0], 3], new_pixel)
    
    arr = np.array(im) - min_image

    threshold = 0
    idx = arr[:,:,:] < threshold
    arr[idx] = 0

    im = Image.fromarray(arr.astype(np.uint8))

    enhancer = ImageEnhance.Brightness(im)
    im = enhancer.enhance(factor)
    return im