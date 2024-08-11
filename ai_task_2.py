from PIL import Image
import sys

def color_to_black(img):
    img = img.convert("RGB")  
    pixels = img.load()  
    
    width, height = img.size
    
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]  
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            pixels[i, j] = (gray, gray, gray)
    
    return img

if __name__ == "__main__": 
    image_path = sys.argv[1]
    img = Image.open(image_path)
    gray_img = color_to_black(img)
    gray_img.show()


