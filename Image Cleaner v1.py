import os
from PIL import Image

def image_delete(input_path):
    try:
        with Image.open(input_path) as img:
            img.verify()
    except:
        os.remove(input_path)
        
def image_compressor(input_path, output_folder, output_name, new_size=(400, 300)):
    try:
        image = Image.open(input_path)
        resized_image = image.resize(new_size)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, output_name)
        resized_image.save(output_path, optimize=True, quality=50)
        
    except Exception as e:
        print(f"Error with image: {e}")

def main():
    folder_path = '/Users/jonathandrake/Desktop/Python Project/Images'
    
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
    
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_name = filename
            image_delete(input_path)
            image_compressor(input_path, folder_path, output_name)
        else:
            print(f'Skipping non-image file: {input_path}')
    
    print('Done')

if __name__ == '__main__':
    main()
