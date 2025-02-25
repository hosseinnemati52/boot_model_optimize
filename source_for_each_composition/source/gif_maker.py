from PIL import Image
import os
import re


def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

def create_gif_from_pngs(directory_path, output_path, duration=50):
    # Ensure the directory path uses the correct separator for Windows
    directory_path = os.path.normpath(directory_path)
    output_path = os.path.normpath(output_path)

    # Get all PNG files from the specified directory
    png_files = [f for f in os.listdir(directory_path) if f.endswith('.PNG')]
    png_files.sort(key=numerical_sort)

    # List to store the images
    images = []

    for file_name in png_files:
        file_path = os.path.join(directory_path, file_name)
        image = Image.open(file_path)
        images.append(image)

    # Save the images as a GIF
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF created and saved at {output_path}")
    else:
        print("No PNG files found in the specified directory.")

# Example usage:
directory_path =os.getcwd()+ '/frames'  # Local directory containing PNG files
output_path = os.getcwd()+'/output.gif'         # Local path to save the output GIF
create_gif_from_pngs(directory_path, output_path)


# Example usage:
directory_path =os.getcwd()+ '/fit_land'  # Local directory containing PNG files
output_path = os.getcwd()+'/fit_land_anim.gif'         # Local path to save the output GIF
create_gif_from_pngs(directory_path, output_path)