from PIL import Image
import os

def create_image_grid(input_folder, output_file, grid_size=(10, 10), padding=2):
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    
    # Sort the files to ensure consistent order
    image_files.sort()
    
    # Group images by class
    class_images = {}
    for img_file in image_files:
        class_name = img_file.split('_')[0]
        if class_name not in class_images:
            class_images[class_name] = []
        if len(class_images[class_name]) < 10:
            class_images[class_name].append(img_file)
    
    if len(class_images) != grid_size[0]:
        raise ValueError(f"Expected {grid_size[0]} classes, but found {len(class_images)}.")
    
    # Flatten the list of images
    selected_images = [img for class_list in class_images.values() for img in class_list]
    
    # Open the first image to get dimensions
    with Image.open(os.path.join(input_folder, selected_images[0])) as img:
        img_width, img_height = img.size
    
    # Create a new image with the size of the grid, including padding
    grid_width = (img_width + padding) * grid_size[1] + padding
    grid_height = (img_height + padding) * grid_size[0] + padding
    grid_image = Image.new('RGB', (grid_width, grid_height), color='black')
    
    # Place images in the grid with padding
    for i, image_file in enumerate(selected_images):
        with Image.open(os.path.join(input_folder, image_file)) as img:
            x = (i % grid_size[1]) * (img_width + padding) + padding
            y = (i // grid_size[1]) * (img_height + padding) + padding
            grid_image.paste(img, (x, y))
    
    # Save the grid image
    grid_image.save(output_file)
    print(f"Grid image saved as {output_file}")

# Example usage
if __name__ == "__main__":
    create_image_grid('./Output_folder/svhn', 'svhn_grid.png', padding=2)  # Changed padding to 2
    create_image_grid('./Output_folder/mnistm', 'mnistm_grid.png', padding=2)  # Changed padding to 2

print("Grid creation completed.")
