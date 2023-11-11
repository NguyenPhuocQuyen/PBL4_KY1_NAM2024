# from PIL import Image
# import os

# def is_image(file_path):
#     try:
#         Image.open(file_path)
#         return True
#     except:
#         return False

# def remove_invalid_images_recursively(directory_path):
#     for item in os.listdir(directory_path):
#         item_path = os.path.join(directory_path, item)
#         if os.path.isdir(item_path):
#             remove_invalid_images_recursively(item_path)
#         else:
#             # print(item_path)
#             if not item_path.endswith('.jpg'):
#                 print(f"Removing {item_path}")
#                 os.remove(item_path)

# # Đặt đường dẫn đến thư mục chứa các thư mục con của bạn ở đây
# main_directory_path = r'D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\archive_split_main'
# remove_invalid_images_recursively(main_directory_path)

from pathlib import Path
from PIL import Image
import os

# remove_array = []
data_dir = r"D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\archive_split_main\train\cardboard"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["jpeg"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        # print(filepath)
        try:
            with Image.open(filepath) as img:
                img_type = img.format.lower()
                img.close()  # Close the image to release the resources
        except Exception as e:
            print(f"Error {filepath} - {e}")
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            # remove this filepath 
            if filepath.is_file():
                filepath.unlink()  # Remove the file if it's not accepted by TensorFlow
#             remove_array.append(filepath)

# for file in remove_array:
#     os.remove(filepath)
#     print("remove" + filepath )
