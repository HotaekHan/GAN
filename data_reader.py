import os

def read_data_list(path_dir, image_ext):

    list_of_images = []

    # Read Train Image
    for (path, dir, files) in os.walk(path_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[-1]

            if ext == image_ext:
                image_path = path + "\\" + file_name
                list_of_images.append(image_path)

                print("Train : %s" % image_path)

    return list_of_images