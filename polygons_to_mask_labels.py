import os
import cv2
import numpy as np


images_dir = "./expanded_dataset/images"
labels_dir = "./expanded_dataset/polygon_labels"
output_dir = "./expanded_dataset/labels"

os.makedirs(output_dir, exist_ok=True)

def read_polygons(file_path):
    polygons = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            points = line.strip().split()[1:]  # Пропускаем первый элемент (класс)
            polygon = np.array(points, dtype=np.float32).reshape(-1, 2)
            polygons.append(polygon)
    return polygons


# Проходим по всем изображениям
for image_name in os.listdir(images_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):

        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось прочитать файл: {image_path}")
            continue

        height, width, _ = image.shape

        mask = np.zeros((height, width), dtype=np.uint8)

        # Закраска по полигонам
        if os.path.exists(label_path):
            polygons = read_polygons(label_path)
            for polygon in polygons:
                polygon[:, 0] *= width
                polygon[:, 1] *= height
                polygon = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon], 255)

        mask_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.jpg')
        cv2.imwrite(mask_path, mask)

        print(f"{image_path}\n{mask_path}\n")

print("Маски успешно созданы и сохранены в директорию:", output_dir)


#check paths
# for image_name in os.listdir(images_dir):
#     if image_name.endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(images_dir, image_name)
#         label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
#         print(f"Путь к изображению: {image_path}")
#         print(f"Путь к меткам: {label_path}")

#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Не удалось прочитать файл: {image_path}")
#             continue
