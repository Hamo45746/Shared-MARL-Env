from PIL import Image
import os

image_dir = "episode_images"
images = []

for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(".png"):
        images.append(Image.open(os.path.join(image_dir, filename)))

if images:
    images[0].save("episode.gif", save_all=True, append_images=images[1:], duration=200, loop=0)