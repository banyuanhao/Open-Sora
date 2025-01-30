import cv2
import os

def images_to_video(image_folder, video_name, fps=30):
    """
    Reads 16 PNG images from a folder and writes them into a video file.
    
    Parameters:
    - image_folder: The directory where the PNG images are stored.
    - video_name: The output video file name (e.g., 'output_video.mp4').
    - fps: Frames per second for the video (default is 30).
    """
    # Get the list of PNG files in the folder, sorted by file name
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    
    if len(images) != 16:
        raise ValueError("The folder must contain exactly 16 PNG images.")
    
    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Loop through the images and write each one to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the VideoWriter object
    video.release()
    print(f"Video saved as {video_name}")

if __name__ == "__main__":
    images_to_video('rope/xaxis', 'output_video.mp4', fps=16)