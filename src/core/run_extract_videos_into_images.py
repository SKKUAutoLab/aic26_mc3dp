import argparse
import glob
import json
import os
import shutil
from copy import deepcopy

from loguru import logger
from tqdm import tqdm

def custom_file_sort(file_path):
	basename       = os.path.basename(file_path)
	basename_noext = os.path.splitext(basename)[0]
	file_index     = basename_noext.split("_")[-1]
	return int(file_index)

def merge_videos(folder_input, scene_name):
	folder_input_scene  = os.path.join(folder_input, scene_name)
	folder_input_videos = os.path.join(folder_input, scene_name, "videos/")
	list_files = sorted(glob.glob(os.path.join(folder_input_videos, "*.mp4")))

	# check list file
	if len(list_files) == 0:
		logger.warning(f"No video files found in {folder_input_videos}")
		return

	# show list files
	list_files_str = ""
	for file_path in list_files:
		list_files_str += f"{os.path.basename(file_path)}, "
	logger.info(f"List files to merge: [{list_files_str}]")

	# Create output filename
	output_file = os.path.join(folder_input_scene, f"{scene_name}.mp4")

	# Create file list for ffmpeg concat
	concat_file = os.path.join(folder_input_scene, "file_list.txt")
	with open(concat_file, 'w') as f:
		for video_file in list_files:
			f.write(f"file 'videos/{os.path.basename(video_file)}'\n")

	try:
		# Change to video directory and run ffmpeg command
		current_dir = os.getcwd()
		os.chdir(folder_input_scene)

		cmd = f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy -y {scene_name}.mp4"

		logger.info(f"Merging {len(list_files)} videos into {output_file}")
		result = os.system(cmd)

		if result == 0:
			logger.success(f"Successfully merged videos to {output_file}")
		else:
			logger.error(f"FFmpeg command failed with exit code: {result}")

	except Exception as e:
		logger.error(f"Error merging videos: {e}")
	finally:
		# Restore original directory
		os.chdir(current_dir)
		# Clean up temporary file
		if os.path.exists(concat_file):
			os.remove(concat_file)

def extract_full_videos_into_images(folder_input, scene_name, folder_output):
	folder_input_scene  = os.path.join(folder_input, scene_name)
	folder_input_videos = os.path.join(folder_input_scene, "videos/")
	list_files = sorted(glob.glob(os.path.join(folder_input_videos, "*.mp4")), key=custom_file_sort)

	# check list file
	if len(list_files) == 0:
		logger.warning(f"No video files found in {folder_input_videos}.")
		return

	# Create output folder if it doesn't exist
	folder_output_scene = os.path.join(folder_output, scene_name)
	os.makedirs(folder_output_scene, exist_ok=True)

	# copy map.png into folder scene
	map_file = os.path.join(folder_input_scene, "map.png")
	if os.path.exists(map_file):
		folder_output_map = os.path.join(folder_output, scene_name, "map.png")
		shutil.copy(map_file, folder_output_map)
		logger.info(f"Copied map file to {folder_output_map}")

	for video_file in tqdm(list_files, desc=f"Extracting images from videos in {scene_name}", colour="yellow", leave=False, unit="video"):
		video_name = os.path.splitext(os.path.basename(video_file))[0]
		folder_output_video = os.path.join(folder_output_scene, video_name)
		os.makedirs(folder_output_video, exist_ok=True)

		cmd = f"ffmpeg -i {video_file} -start_number 0 {folder_output_video}/%08d.jpg"
		logger.info(f"Extracting images from {video_file} to {folder_output_video}")
		result = os.system(cmd)

		if result != 0:
			logger.error(f"FFmpeg command failed for {video_file} with exit code: {result}")


def main():
	parser = argparse.ArgumentParser(description="Process prepare detection dataset.")
	parser.add_argument('--input_dataset' , type = str, default = "/media/vsw-ws-05/SSD_1/AI_City_Challenge/test", help = "Folder where dataset")
	parser.add_argument('--output_dataset' , type = str, default = "/media/vsw-ws-05/SSD_1/AI_City_Challenge_processing_baseline/images", help = "Folder where result out")
	parser.add_argument('--scenes', nargs='+', type=str,
                    default=["Warehouse_023", "Warehouse_024", "Warehouse_025", "Warehouse_026", "Warehouse_027"],
                    help='List of scene names to process')
	args = parser.parse_args()

	folder_intput = args.input_dataset
	folder_output = args.output_dataset
	list_scene    = args.scenes
	# list_scene    = ["Warehouse_017"]

	# NOTE: Adjust camera IDs in each JSON file
	pbar = tqdm(list_scene, desc="Adjust camera IDs in calibration files")
	for scene_name in tqdm(list_scene, desc="Processing scenes", colour="blue"):
		pbar.set_description(f"Preparing detection dataset scene: {scene_name}")

		# Merge videos
		# merge_videos(folder_intput, scene_name)

		# Extract images from each videos
		extract_full_videos_into_images(folder_intput, scene_name, folder_output)

		pbar.update(1)
	pbar.close()

if __name__ == "__main__":
	main()
