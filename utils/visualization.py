"""
    Q. Why this file is almost empty?

    I don't need seperate code for visualization since I have already integrated it 
	with DataMapping class in utils/data_mapping.py
	It has a visualize function which does good enough of a job.
	
	The reason behind doing this is that I didn't want to keep functionality
	away from the data it is supposed to work on.
	
	This way one class can hold all the data and mapping, and utilize it for visualization
"""
import os
def shorten_path(path, max_length=30):
	"""
	Shorten the path for display purposes.
	"""
	if len(path) <= max_length:
		return path
	head, tail = os.path.split(path)
	head = os.path.dirname(head)
	return os.path.join(os.path.basename(head), '...', tail)


if __name__ == "__main__":

    # Example usage
	# the path is from my own desktop
	# Here I am shortening the path so when I am displaying the table I can 
	# do it in a short manner
    long_path = "/home/odin/Documents/internship-task/data/input_images/000000000025.jpg"
    short_path = shorten_path(long_path)
    print(short_path)  # Output: internship-task/.../000000000025.jpg