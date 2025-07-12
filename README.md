# COMP8510_Project
Computer Vision Project

How To Run the Program:
Before running the file, the following items need to be imported into your folder's venv:
- cv2
- numpy
- PIL
- PyQt5

To run the program from the command line: python .\epipolar.py

From there, once the program is executed, a window will pop up which will let you select the left image and the right image, and then calculate the fundamental matrix.
After the fundamental matrix is calculated, if you click somewhere on the left image, it will try to find a match to the right image. The bottom left corner will indicate the pixel in the right image and the ZNCC score as well.
