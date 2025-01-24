#
#	makeRawImage.py
#
#	Created	:	24 Jan 2020
#	Author	:	Abhishek Bhaumick
#



import os
import argparse
import numpy
try:
	import tkinter
	from tkinter import filedialog
except ImportError:
	print("\n info: No GUI element found - disabling GUI \n")
	hasGUI = False
else:
	hasGUI = True
from PIL import Image
if hasGUI:
	from PIL import ImageTk


intensity_default = 0.04
s_vs_p_default = 0.5
displayImage = None

showChOrig = True
showChNoise = False

def main():

	parser = argparse.ArgumentParser(description='Prepare Image for CUDA Programming Lab 2')
	parser.add_argument("-f", dest="filename", 	required=False, help="input image file path", metavar="FILE")
	args = parser.parse_args()

	imgFilePath = args.filename
	if (not hasGUI) :
		if imgFilePath == None :
			parser.print_help()
			parser.error("No GUI - must specify image filepath using -f ")
	
	if hasGUI :
		tkWndw = tkinter.Tk()
		if (imgFilePath == None) :
			imgFilePath = filedialog.askopenfilename( initialdir=os.getcwd() )

	img = createRawBytes(imgFilePath)

	if hasGUI :
		img.show()
		displayImage = img
		tkImage = ImageTk.PhotoImage(displayImage)
		tkinter.Label(tkWndw, image=tkImage).pack()
		tkWndw.mainloop()

def createRawBytes(bytesFilePath: str) -> Image:
	'''
	Reads a .bytes file and converts it into a PIL.Image object

	Param
	-------
	bytesFilePath :	str : 
		path to the .bytes file
   
	Returns
	-------
	out : PIL.Image :
		object created from .bytes file data
	'''
	if os.path.isfile( bytesFilePath ):
		with open(bytesFilePath, "rb") as bytesFile :
			imgBytes = bytes(bytesFile.read())

			imgH = int.from_bytes(imgBytes[0:4], byteorder='little')
			imgW = int.from_bytes(imgBytes[4:8], byteorder='little')
			imgSize = imgH, imgW
			img = Image.frombytes('RGB', imgSize, imgBytes[16:])

			# print([int(x) for x in imgBytes[3088:4000:3]])

			return img

if __name__ == "__main__":
	main()
