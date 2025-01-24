#
#	makeBytesImage.py
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
	parser.add_argument("-l", dest="level", 	type=float, required=False, help="added noise intensity", metavar="LEVEL")
	parser.add_argument("-r", dest="spRatio", 	type=float, required=False, help="Salt vs Pepper Ratio", metavar="RATIO")
	args = parser.parse_args()

	imgFilePath = args.filename
	
	try :
		tkWndw = tkinter.Tk()
	except Exception:
		hasGUI = False
	else :
		hasGUI = True
		if (imgFilePath == None) :
			imgFilePath = filedialog.askopenfilename( initialdir=os.getcwd() )

	if (not hasGUI) :
		if imgFilePath == None :
			parser.print_help()
			parser.error("No GUI - must specify image filepath using -f ")

	addNoiseToImage(imgFilePath, args.level, args.spRatio)

	if hasGUI :
		displayImage = Image.open(imgFilePath)
		tkImage = ImageTk.PhotoImage(displayImage)
		tkinter.Label(tkWndw, image=tkImage).pack()
		tkWndw.mainloop()



def addNoiseToImage(imgFilePath: str, intensity: float, spRatio: float) :
	if os.path.isfile( imgFilePath ):
		with Image.open( imgFilePath ) as origImage:
			 
			if hasGUI :
				origImage.show()

			if origImage.mode != 'RGB' :
				origImage.convert("RGB")
			imBytes = origImage.tobytes()

			print( origImage.format, '%dx%d' % origImage.size, origImage.mode )

			npImg = numpy.asarray(origImage)

			pixelCount = origImage.size[0] * origImage.size[1]
			rPixels = imBytes[0: pixelCount * 3 : 3]
			gPixels = imBytes[1: pixelCount * 3 : 3]
			bPixels = imBytes[2: pixelCount * 3 : 3]


			if hasGUI and showChOrig :
				rImg = Image.frombytes( 'L', origImage.size, rPixels )
				gImg = Image.frombytes( 'L', origImage.size, gPixels )
				bImg = Image.frombytes( 'L', origImage.size, bPixels )
				rImg.show()
				gImg.show()
				bImg.show()
		
			# Add SnP noise to individual channels
			npNoiseArray = genNoiseSnP(npImg, intensity, spRatio)
			noiseImg = Image.fromarray(npNoiseArray)
			if hasGUI :
				noiseImg.show()

			if hasGUI and showChNoise :
				rImg = Image.frombytes( 'L', origImage.size, npNoiseArray[:, :, 0].tobytes() )
				gImg = Image.frombytes( 'L', origImage.size, npNoiseArray[:, :, 1].tobytes() )
				bImg = Image.frombytes( 'L', origImage.size, npNoiseArray[:, :, 2].tobytes() )
				rImg.show()
				gImg.show()
				bImg.show()

			# Infer output path
			fileDir = os.path.dirname(os.path.abspath(imgFilePath))
			oFileName = os.path.basename(imgFilePath) + ".bytes"
			oFilePath = fileDir + os.sep + oFileName
			print("Storing bytes to - " + oFilePath)

			# Store output .bytes format
			bytesOut = bytearray()

			bytesOut.extend((origImage.size[0]).to_bytes(4, byteorder='little'))	#	Width
			bytesOut.extend((origImage.size[1]).to_bytes(4, byteorder='little'))	#	Height
			bytesOut.extend((3).to_bytes(4, byteorder='little'))					#	Channels RGB = 3
			bytesOut.extend((1).to_bytes(4, byteorder='little'))					#	Pixel Size = 1 Byte
			bytesOut.extend(npNoiseArray.tobytes())
			print("%d bytes written" % len(bytesOut))

			with open(oFilePath, "wb") as bytesFile :
				bytesFile.write(bytesOut)

			print(" ... Done !")

	else:
		print("Couldn't find image at " + imgFilePath)



def genNoiseSnP(image: numpy.ndarray, intensity : float, s_vs_p : float) -> numpy.ndarray :
	"""
	Adds Salt & Pepper noise to a 2D RGB image packaged as a numpy ndarray

	Param
	-------
	image :	numpy.ndarray : object holding a 2D RGB image
	
	intensity : float : noise intensity in fraction

	s_vs_p : float : salt vs pepper ratio (default 0.5)
   
	Returns
	-------
	out : ndarray
		object with added SnP noise
	"""

	row, col, ch = image.shape
	print(image.shape)
	# s_vs_p = 0.5
	# intensity  = 0.04
	if s_vs_p == None :
		s_vs_p = s_vs_p_default
	if intensity == None :
		intensity = intensity_default

	threshold = int(intensity * 255)
	out = numpy.copy(image)
	
	# random_matrix = numpy.random.random(image.shape)      
	# out[random_matrix >= (1 - threshold)] 	= 255		#	upperValue
	# out[random_matrix <= threshold] 		= 0			#	lowerValue

	# Salt mode
	num_salt = numpy.ceil((s_vs_p * intensity) * image.size)
	coords = [numpy.random.randint(0, i, int(num_salt))
			for i in image.shape]
	out[tuple(coords)] = 1

	# Pepper mode
	num_pepper = numpy.ceil((intensity  * (1. - s_vs_p)) * image.size)
	coords = [numpy.random.randint(0, i, int(num_pepper))
			for i in image.shape]
	out[tuple(coords)] = 0
	return out

if __name__ == "__main__":
	main()
