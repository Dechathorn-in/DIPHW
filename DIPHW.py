import cv2 as cv
import numpy as np
import math
from scipy import ndimage
from matplotlib import pyplot as plt
from numpy.fft.helper import ifftshift
from numpy.lib.function_base import angle, meshgrid

#1.1 ----------------------------------------------
img = cv.imread("Cross.pgm",0)
image_pad = cv.copyMakeBorder(img, 28, 28, 28, 28, cv.BORDER_CONSTANT)

dft = np.fft.fft2(image_pad)
dft_shift = np.fft.fftshift(dft)
phase_spectrum = angle(dft_shift)
amplitude = 20*np.log(np.abs(dft_shift))

plt.subplot(121)
plt.imshow(amplitude, cmap='gray')
plt.title('Amplitude'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase spectrum'), plt.xticks([]), plt.yticks([])

plt.show()

#1.2 ----------------------------------------------
nx, ny = (256,256)
X = np.linspace(-128,127,nx)
Y = np.linspace(-128,127,ny)
xv, yv = np.meshgrid(X, Y)
a = phase_spectrum
complex_number = np.exp(-2j*np.pi*(((20*xv)/256)+((30*yv)/256)))
new_phase = a * complex_number
ifftshifted = ifftshift(new_phase)
img_ifft = np.fft.fft2(ifftshifted)
img_ifft = abs(img_ifft)

plt.subplot(121),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(img_ifft, cmap='gray')
plt.title("inverse fourier transform"), plt.xticks([]), plt.yticks([])

plt.show()

#1.3 ----------------------------------------------

# rotated = ndimage.rotate(img, 30)
# dft = np.fft.fft2(rotated)
# dft_shift = np.fft.fftshift(dft)
# phase_spectrum = angle(dft_shift)
# amplitude = 20*np.log(np.abs(dft_shift))

# cv.imshow("Image Rotate", rotated)
# cv.waitKey(0)

# plt.subplot(121)
# plt.imshow(amplitude, cmap='gray')
# plt.title('Amplitude Rotate'), plt.xticks([]), plt.yticks([])

# plt.subplot(122),plt.imshow(phase_spectrum, cmap = 'gray')
# plt.title('Phase spectrum Rotate'), plt.xticks([]), plt.yticks([])

# plt.show()

# #1.4 ----------------------------------------------

# img_downsample = cv.resize(img, (100,100))
# image_pad_downsample = cv.copyMakeBorder(img_downsample , 14, 14, 14, 14, cv.BORDER_CONSTANT)

# dft = np.fft.fft2(image_pad_downsample)
# dft_shift = np.fft.fftshift(dft)
# phase_spectrum = angle(dft_shift)
# amplitude = 20*np.log(np.abs(dft_shift))

# plt.subplot(121)
# plt.imshow(amplitude, cmap='gray')
# plt.title('Amplitude Down'), plt.xticks([]), plt.yticks([])

# plt.subplot(122),plt.imshow(phase_spectrum, cmap = 'gray')
# plt.title('Phase spectrum Down'), plt.xticks([]), plt.yticks([])

# plt.show()

# #1.5 ----------------------------------------------

# inverse_amp = np.fft.ifftshift(amplitude)
# ifft_amp = np.fft.ifft2(inverse_amp)

# inverse_phase = np.fft.ifftshift(phase_spectrum)
# inverse_phase = np.fft.ifft2(inverse_phase)
# inverse_phase = abs(inverse_phase)


# plt.subplot(121),plt.imshow(inverse_amp , cmap='gray')
# plt.title('Amplitude Inverse FFT'), plt.xticks([]), plt.yticks([])

# plt.subplot(122),plt.imshow(inverse_phase , cmap='gray')
# plt.title('Phase Inverse FFT'), plt.xticks([]), plt.yticks([])
# plt.show()

# #1.6 ----------------------------------------------

# img_lenna = cv.imread("Lenna.pgm",0)
# dft = np.fft.fft2(img_lenna)
# dft_shift = np.fft.fftshift(dft)
# phase_spectrum = angle(dft_shift)
# amplitude = np.log(np.abs(dft_shift))

# inverse_amp = np.fft.ifftshift(amplitude)
# ifft_amp = np.fft.ifft2(inverse_amp)

# plt.figure
# plt.subplot(121),plt.imshow(inverse_amp , cmap='gray')
# plt.title('Lenna Amplitude Inverse FFT'), plt.xticks([]), plt.yticks([])

# inverse_phase = np.fft.ifftshift(phase_spectrum)
# inverse_phase = np.fft.ifft2(inverse_phase)
# inverse_phase = abs(inverse_phase)


# plt.subplot(122),plt.imshow(inverse_phase , cmap='gray')
# plt.title('Lenna Phase Inverse FFT'), plt.xticks([]), plt.yticks([])
# plt.show()

# #1.7 ----------------------------------------------

# img_chess = cv.imread("Chess.pgm",0)
# kernel = np.ones((3, 3), np.float32) / 9

# img_kernel_blur = cv.filter2D(img_chess, -1, kernel)


# img_chess_fft = np.fft.fft2(img_kernel_blur)
# shifted_img_chess_fft = np.fft.fftshift(img_chess_fft)

# pad_kernel = cv.copyMakeBorder(kernel, 127, 126, 127, 126, cv.BORDER_CONSTANT)
# kernel_fft = np.fft.fft2(pad_kernel)
# shifted_kernel_fft = np.fft.fftshift(kernel_fft)

# img_chess_fillter = shifted_kernel_fft  * shifted_img_chess_fft 
# img_blur_ifft = np.fft.ifftshift(img_chess_fillter)
# img_blur_ifft = np.fft.ifft2(img_blur_ifft)
# img_blur_ifft = np.abs(img_blur_ifft)


# plt.subplot(132),plt.imshow(img_kernel_blur , cmap='gray')
# plt.title('Convolution'), plt.xticks([]), plt.yticks([])

# plt.subplot(133),plt.imshow(img_blur_ifft , cmap='gray')
# plt.title('Frequency domain'), plt.xticks([]), plt.yticks([])

# plt.subplot(131),plt.imshow(img_chess , cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.show()





