import tkinter
from tkinter import *
from PIL import Image,ImageTk
from skimage import io, filters, color, exposure, feature,transform,morphology,segmentation
from matplotlib import pyplot as plt
import numpy as np
import cv2

root = Tk()
root.title("IP Project")

def exit():
    root.destroy()

def showImg():
    home.destroy()

    global img
    global grayScale
    global mainPic
    global showImg

    showImg=tkinter.Label(root)
    showImg.pack()

    frame1= tkinter.Frame(showImg)
    frame1.pack(side=tkinter.TOP)
    frame2 = tkinter.Frame(showImg)
    frame2.pack(side=tkinter.TOP)
    frame3 = tkinter.Frame(showImg)
    frame3.pack(side=tkinter.TOP)

    img = ImageTk.PhotoImage(Image.open(r'C:/Users/pcx/Desktop/x/ft.jpg'))
    sciImage=io.imread(r'C:/Users/pcx/Desktop/x/ft.jpg')
    grayScale=color.rgb2gray(sciImage)
    mainPic=tkinter.Label(frame1,image=img)
    mainPic.pack(side=tkinter.TOP)

    button1 = tkinter.Button(frame2, text='Filtreleme',command=filtering ,width=20)
    button1.pack(side=tkinter.LEFT)
    button2 = tkinter.Button(frame2, text='Histogram Eşitleme',command=histogramEsitleme, width=20)
    button2.pack(side=tkinter.LEFT)
    button3 = tkinter.Button(frame2, text='Uzaysal Dönüşüm',command=uzaysalDonusum, width=20)
    button3.pack(side=tkinter.LEFT)
    button4 = tkinter.Button(frame2, text='Yoğunluk Dönüşümü',command=yogunlukDonusumu, width=20)
    button4.pack(side=tkinter.LEFT)
    button5 = tkinter.Button(frame2, text='Morfolojik İşlemler',command=morfolojikDonusum, width=20)
    button5.pack(side=tkinter.LEFT)


    showImg.mainloop()

#filtreler

def sobelFilt():
    filtered = filters.sobel(grayScale)
    io.imshow(filtered)
    io.show()
    io.imsave('C:/Users/pcx/Desktop/x/sobel.jpg',filtered)

def gaborFilt():
    filt_real, filt_imag = filters.gabor(grayScale, frequency=0.6)
    io.imshow(filt_real)
    io.show()
    io.imsave('C:/Users/pcx/Desktop/x/gabor.jpg',filt_real)

def gaborKernelFilt():
    filtered=filters.gabor_kernel(frequency=0.3)
    io.imshow(filtered.real)
    io.show()

def robertsFilt():
    filtered = filters.roberts(grayScale)
    io.imshow(filtered.real)
    io.show()

def scharrFilt():
    filtered =filters.scharr(grayScale)
    io.imshow(filtered.real)
    io.show()

def scharr_vFilt():
    filtered = filters.scharr_v(grayScale)
    io.imshow(filtered.real)
    io.show()

def medianFilt():
    filtered=filters.median(grayScale)
    io.imshow(filtered)
    io.show()

def hessianFilt():
    filtered=filters.hessian(grayScale,mode='constant')
    io.imshow(filtered)
    io.show()

def laplaceFilt():
    filtered=filters.laplace(grayScale,ksize=5)
    io.imshow(filtered)
    io.show()

def meijeringFilt():
    filtered=filters.meijering(grayScale,mode='constant')
    io.imshow(filtered)
    io.show()

#filtreler bitiş

#Uzaysal Dönüşüm

def radonT():
    transformed=transform.radon(grayScale,preserve_range=True)
    io.imshow(transformed)
    io.show()

def resizeT():
    transformed=transform.resize(grayScale,(200,200))
    io.imshow(transformed)
    io.show()

def rotateT():
    transformed=transform.rotate(grayScale,40,resize=True)
    io.imshow(transformed)
    io.show()

def swirlT():
    transformed=transform.swirl(grayScale,strength=400)
    io.imshow(transformed)
    io.show()

def rescaleT():
    transformed=transform.rescale(grayScale,scale=0.25)
    io.imshow(transformed)
    io.show()

#Uzaysal Dönüşüm bitiş

#histogram
def histogramEsitleme():
    hist=exposure.equalize_hist(grayScale)
    io.imshow(hist)
    io.show()
#histogram bitiş

#morfolojik
def areaOpeningMr():
    aOpened=morphology.area_opening(grayScale,connectivity=1, tree_traverser=None)
    io.imshow(aOpened)
    io.show()

def areaClosingMr():
    aClosed=morphology.area_closing(grayScale,area_threshold=32)
    io.imshow(aClosed)
    io.show()

def openingMr():
    op=morphology.opening(grayScale)
    io.imshow(op)
    io.show()

def closingMr():
    cl=morphology.closing(grayScale)
    io.imshow(cl)
    io.show()

def whiteTopHatMr():
    wth=morphology.white_tophat(grayScale)
    io.imshow(wth)
    io.show()


def blackTopHatMr():
    bth =morphology.black_tophat(grayScale)
    io.imshow(bth)
    io.show()


def diameterOpeningMr():
    dio =morphology.diameter_opening(grayScale, diameter_threshold=4, connectivity=1)
    io.imshow(dio)
    io.show()

def diameterClosingMr():
    dic =morphology.diameter_closing(grayScale, diameter_threshold=2)
    io.imshow(dic)
    io.show()

def  dilationMr():
    dl=morphology.dilation(grayScale,shift_x=True)
    io.imshow(dl)
    io.show()

def  erosionMr():
    er=morphology.erosion(grayScale,shift_x=True)
    io.imshow(er)
    io.show()



#morfolojik bitiş

#active contour

def activeContour():
    sciImage = io.imread(r'C:/Users/pcx/Desktop/x/ft.jpg')
    grayScale = color.rgb2gray(sciImage)

    s = np.linspace(0, 2 * np.pi, 400)
    x = 400 + 150 * np.cos(s)
    y = 165 + 150 * np.sin(s)
    init = np.array([x, y]).T

    contour = segmentation.active_contour(filters.gaussian(grayScale, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(grayScale)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(contour[:, 0], contour[:, 1], '-b', lw=3)

    plt.show()

#active contour bitiş

#video işleme

def detectingEdges():

    cap = cv2.VideoCapture('C:/Users/pcx/Desktop/x/videoplayback.mp4')

    while (1):
        ret, frame = cap.read()
        gray_vid = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Original', frame)
        edged_frame = cv2.Canny(frame, 100, 200)
        cv2.imshow('Edges', edged_frame)
        # ESC tuşu ile çıkılır
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

#video işleme bitiş

#sosyal medya filtresi

def newFilter():
    sciImage = io.imread(r'C:/Users/pcx/Desktop/x/ft.jpg')
    grayScale = color.rgb2gray(sciImage)

    first=filters.median(grayScale)
    second=transform.swirl(first,strength=10)
    last=exposure.equalize_hist(second)

    io.imshow(last)
    io.show()

#sosyal medya filtresi bitiş


def filtering():
    showImg.destroy()
    filtering=tkinter.Label(root)
    filtering.pack()

    frame1= tkinter.Frame(filtering)
    frame1.pack(side=tkinter.TOP)
    frame2 = tkinter.Frame(filtering)
    frame2.pack(side=tkinter.TOP)
    frame3 = tkinter.Frame(filtering)
    frame3.pack(side=tkinter.TOP)

    label = tkinter.Label(frame1,text='Filtreleme Menüsü')
    label.pack(side=tkinter.TOP)

    mainPic = tkinter.Label(frame1, image=img)
    mainPic.pack(side=tkinter.TOP)

    button1 = tkinter.Button(frame2, text='Sobel',command=sobelFilt ,width=20)
    button1.pack(side=tkinter.LEFT)
    button2 = tkinter.Button(frame2, text='Gabor ',command=gaborFilt , width=20)
    button2.pack(side=tkinter.LEFT)
    button3 = tkinter.Button(frame2, text='Gabor Kernel',command=gaborKernelFilt , width=20)
    button3.pack(side=tkinter.LEFT)
    button4 = tkinter.Button(frame2, text='Roberts', command=robertsFilt ,width=20)
    button4.pack(side=tkinter.LEFT)
    button5 = tkinter.Button(frame2, text='Scharr',command=scharrFilt , width=20)
    button5.pack(side=tkinter.LEFT)
    button6= tkinter.Button(frame2, text='Scharr_v',command=scharr_vFilt , width=20)
    button6.pack(side=tkinter.LEFT)
    button7 = tkinter.Button(frame2, text='Median',command=medianFilt , width=20)
    button7.pack(side=tkinter.LEFT)
    button8 = tkinter.Button(frame2, text='Hessian',command=hessianFilt , width=20)
    button8.pack(side=tkinter.LEFT)
    button9 = tkinter.Button(frame2, text='Laplace',command=laplaceFilt , width=20)
    button9.pack(side=tkinter.LEFT)
    button10 = tkinter.Button(frame2, text='Meijering',command=meijeringFilt , width=20)
    button10.pack(side=tkinter.LEFT)

    filtering.mainloop()

def uzaysalDonusum():
    showImg.destroy()

    uzaysalDonusum=tkinter.Label(root)
    uzaysalDonusum.pack()

    frame1= tkinter.Frame(uzaysalDonusum)
    frame1.pack(side=tkinter.TOP)
    frame2 = tkinter.Frame(uzaysalDonusum)
    frame2.pack(side=tkinter.TOP)
    frame3 = tkinter.Frame(uzaysalDonusum)
    frame3.pack(side=tkinter.TOP)

    label = tkinter.Label(frame1, text='Uzaysal Dönüşüm Menüsü')
    label.pack(side=tkinter.TOP)

    mainPic = tkinter.Label(frame1, image=img)
    mainPic.pack(side=tkinter.TOP)

    button1 = tkinter.Button(frame2, text='Radon',command=radonT ,width=20)
    button1.pack(side=tkinter.LEFT)
    button2 = tkinter.Button(frame2, text='Resize ',command=resizeT, width=20)
    button2.pack(side=tkinter.LEFT)
    button3 = tkinter.Button(frame2, text='Rotate',command=rotateT , width=20)
    button3.pack(side=tkinter.LEFT)
    button4 = tkinter.Button(frame2, text='Swirl', command=swirlT ,width=20)
    button4.pack(side=tkinter.LEFT)
    button5 = tkinter.Button(frame2, text='Rescale',command=rescaleT , width=20)
    button5.pack(side=tkinter.LEFT)

    uzaysalDonusum.mainloop()



def yogunlukDonusumu():
    showImg.destroy()

    yogunlukDonusumu = tkinter.Label(root)
    yogunlukDonusumu.pack()

    frame1 = tkinter.Frame(yogunlukDonusumu)
    frame1.pack(side=tkinter.TOP)
    frame2 = tkinter.Frame(yogunlukDonusumu)
    frame2.pack(side=tkinter.TOP)
    frame3 = tkinter.Frame(yogunlukDonusumu)
    frame3.pack(side=tkinter.TOP)

    label = tkinter.Label(frame1, text='Yoğunluk Dönüşümü Menüsü')
    label.pack(side=tkinter.TOP)

    inputLabel1=tkinter.Label(frame2,text='Birinci değer,')
    inputLabel1.pack(side=tkinter.LEFT)
    inputEntry1 = tkinter.Entry(frame2, bd=5)
    inputEntry1.pack(side=tkinter.RIGHT)
    inputLabel2 = tkinter.Label(frame2, text='İkinci değer: ')
    inputLabel2.pack(side=tkinter.LEFT)
    inputEntry2 = tkinter.Entry(frame2, bd=5)
    inputEntry2.pack(side=tkinter.RIGHT)

    def press():
        res=exposure.rescale_intensity(grayScale,out_range=(int(inputEntry1.get()), int(inputEntry2.get())))
        io.imshow(res)
        io.show()

    button = tkinter.Button(frame3, text='OK', command=press, width=20)
    button.pack(side=tkinter.TOP)

    yogunlukDonusumu.mainloop()

def morfolojikDonusum():
    showImg.destroy()
    morfolojikDonusum = tkinter.Label(root)
    morfolojikDonusum.pack()

    frame1 = tkinter.Frame(morfolojikDonusum)
    frame1.pack(side=tkinter.TOP)
    frame2 = tkinter.Frame(morfolojikDonusum)
    frame2.pack(side=tkinter.TOP)
    frame3 = tkinter.Frame(morfolojikDonusum)
    frame3.pack(side=tkinter.TOP)

    label = tkinter.Label(frame1, text='Morfolojik Dönüşüm Menüsü')
    label.pack(side=tkinter.TOP)

    mainPic = tkinter.Label(frame1, image=img)
    mainPic.pack(side=tkinter.TOP)

    button1 = tkinter.Button(frame2, text='Area Opening', command=areaOpeningMr, width=20)
    button1.pack(side=tkinter.LEFT)
    button2 = tkinter.Button(frame2, text='Area Closing', command=areaClosingMr, width=20)
    button2.pack(side=tkinter.LEFT)
    button3 = tkinter.Button(frame2, text='Opening', command=openingMr, width=20)
    button3.pack(side=tkinter.LEFT)
    button4 = tkinter.Button(frame2, text='Closing', command=closingMr, width=20)
    button4.pack(side=tkinter.LEFT)
    button5 = tkinter.Button(frame2, text='White Top Hat', command=whiteTopHatMr, width=20)
    button5.pack(side=tkinter.LEFT)
    button6 = tkinter.Button(frame2, text='Black Top Hat', command=blackTopHatMr, width=20)
    button6.pack(side=tkinter.LEFT)
    button7 = tkinter.Button(frame2, text='Diameter Opening', command=diameterOpeningMr, width=20)
    button7.pack(side=tkinter.LEFT)
    button8 = tkinter.Button(frame2, text='Diameter Closing', command=diameterClosingMr, width=20)
    button8.pack(side=tkinter.LEFT)
    button9 = tkinter.Button(frame2, text='Dilation', command=dilationMr, width=20)
    button9.pack(side=tkinter.LEFT)
    button10 = tkinter.Button(frame2, text='Erosion', command=erosionMr, width=20)
    button10.pack(side=tkinter.LEFT)

    morfolojikDonusum.mainloop()
def home():
    global home


    home = tkinter.Label(root)
    home.pack()

    frame1 = tkinter.Frame(home)
    frame1.pack(side=tkinter.TOP)
    frame2 = tkinter.Frame(home)
    frame2.pack(side=tkinter.TOP)

    label=tkinter.Label(frame1, text='IP Project')
    label.pack(side=tkinter.LEFT)

    button1 = tkinter.Button(frame2, text='Resim Aç ', command=showImg, width=50)
    button1.pack(side=tkinter.TOP)
    button2 = tkinter.Button(frame2, text='Active Contour nedir ?',command=activeContour, width=50)
    button2.pack(side=tkinter.TOP)
    button3 = tkinter.Button(frame2, text='Video İşleme',command=detectingEdges, width=50)
    button3.pack(side=tkinter.TOP)
    button4 = tkinter.Button(frame2, text='Sosyal Medya Filtresi',command=newFilter, width=50)
    button4.pack(side=tkinter.TOP)
    button5 = tkinter.Button(frame2, text='Çıkış',command=exit, width=50)
    button5.pack(side=tkinter.TOP)

home()



root.mainloop()
