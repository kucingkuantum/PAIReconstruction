import sys
#from PyQt5.QtCore import *
#from PyQt5.QtGui import QFileDialog
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.Qt import QMainWindow,qApp,  QTimer
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread, QThreadPool,pyqtSignal)
from PAI_UI import Ui_MainWindow
from os.path import basename, dirname
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import siregar_function as ims
from skimage.restoration import denoise_nl_means


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyMainWindow, self).__init__(parent)
        qApp.installEventFilter(self)
        self.setupUi(self)
        self.pushButton_open.clicked.connect(self.ope)
        self.pushButton_auto_f.clicked.connect(self.mpa)
        self.horizontalSlider_min.valueChanged.connect(self.valuechange1)
        self.horizontalSlider_min.valueChanged.connect(self.visual_touch)
        self.horizontalSlider_max.valueChanged.connect(self.valuechange2)
        self.horizontalSlider_max.valueChanged.connect(self.visual_touch)
        self.comboBox_4.currentIndexChanged.connect(self.visual_touch)      
        self.comboBoxCmaps.currentIndexChanged.connect(self.visual_touch) 
        self.comboBox_5.currentIndexChanged.connect(self.visual_touch)
        self.comboBox_5.currentIndexChanged.connect(self.comboboxchange)
        self.pushButton_denoising.clicked.connect(self.NLMD)
        self.pushButton_Run_pre.clicked.connect(self.BandPass)
        self.pushButton.clicked.connect(self.parameter_default)
        self.spinBox_slice.valueChanged.connect(self.visual_touch)
        self.pushButton_Expand.clicked.connect(self.Expand)
        
    def ope(self):       
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self,"Open Matrix File", "","Matrix File (*.dat)")#, options=options)
        if filename:
            self.Clear_all()
            name = basename(filename)
            self.lineEdit_filename.setText(name)
            x = re.findall('\\d+', name)
            self.lineEdit_size1.setText(x[4])
            self.lineEdit_size2.setText(x[2])
            self.lineEdit_size3.setText(x[3])
            self.readin(filename)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No file is loaded')
            msg.setInformativeText("Please select a file")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()
            pass


    def readin(self,filename):
        global raw
        
        with open(filename,'r') as f:
            raw=np.fromfile(f,np.uint16)
            raw = raw - np.mean(raw)
            self.init_vis(raw)
            self.Clear_combobox()
        f.close()
    
    
    def BandPass(self):
        rawbf = self.BandPass_process(raw)
        self.init_vis(rawbf)
        self.comboBox_4.setCurrentIndex(1)
            
        
    def BandPass_process(self,raw):
        global rawbf
        fl = float(self.lineEdit_flow.text())*1e6   #lower frequency
        fh = float(self.lineEdit_fhigh.text())*1e6   #higher frequency
        fs = float(self.lineEdit_fsampling.text())*1e6 #sampling frequency
        typefilt =int(self.comboBox_type.currentIndex())
        orde =  int(self.spinBox.value())
        if typefilt == 0:
            rawbf = ims.butter_bandpass_filter(raw,fl,fh,fs,order=orde)
        elif typefilt == 1:
            rawbf = ims.butter_bandpass_filter2(raw,fl,fh,fs,order=orde)
        
        self.comboBox_4.addItem('Band Pass Filtered Image')
        return rawbf
        

    
    def mpa(self):
        self.lineEdit_flow.setText(str(5))
        self.lineEdit_fhigh.setText(str(60))
        self.lineEdit_fsampling.setText(str(500))

    
    def reconstruct(self,d3p): 
        imagetype = int(self.comboBox_5.currentIndex())     
        if imagetype == 0:
            pp = ims.cmodemip(d3p)
            lum_img  = pp/np.max(pp)
        
        elif imagetype == 1:
            pp = ims.bmodemip(d3p)
            lum_img  = pp/np.max(pp)

        elif imagetype == 2:
            nslices = int(self.spinBox_slice.value())
            lum_img = d3p[:,:,nslices] 
            lum_img /= np.max(lum_img)
       
        return lum_img

    def Expand(self):
        nmin = float(self.lineEdit_min_scroll.text())
        nmax = float(self.lineEdit_max_scroll.text())
        plt.close(1)
        plt.imshow(expa,str(self.comboBoxCmaps.currentText()), clim=(nmin, nmax))
        plt.show()
        
    
    def init_vis(self,draw):
        n1 =int(self.lineEdit_size1.text())
        n2 =int(self.lineEdit_size2.text())
        n3 =int(self.lineEdit_size3.text())
        
        p = draw.reshape(n1,n2,n3,order='F')
        lum_img=self.reconstruct(p)
        self.visualization(lum_img)
    
    def visual_touch(self):
        
        if self.comboBox_4.currentText() == 'Raw Image':
            self.init_vis(raw)
        elif self.comboBox_4.currentText() == 'Band Pass Filtered Image':
            self.init_vis(rawbf)
        elif self.comboBox_4.currentText() == "Denoised Image":
            self.visualization(NLM)
        
        
    
    
    
    def visualization(self,d2image):
        global expa
        expa = d2image
        
        nmin = float(self.lineEdit_min_scroll.text())
        nmax = float(self.lineEdit_max_scroll.text())
        
        self.Plot_Visual.canvas.ax.clear()
        self.Plot_Visual.canvas.draw()
        self.Plot_Visual.canvas.ax.imshow(d2image,str(self.comboBoxCmaps.currentText()), clim=(nmin, nmax))
        self.Plot_Visual.canvas.fig.tight_layout()
        self.Plot_Visual.canvas.draw()    
        
    
        self.Plot_Histogram.canvas.ax.clear()
        self.Plot_Histogram.canvas.ax.hist(d2image.ravel(), bins=256, range=(nmin, nmax), fc='k', ec='k')
        self.Plot_Histogram.canvas.fig.tight_layout()
        self.Plot_Histogram.canvas.draw()
    
    def valuechange1(self):
        size1 = float(self.horizontalSlider_min.value())/1000
        self.lineEdit_min_scroll.setText(str(size1))
        
    def valuechange2(self):
        size2 = float(self.horizontalSlider_max.value())/1000
        self.lineEdit_max_scroll.setText(str(size2))
               
    def parameter_default(self):
        patch_size = int(self.comboBox_2.currentIndex())
        global patch
        self.mpa()
        if patch_size == 0:
            patch = 3
            self.spinBox_patch_distance.setValue((9))
            self.lineEdit_h.setText(str(0.5))
        if patch_size == 1:
            patch = 5
            self.spinBox_patch_distance.setValue((10))
            self.lineEdit_h.setText(str(0.35))            
        if patch_size == 2:
            patch = 7
            self.spinBox_patch_distance.setValue((10))
            self.lineEdit_h.setText(str(0.35))            
        if patch_size == 3:
            patch = 9
            self.spinBox_patch_distance.setValue((10))
            self.lineEdit_h.setText(str(0.35))            


    def NLMD(self):
        global NLM

        self.Clear_combobox()
        rawbf = self.BandPass_process(raw)
       
        n1 =int(self.lineEdit_size1.text())
        n2 =int(self.lineEdit_size2.text())
        n3 =int(self.lineEdit_size3.text())
        
        p = rawbf.reshape(n1,n2,n3,order='F')
        NLM = self.NLMD_cal(self.reconstruct(p))
        
        self.visualization(NLM)
        self.comboBox_4.addItem("Denoised Image")
        self.comboBox_4.setCurrentIndex(2)

    def NLMD_cal(self,d2image):

        patch_dist = self.spinBox_patch_distance.value()
        h = float(self.lineEdit_h.text())
        vari_raw = np.std(d2image)
        nlm = denoise_nl_means(d2image,patch,patch_dist,h*vari_raw, multichannel=True)
        nlm = nlm/np.max(nlm)
        return nlm


    def Clear_all(self):
        self.lineEdit_filename.setText('')
        self.lineEdit_size1.setText('')
        self.lineEdit_size2.setText('')
        self.lineEdit_size3.setText('')
#        self.lineEdit_flow.setText('')
#        self.lineEdit_fhigh.setText('')  
#        self.lineEdit_fsampling.setText('')
        
        self.Plot_Histogram.canvas.ax.clear()
        self.Plot_Histogram.canvas.draw()
        self.Plot_Visual.canvas.ax.clear()
        self.Plot_Visual.canvas.draw()

    def Clear_combobox(self):
        self.comboBox_4.clear()
        self.comboBox_4.addItem('Raw Image')
         
        
    
    def comboboxchange(self):
        combo=int(self.comboBox_5.currentIndex())
        #self.spinBox.setMinimum(1)n1 
        n1 = int(self.lineEdit_size1.text())
        self.spinBox_slice.setMaximum(n1)
        if combo == 2:
            self.label_19.setEnabled(True)
            self.spinBox_slice.setEnabled(True)
            
        else:
            self.label_19.setEnabled(False)
            self.spinBox_slice.setEnabled(False)  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())

    
    