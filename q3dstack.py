import javabridge
import bioformats
import glob
import argparse
import numpy as np
import sys
import vtk
from vtk.util.vtkImageExportToArray import vtkImageExportToArray
import wx
from wx.glcanvas import GLCanvas
import scipy.ndimage

typeEVT_PICK = wx.NewEventType()
EVT_PICK = wx.PyEventBinder(typeEVT_PICK, 1)

class PickEvent(wx.PyEvent):
    def __init__(self, mouse_event, x, y, z, actor):
        assert isinstance(mouse_event, wx.MouseEvent)
        super(PickEvent, self).__init__(mouse_event.GetId(), typeEVT_PICK)
        self.__mouse_event = mouse_event
        self.__x = x
        self.__y = y
        self.__z = z
        self.__actor = actor
        
    @property
    def mouse_event(self):
        return self.__mouse_event
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def z(self):
        return self.__z
    
    @property
    def actor(self):
        return self.__actor
    
class Volume(object):
    def __init__(self, img, color):
        self.volume = vtk.vtkVolume()
        self.__color = color
        dataImporter = vtk.vtkImageImport()
        simg = np.ascontiguousarray(img, np.uint8)
        dataImporter.CopyImportVoidPointer(simg.data, len(simg.data))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
        dataImporter.SetWholeExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
        self.__smoother = vtk.vtkImageGaussianSmooth()
        self.__smoother.SetStandardDeviation(1, 1, 1)
        self.__smoother.SetInputConnection(dataImporter.GetOutputPort())
        volumeMapper = vtk.vtkSmartVolumeMapper()
        volumeMapper.SetInputConnection(self.__smoother.GetOutputPort())
        self.__volumeProperty = vtk.vtkVolumeProperty()
        self.__colorFunc = vtk.vtkColorTransferFunction()
        self.__alpha = vtk.vtkPiecewiseFunction()
        for i in range(256):
            self.__colorFunc.AddRGBPoint(i, i * color[0], i * color[1], i * color[2])
        self.__alpha.AddPoint(5, .01)
        self.__alpha.AddPoint(10, .03)
        self.__alpha.AddPoint(50, .1)
        self.__alpha.AddPoint(150, .2)
        self.__volumeProperty.SetColor(self.__colorFunc)
        self.__volumeProperty.SetScalarOpacity(self.__alpha)
        self.volume.SetMapper(volumeMapper)
        self.volume.SetProperty(self.__volumeProperty)
        
    def set_smoothing(self, xysigma, zsigma):
        self.__smoother.SetStandardDeviation(xysigma, xysigma, zsigma)
        
    def set_color_func(self, values):
        '''Set the colors for all 256 intensity levels
        
        values - an integer numpy array of 256 rows and 3 color values
                 between 0 and 255
        '''
        self.__colorFunc = vtk.vtkColorTransferFunction()
        for i in range(256):
            self.__colorFunc.AddRGBPoint(i, *list(values[i]))
        self.__volumeProperty.SetColor(self.__colorFunc)
        self.volume.SetProperty(self.__volumeProperty)
        self.volume.Update()
        
    def set_alpha(self, values):
        '''Set the transparency for all 256 intensity levels
        
        values - an nx2 array of rows of intensity and opacity
        '''
        self.__alpha = vtk.vtkPiecewiseFunction()
        if values[0, 0] != 0:
            self.__alpha.AddPoint(0, values[0][1])
        for intensity, opacity in values:
            self.__alpha.AddPoint(intensity, opacity)
        if values[-1, 0] != 255:
            self.__alpha.AddPoint(255, values[-1, 1])
        self.__volumeProperty.SetScalarOpacity(self.__alpha)
        self.volume.SetProperty(self.__volumeProperty)
        self.volume.Update()
        
    def use_linear_intensity(self):
        values = np.column_stack([np.arange(256)]*3) * \
            np.array(self.__color)[np.newaxis, :]
        self.set_color_func(values)
        
    def use_log_intensity(self):
        values = (np.log(np.arange(1, 257)) / np.log(256)).astype(int)
        values = np.column_stack([values]*3) * \
            np.array(self.__color)[np.newaxis, :]
        self.set_color_func(values)
        
class Contour(vtk.vtkActor):
    def __init__(self, img, threshold, xysmoothing, zsmoothing, color, alpha):
        dataImporter = vtk.vtkImageImport()
        simg = np.ascontiguousarray(img, np.uint8)
        dataImporter.CopyImportVoidPointer(simg.data, len(simg.data))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
        dataImporter.SetWholeExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
        self.__smoother = vtk.vtkImageGaussianSmooth()
        self.__smoother.SetStandardDeviation(xysmoothing, xysmoothing, zsmoothing)
        self.__smoother.SetInputConnection(dataImporter.GetOutputPort())
        self.__threshold = vtk.vtkImageThreshold()
        self.__threshold.SetInputConnection(self.__smoother.GetOutputPort())
        self.__threshold.ThresholdByUpper(threshold)
        self.__threshold.ReplaceInOn()
        self.__threshold.SetInValue(1)
        self.__threshold.ReplaceOutOn()
        self.__threshold.SetOutValue(0)
        self.__threshold.Update()
        contour = vtk.vtkDiscreteMarchingCubes()
        contour.SetInputConnection(self.__threshold.GetOutputPort())
        contour.ComputeNormalsOn()
        contour.SetValue(0, 1)
        contour.Update()
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        triangleCellNormals=vtk.vtkPolyDataNormals()
        triangleCellNormals.SetInputConnection(smoother.GetOutputPort())
        triangleCellNormals.ComputeCellNormalsOn()
        triangleCellNormals.ComputePointNormalsOff()
        triangleCellNormals.ConsistencyOn()
        triangleCellNormals.AutoOrientNormalsOn()
        triangleCellNormals.Update()
        triangleCellAn = vtk.vtkMeshQuality()
        triangleCellAn.SetInputConnection(triangleCellNormals.GetOutputPort())
        triangleCellAn.SetTriangleQualityMeasureToArea()
        triangleCellAn.SaveCellQualityOn()
        triangleCellAn.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(triangleCellAn.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(.3, 1)
        mapper.SetScalarModeToUsePointData()
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetHueRange(.6, 1)
        colorLookupTable.Build()
        mapper.SetLookupTable(colorLookupTable)
        self.SetMapper(mapper)
        self.GetProperty().SetColor(*color)
        self.GetProperty().SetOpacity(alpha)
        
    def set_threshold(self, threshold, update= True):
        self.__threshold.ThresholdByUpper(threshold)
        if update:
            self.GetMapper().Update()
        
    def set_opacity(self, opacity, update= True):
        self.GetProperty().SetOpacity(opacity)
        if update:
            self.GetMapper().Update()
        
    def set_smoothing(self, xysmoothing, zsmoothing, update= True):
        self.__smoother.SetStandardDeviation(zsmoothing, xysmoothing, xysmoothing)
        if update:
            self.GetMapper().Update()
            
    def get_threshold(self):
        converter = vtkImageExportToArray()
        converter.SetInputConnection(self.__threshold.GetOutputPort())
        return converter.GetArray()
        
        
###########################
# 
# VTKCanvas - some ideas borrowed from 
#             https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtk/wx/wxVTKRenderWindow.py
#
###########################
class VTKCanvas(GLCanvas):
    def __init__(self, *args, **kwargs):
        super(VTKCanvas, self).__init__(*args, **kwargs)
        self.__render_window = vtk.vtkRenderWindow()
        self.__render_window.SetNextWindowInfo(
            str(self.GetHandle()))
        self.__render_window.WindowRemap()
        self.__renderer = vtk.vtkRenderer()
        self.__render_window.AddRenderer(self.__renderer)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda event: None)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.on_mouse_event)
        
    def on_size(self, event):
        assert isinstance(event, wx.SizeEvent)
        self.__render_window.SetSize(event.GetSize()[0],
                                     event.GetSize()[1])
        self.__render_window.Render()
    
    def on_paint(self, event):
        dc = wx.PaintDC(self)
        self.__render_window.Render()
        
    def on_mouse_event(self, event):
        assert isinstance(event, wx.MouseEvent)
        mouseX, mouseY = event.GetPosition()
        window_y = self.render_window.GetSize()[1]
        picker = vtk.vtkPicker()
        has_something = picker.Pick(mouseX, window_y - mouseY, 0, self.renderer)
        p3ds = picker.GetProp3Ds()
        p3ds = [
            p3ds.GetItemAsObject(idx) 
            for idx in range(p3ds.GetNumberOfItems())]
        points = picker.GetPickedPositions()
        points = [
            points.GetPoint(idx) 
            for idx in range(points.GetNumberOfPoints())]
        for actor, (x, y, z) in zip(p3ds, points):
            pick_event = PickEvent(event, x, y, z, actor)
            self.GetEventHandler().ProcessEvent(pick_event)
        
    @property
    def render_window(self):
        return self.__render_window
    
    @property
    def renderer(self):
        return self.__renderer
    
    @property
    def camera(self):
        return self.renderer.GetActiveCamera()
    
class ContourPanel(wx.Panel):
    def __init__(self, canvas, img, label, color, *args, **kwargs):
        super(ContourPanel, self).__init__(*args, **kwargs)
        self.__img = img
        self.__label = label
        self.__color = color
        self.__canvas = canvas
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__checkbox = wx.CheckBox(self, label = label)
        self.Sizer.Add(self.__checkbox, 0, wx.EXPAND)
        self.Sizer.AddSpacer(2)
        self.Sizer.Add(
            wx.StaticText(self, label = "Threshold"), 0, wx.ALIGN_RIGHT)
        self.Sizer.AddSpacer(2)
        self.__threshold = wx.SpinCtrl(self, min=0, max=255, initial=10)
        self.Sizer.Add(self.__threshold, 0, wx.EXPAND)
        self.Sizer.AddSpacer(2)
        self.Sizer.Add(
            wx.StaticText(self, label = "Opacity"), 0, wx.ALIGN_RIGHT)
        self.__opacity = wx.SpinCtrl(self, min = 0, max=255, initial=128)
        self.Sizer.AddSpacer(2)
        self.Sizer.Add(self.__opacity, 0, wx.EXPAND)
        self.Sizer.AddSpacer(2)
        self.Sizer.Add(
            wx.StaticText(self, label = "XYSmoothing"), 0, wx.ALIGN_RIGHT)
        self.__xysmoothing = wx.SpinCtrl(self, min = 1, max=100, initial=1)
        self.Sizer.Add(self.__xysmoothing, 0, wx.EXPAND)
        self.Sizer.AddSpacer(2)
        self.Sizer.Add(
            wx.StaticText(self, label = "ZSmoothing"), 0, wx.ALIGN_RIGHT)
        self.__zsmoothing = wx.SpinCtrl(self, min = 1, max=100, initial=1)
        self.Sizer.Add(self.__zsmoothing, 0, wx.EXPAND)
        self.Sizer.AddSpacer(2)
        for ctrl in self.__threshold, self.__opacity, self.__xysmoothing, self.__zsmoothing:
            ctrl.Bind(wx.EVT_SPINCTRL, self.on_change)
        self.__checkbox.Bind(wx.EVT_CHECKBOX, self.on_check)
        self.__actor = None
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Layout()

    def on_size(self, event):
        self.Layout()
        
    def on_change(self, event):
        if self.__checkbox.IsChecked():
            threshold = self.__threshold.Value
            opacity = self.__opacity.Value
            xysmoothing = self.__xysmoothing.Value
            zsmoothing = self.__zsmoothing.Value
            msg = "Threshold = %d, x/y smoothing = %d, z smoothing = %d" %\
                (threshold, xysmoothing, zsmoothing)
            with wx.ProgressDialog(
                "Segmenting %s" % self.__label,
                msg,
                parent = self) as dlg:
                dlg.Pulse(msg)
                if self.__actor is None:
                    self.__actor = Contour(
                        self.__img, threshold, xysmoothing, zsmoothing,
                        self.__color, opacity)
                    self.__canvas.renderer.AddActor(self.__actor)
                else:
                    self.__actor.set_threshold(threshold, update=False)
                    self.__actor.set_smoothing(xysmoothing, zsmoothing, 
                                               update=False)
                    self.__actor.set_opacity(opacity)
            self.__canvas.render_window.Render()
            
    def on_check(self, event):
        if self.__checkbox.IsChecked():
            self.on_change(event)
        elif self.__actor is not None:
            self.__canvas.renderer.RemoveActor(self.__actor)
            self.__actor = None
            
    def is_my_actor(self, actor):
        return self.__actor == actor
        
    @property
    def threshold(self):
        return self.__threshold.Value
    
    @property
    def xysmoothing(self):
        return self.__xysmoothing.Value
    
    @property
    def zsmoothing(self):
        return self.__zsmoothing.Value
    
    @property
    def enabled(self):
        return self.__checkbox.IsChecked()
    
    @property
    def contour(self):
        return self.__actor
        
class Q3DFrame(wx.Frame):
    def __init__(self, imgRed, imgGreen, imgBlue, *args, **kwargs):
        super(Q3DFrame, self).__init__(*args, **kwargs)
        self.imgRed = imgRed
        self.imgGreen = imgGreen
        self.imgBlue = imgBlue
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas_panel = wx.Panel(self)
        self.Sizer.Add(self.canvas_panel, 1, wx.EXPAND)
        self.canvas_panel.Sizer = wx.BoxSizer()
        self.canvas = VTKCanvas(
            self.canvas_panel, 
            style = wx.WANTS_CHARS | wx.NO_FULL_REPAINT_ON_RESIZE)
        self.canvas_panel.Sizer.Add(self.canvas, 1, wx.EXPAND)
        
        self.volumeRed = Volume(imgRed, (1., 0, 0))
        self.volumeGreen = Volume(imgGreen, (0, 1., 0))
        self.volumeBlue = Volume(imgBlue, (0, 0, 1.))
        self.canvas.renderer.AddVolume(self.volumeRed.volume)
        self.canvas.renderer.AddVolume(self.volumeGreen.volume)
        self.canvas.renderer.AddVolume(self.volumeBlue.volume)
        
        self.redContour = ContourPanel(
            self.canvas, imgRed, "Red contour", (255, 0, 0), self)
        self.Sizer.Add(self.redContour, 0, wx.EXPAND)
        self.greenContour = ContourPanel(
            self.canvas, imgGreen, "Green contour", (0, 255, 0), self)
        self.Sizer.Add(self.greenContour, 0, wx.EXPAND)
        self.blueContour = ContourPanel(
            self.canvas, imgBlue, "Blue contour", (0, 0, 255), self)
        self.Sizer.Add(self.blueContour, 0, wx.EXPAND)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(button_sizer, 0, wx.EXPAND)
        roll = wx.Button(self, label="Roll")
        button_sizer.Add(roll, 1, wx.EXPAND)
        button_sizer.AddSpacer(2)
        def on_roll(event):
            self.canvas.camera.Roll(5)
            self.canvas.renderer.ResetCamera()
            self.canvas.render_window.Render()
        roll.Bind(wx.EVT_BUTTON, on_roll)
        pitch = wx.Button(self, label="Pitch")
        button_sizer.Add(pitch, 1, wx.EXPAND)
        button_sizer.AddSpacer(2)
        def on_pitch(event):
            pitch = -5 if wx.GetKeyState(wx.ACCEL_SHIFT) else 5
            self.canvas.camera.Pitch(pitch)
            self.canvas.renderer.ResetCamera()
            self.canvas.render_window.Render()
        pitch.Bind(wx.EVT_BUTTON, on_pitch)
        yaw = wx.Button(self, label="Yaw")
        button_sizer.Add(yaw, 1, wx.EXPAND)
        button_sizer.AddSpacer(2)
        def on_yaw(event):
            yaw = -5 if wx.GetKeyState(wx.ACCEL_SHIFT) else 5
            self.canvas.camera.Yaw(yaw)
            self.canvas.renderer.ResetCamera()
            self.canvas.render_window.Render()
        yaw.Bind(wx.EVT_BUTTON, on_yaw)
        zoom_in = wx.Button(self, label="+")
        button_sizer.Add(zoom_in, 1, wx.EXPAND)
        button_sizer.AddSpacer(2)
        def on_zoom_in(event):
            self.canvas.camera.Zoom(1.333)
            self.canvas.render_window.Render()
        zoom_in.Bind(wx.EVT_BUTTON, on_zoom_in)
        zoom_out = wx.Button(self, label="-")
        button_sizer.Add(zoom_out, 1, wx.EXPAND)
        button_sizer.AddSpacer(2)
        def on_zoom_out(event):
            self.canvas.camera.Zoom(.75)
            self.canvas.render_window.Render()
        zoom_out.Bind(wx.EVT_BUTTON, on_zoom_out)
        for label, xinc, yinc in (
            ("<", -10, 0),
            ("^", 0, -10),
            ("v", 0, 10),
            (">", 10, 0)):
            pan = wx.Button(self, label=label,
                            style = wx.BU_EXACTFIT)
            button_sizer.Add(pan, 0, wx.ALIGN_CENTER)
            button_sizer.AddSpacer(2)
            pan.Bind(
                wx.EVT_BUTTON, 
                lambda e, xinc=xinc, yinc=yinc: self.on_pan(xinc, yinc))
        
        reset =  wx.Button(self, label="Reset")
        button_sizer.Add(reset, 1, wx.EXPAND)
        reset.Bind(wx.EVT_BUTTON, self.set_camera_back_to_square_one)
        self.set_camera_back_to_square_one()
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.canvas.Bind(EVT_PICK, self.on_pick)
        self.redContour.Layout()
        self.greenContour.Layout()
        self.blueContour.Layout()
        self.Layout()
        self.SetAutoLayout(True)
        self.redContour.SetAutoLayout(True)
        self.greenContour.SetAutoLayout(True)
        self.blueContour.SetAutoLayout(True)
        self.Show()
        
    def set_camera_back_to_square_one(self, event=None):
        self.canvas.camera.SetFocalPoint(self.imgRed.shape[2],
                                         self.imgRed.shape[1] / 2,
                                         self.imgRed.shape[0] / 2)
        self.canvas.camera.SetPosition(self.imgRed.shape[0],
                                       self.imgRed.shape[1] / 2,
                                       self.imgRed.shape[0] / 2)
        self.canvas.renderer.ResetCamera()
        if event is not None:
            self.canvas.render_window.Render()
        
    def on_pan(self, x, y):
        normal = np.array(self.canvas.camera.GetViewPlaneNormal())
        up = np.array(self.canvas.camera.GetViewUp())
        cross = np.cross(up, normal)
        delta = up * y + cross * x
        fp = np.array(self.canvas.camera.GetFocalPoint()) + delta
        cp = self.canvas.camera.GetPosition() + delta
        self.canvas.camera.SetFocalPoint(fp[0], fp[1], fp[2])
        self.canvas.camera.SetPosition(cp[0], cp[1], cp[2])
        self.canvas.render_window.Render()
        
    def on_size(self, event):
        self.redContour.Layout()
        self.greenContour.Layout()
        self.blueContour.Layout()
        self.Layout()
        
    def on_pick(self, event):
        assert isinstance(event, PickEvent)
        if event.mouse_event.GetEventType() not in wx.EVT_LEFT_UP.evtType:
            return
        actor = event.actor
        if isinstance(actor, Contour) and self.redContour.is_my_actor(actor):
            assert isinstance(actor, Contour)
            with wx.ProgressDialog(
                "Calculating statistics",
                "Thresholding Golgi", 
                maximum = 70,
                parent = self) as progress:
                assert isinstance(progress, wx.ProgressDialog)
                progress.SetSize((480, 160))
                redThreshold = actor.get_threshold()
                progress.Update(10, "Segmenting and picking Golgi")
                progress.Refresh
                redSegmentation, count = scipy.ndimage.label(
                    redThreshold, np.ones((3, 3, 3), bool))
                idx = np.max(redSegmentation[int(event.z), int(event.y), :])
                if idx == 0:
                    return
                stats = []
                progress.Update(20, "Calculating red total intensity")
                x, y, z = np.where(redSegmentation == idx)
                total_green = np.sum(self.imgGreen[x, y, z])
                stats.append(("Total red intensity", total_green))
                progress.Update(30, "Calculating red mean intensity")
                mean_green = float(total_green) / len(x)
                stats.append(("Mean red intensity", mean_green))
                if self.blueContour.enabled:
                    progress.Update(40, "Thresholding nucleii")
                    blueThreshold = self.blueContour.contour.get_threshold()
                    progress.Update(50, "Calculating distance from nucleii")
                    d = scipy.ndimage.distance_transform_edt(
                        blueThreshold==0)
                    progress.Update(60, "Calculating anisotropy")
                    mean_d = np.mean(d[x, y, z])
                    sd_d = np.std(d[x, y, z])
                    anisotropy = np.mean(
                        self.imgGreen[x, y, z] * (d[x, y, z] - mean_d) / sd_d) / \
                        mean_green
                    stats.append(("Anisotropy", anisotropy))
            wx.MessageBox(
                "\n".join(["%s: %f" %stat for stat in stats]),
                caption = "Statistics for golgi # %d" %idx,
                parent = self,
                style = wx.ID_OK | wx.ICON_INFORMATION)
    
def main(args):
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        filenames = sum(map(glob.glob, sys.argv[1:]), [])
        planes = [bioformats.load_image(filename) for filename in filenames]
        img_red, img_green, img_blue = [
            np.dstack([plane[:, :, i] for plane in planes]) *255
            for i in range(3)]
        app = wx.PySimpleApp()
        frame = Q3DFrame(img_red, img_green, img_blue, None)
        app.MainLoop()
        
    finally:
        javabridge.kill_vm()
        
if __name__=="__main__":
    main(sys.argv)