import numpy as np
import tecplot as tp
from tecplot.constant import PlotType, Color
import logging
from os import path

# Run this script with "-c" to connect to Tecplot 360 on port 7600
# To enable connections in Tecplot 360, click on:
#   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"
import sys
if '-c' in sys.argv:
    tp.session.connect()

def read(i,k):
    if (k==0):
        method = "lax"
    if (k==1):
        method = "roe"
    if (k==2):
        method = "analytic"
    if (i < 10):
        input = method + "0" + str(i)
    else:
        input = method + str(i)

    #inputAddress = "C:\\\\Users\\Me\\CudaShockTube\\ShockTube\\" + input + ".dat"
    inputAddress = "..\\ShockTube\\" + input + ".dat"

    infile = path.join(inputAddress)
    tp.data.load_tecplot(infile,read_data_option=2)
    plot = tp.active_frame().plot(PlotType.XYLine)
    plot.activate()
    tp.active_frame().plot().axes.grid_area.show_border=True
    tp.macro.execute_command('$!FrameLayout ShowBorder = No')
    tp.active_frame().plot().axes.x_axis(0).ticks.show_on_border_max=True
    tp.active_frame().plot().axes.y_axis(0).ticks.show_on_border_max=True
    #tp.macro.execute_command('$!FrameLayout IsTransparent = Yes')
    tp.active_frame().plot().axes.y_axis(0).title.font.size=6
    tp.active_frame().plot().axes.y_axis(0).title.offset=8
    tp.active_frame().plot().axes.y_axis(0).tick_labels.font.size=4
    tp.active_frame().plot().axes.x_axis(0).title.font.size=5
    tp.active_frame().plot().axes.x_axis(0).title.offset=5
    tp.active_frame().plot().axes.x_axis(0).tick_labels.font.size=4
    tp.active_frame().add_text("Time =%5.2f" %(i*0.01), position=(40, 90), size=24)
    tp.active_frame().plot().axes.y_axis(0).title.font.bold=False
    tp.active_frame().plot().axes.x_axis(0).title.font.bold=False



def plot(i,j,k):
    variables = ("density",  "velocity", "pressure",  "momentum", "energy",  "totalEnergy", "temperature", "soundVelocity", "machNumber", "enthalpy")
    if (k==0):
        method = "lax"
    if (k==1):
        method = "roe"
    if (k==2):
        method = "analytic"
    if (i < 10):
        input = method + "0" + str(i)
    else:
        input = method + str(i)

    tp.active_frame().plot().axes.y_axis(0).ticks.auto_spacing=False
    if (j==0):
        #tp.active_frame().plot().axes.y_axis(0).ticks.spacing_anchor=0.2
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.2
        tp.active_frame().plot().axes.y_axis(0).max=1.05
        tp.active_frame().plot().axes.y_axis(0).min=0
    if (j==1):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.3
        tp.active_frame().plot().axes.y_axis(0).max=1.2
        tp.active_frame().plot().axes.y_axis(0).min=-0.1
    if (j==2):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.2
        tp.active_frame().plot().axes.y_axis(0).max=1.1
        tp.active_frame().plot().axes.y_axis(0).min=0
    if (j==3):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.1
        tp.active_frame().plot().axes.y_axis(0).max=0.5
        tp.active_frame().plot().axes.y_axis(0).min=-0.1
    if (j==4 or j==5):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.5
        tp.active_frame().plot().axes.y_axis(0).max=2.7
        tp.active_frame().plot().axes.y_axis(0).min=0.0
    if (j==6):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.2
        tp.active_frame().plot().axes.y_axis(0).max=1.6
        tp.active_frame().plot().axes.y_axis(0).min=0.5
    if (j==7):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.1
        tp.active_frame().plot().axes.y_axis(0).max=1.5
        tp.active_frame().plot().axes.y_axis(0).min=0.9
    if (j==8):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.3
        tp.active_frame().plot().axes.y_axis(0).max=1.4
        tp.active_frame().plot().axes.y_axis(0).min=-0.2
    if (j==9):
        tp.active_frame().plot().axes.y_axis(0).ticks.spacing=0.5
        tp.active_frame().plot().axes.y_axis(0).max=4.0
        tp.active_frame().plot().axes.y_axis(0).min=0.8

    plot = tp.active_frame().plot(PlotType.XYLine)
    lmap = plot.linemap("density")
    lmap.show = False
    lmap = plot.linemap(j-1)
    lmap.show = False
    lmap = plot.linemap(j)
    lmap.show = True
    lmap.y_axis_index = 0
    lmap.line.line_thickness = 0.4
    lmap.line.color = Color.Custom40
    output = variables[j] + input + ".png"
    tp.export.save_png(output, 300, supersample=3)
    #output = variables[j] + input + ".eps"
    #tp.export.save_eps(output)

# Main()
for k in range(3):
    for i in range(21):
        read(i,k)
        for j in range(10):
            if(j < 3):
                plot(i,j,k)
