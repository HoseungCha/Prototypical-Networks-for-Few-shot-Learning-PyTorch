import matplotlib
import matplotlib.pyplot as plt


class util_figure:
    """ how to use
    f = plt.figure(1)
    plt.plot(np.linspace(0,1,100))
    util_figure_obj = util_figure()
    rect_pos = util_figure_obj.getFigPostionRect()
    util_figure_obj.move_figure(f,rect_pos[0], rect_pos[1])
    """
    def __init__(self):
        self.mngr = plt.get_current_fig_manager()

    def getFigPostionRect(self):
        dx = int(self.mngr.window.geometry().split('+')[0].split('x')[0])
        dy = int(self.mngr.window.geometry().split('+')[0].split('x')[1])
        xpos = int(self.mngr.window.geometry().split('+')[1])
        ypos = int(self.mngr.window.geometry().split('+')[2])

        x1 = xpos
        y1 = ypos
        x2 = x1 + dx
        y2 = y1 + dy

        return x1, y1, x2, y2

    def move_figure(self, f, x, y):
        """
        cxrodgers @ https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib/7450808#7450808
        """
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            f.canvas.manager.window.move(x, y)



