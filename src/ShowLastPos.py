import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import pickle

class ShowLastPosClass:

    def __init__(self, plt):
        self.plt = plt

    def ShowLastPosSingleFig(self):
        # Call plt.show but pickles the plot position on window close.  When called a second time
        # it loads the figure to the last position.  So matplotlib now remembers figure positions!
        # This version works for QT and WX backends.

        plt = self.plt

        backend = matplotlib.get_backend()

        # WX backend
        if 'WX' in backend:
            mgr = plt.get_current_fig_manager()
            try:
                with open('CurrentWindowPos.pkl', 'r') as f:
                    CurPos = pickle.load(f)
                mgr.window.SetPosition((CurPos[0], CurPos[1]))
                mgr.window.SetSize((CurPos[2], CurPos[3]))
            except:
                pass
            plt.show()
            p = mgr.window.GetPosition()
            s = mgr.window.GetSize()
            CurPos = (p[0], p[1], s[0], s[1])
            with open('CurrentWindowPos.pkl', 'w') as f:
                pickle.dump(CurPos, f)
        # QT backend.
        elif 'QT' in backend:
            mgr = plt.get_current_fig_manager()
            try:
                with open('CurrentWindowPos.pkl', 'r') as f:
                    CurPos = pickle.load(f)
                mgr.window.setGeometry(CurPos[0], CurPos[1], CurPos[2], CurPos[3])
            except:
                pass
            plt.show()
            CurPos = mgr.window.geometry().getRect()
            with open('CurrentWindowPos.pkl', 'w') as f:
                pickle.dump(CurPos, f)
        else:
            print
            'Backend ' + backend + ' not supported.  Plot figure position will not be sticky.'
            plt.show()

    ###########################################
    # Version for dealing with multiple figures.
    ###########################################

    def ShowLastPos(self):
        # Call plt.show but pickles the plot position on window close.  When called a second time
        # it loads the figure to the last position.  So matplotlib now remembers figure positions!
        # This version works for QT and WX backends.

        plt = self.plt

        backend = matplotlib.get_backend()

        FigNums = plt.get_fignums()

        for FigNum in FigNums:
            plt.figure(FigNum)
            fig = plt.gcf()
            fig.canvas.mpl_connect('close_event', RecordLastPos)
            mgr = plt.get_current_fig_manager()
            # WX backend
            if 'WX' in backend:
                try:
                    with open('CurrentWindowPos%d.pkl' % FigNum, 'r') as f:
                        CurPos = pickle.load(f)
                    mgr.window.SetPosition((CurPos[0], CurPos[1]))
                    mgr.window.SetSize((CurPos[2], CurPos[3]))
                except:
                    pass
            # QT backend.
            elif 'QT' in backend:
                try:
                    with open('CurrentWindowPos%d.pkl' % FigNum, 'r') as f:
                        CurPos = pickle.load(f)
                    mgr.window.setGeometry(CurPos[0], CurPos[1], CurPos[2], CurPos[3])
                except:
                    pass
            else:
                print
                'Backend ' + backend + ' not supported.  Plot figure position will not be sticky.'

        plt.show()

    def RecordLastPos(self, evt):

        backend = matplotlib.get_backend()

        FigNums = plt.get_fignums()

        for FigNum in FigNums:
            plt.figure(FigNum)
            mgr = plt.get_current_fig_manager()
            # WX backend
            if 'WX' in backend:
                p = mgr.window.GetPosition()
                s = mgr.window.GetSize()
                CurPos = (p[0], p[1], s[0], s[1])
                with open('CurrentWindowPos%d.pkl' % FigNum, 'w') as f:
                    pickle.dump(CurPos, f)
            # QT backend.
            elif 'QT' in backend:
                CurPos = mgr.window.geometry().getRect()
                with open('CurrentWindowPos%d.pkl' % FigNum, 'w') as f:
                    pickle.dump(CurPos, f)
            else:
                pass