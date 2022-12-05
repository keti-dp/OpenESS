import sys

try:
    import PyQt5
    SUPPORT_QT = True
    del PyQt5
    from GUIProcess import *
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
except ModuleNotFoundError:
    SUPPORT_QT = False


if __name__ == '__main__':
    # Not Support QT Interface
    if not SUPPORT_QT:
        print(f'Fatal Error : GUI Not Supported')
        sys.exit(-1)
    # Support QT Interface
    if SUPPORT_QT:
        gui_process = GUIClass()
        gui_process.show()
        app.exec_()
    sys.exit(0)
