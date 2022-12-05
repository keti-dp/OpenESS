# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Main_uizywGEq.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
from enum import Enum

from DataProcessing import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.layout_get_data = QHBoxLayout()
        self.layout_get_data.setObjectName(u"layout_get_data")
        self.btn_connect_query = QPushButton(self.centralwidget)
        self.btn_connect_query.setObjectName(u"btn_connect_query")

        self.layout_get_data.addWidget(self.btn_connect_query)

        self.btn_load_logfile = QPushButton(self.centralwidget)
        self.btn_load_logfile.setObjectName(u"btn_load_logfile")

        self.layout_get_data.addWidget(self.btn_load_logfile)

        self.horizontalSpacer = QSpacerItem(0, 0, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)

        self.layout_get_data.addItem(self.horizontalSpacer)

        self.cb_run_normalization = QCheckBox(self.centralwidget)
        self.cb_run_normalization.setObjectName(u"cb_run_normalization")
        self.cb_run_normalization.setChecked(True)

        self.layout_get_data.addWidget(self.cb_run_normalization)

        self.cb_run_standardization = QCheckBox(self.centralwidget)
        self.cb_run_standardization.setObjectName(u"cb_run_standardization")
        self.cb_run_standardization.setChecked(True)

        self.layout_get_data.addWidget(self.cb_run_standardization)

        self.btn_visualize = QPushButton(self.centralwidget)
        self.btn_visualize.setObjectName(u"btn_visualize")

        self.layout_get_data.addWidget(self.btn_visualize)

        self.verticalLayout.addLayout(self.layout_get_data)

        self.layout_visualize = QHBoxLayout()
        self.layout_visualize.setObjectName(u"layout_visualize")

        self.verticalLayout.addLayout(self.layout_visualize)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.layout_selectable_data = QVBoxLayout()
        self.layout_selectable_data.setObjectName(u"layout_selectable_data")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.layout_selectable_data.addWidget(self.label_2)

        self.list_unselected_columns = QListWidget(self.groupBox)
        self.list_unselected_columns.setObjectName(u"list_unselected_columns")

        self.layout_selectable_data.addWidget(self.list_unselected_columns)

        self.horizontalLayout.addLayout(self.layout_selectable_data)

        self.layout_selectbuttonbox = QVBoxLayout()
        self.layout_selectbuttonbox.setObjectName(u"layout_selectbuttonbox")
        self.verticalSpacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.layout_selectbuttonbox.addItem(self.verticalSpacer)

        self.btn_select_data = QPushButton(self.groupBox)
        self.btn_select_data.setObjectName(u"btn_select_data")
        self.btn_select_data.setMinimumSize(QSize(0, 50))
        self.btn_select_data.setMaximumSize(QSize(50, 16777215))

        self.layout_selectbuttonbox.addWidget(self.btn_select_data)

        self.btn_unselect_data = QPushButton(self.groupBox)
        self.btn_unselect_data.setObjectName(u"btn_unselect_data")
        self.btn_unselect_data.setMinimumSize(QSize(0, 50))
        self.btn_unselect_data.setMaximumSize(QSize(50, 16777215))

        self.layout_selectbuttonbox.addWidget(self.btn_unselect_data)

        self.verticalSpacer_2 = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.layout_selectbuttonbox.addItem(self.verticalSpacer_2)

        self.horizontalLayout.addLayout(self.layout_selectbuttonbox)

        self.layout_selected_data = QVBoxLayout()
        self.layout_selected_data.setObjectName(u"layout_selected_data")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.layout_selected_data.addWidget(self.label_3)

        self.list_selected_columns = QListWidget(self.groupBox)
        self.list_selected_columns.setObjectName(u"list_selected_columns")

        self.layout_selected_data.addWidget(self.list_selected_columns)

        self.horizontalLayout.addLayout(self.layout_selected_data)

        self.verticalLayout.addWidget(self.groupBox)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QWidget.setTabOrder(self.btn_connect_query, self.btn_load_logfile)
        QWidget.setTabOrder(self.btn_load_logfile, self.cb_run_normalization)
        QWidget.setTabOrder(self.cb_run_normalization, self.cb_run_standardization)
        QWidget.setTabOrder(self.cb_run_standardization, self.btn_visualize)

        self.process_addon_UI()

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", u"ESS \ub370\uc774\ud130 \uc870\ud68c", None))
        self.btn_connect_query.setText(QCoreApplication.translate("MainWindow", u"\ucffc\ub9ac \uc811\uc18d", None))
        self.btn_load_logfile.setText(
            QCoreApplication.translate("MainWindow", u"\ud30c\uc77c \uae30\ub85d \uc870\ud68c", None))
        self.cb_run_normalization.setText(QCoreApplication.translate("MainWindow", u"\uc815\uaddc\ud654", None))
        self.cb_run_standardization.setText(QCoreApplication.translate("MainWindow", u"\ud45c\uc900\ud654", None))
        self.btn_visualize.setText(QCoreApplication.translate("MainWindow", u"\ubd84\uc11d/\uc2dc\uac01\ud654", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\ub370\uc774\ud130 \uc120\ud0dd", None))
        self.label_2.setText(
            QCoreApplication.translate("MainWindow", u"\ubbf8 \uc120\ud0dd\ub41c \ub370\uc774\ud130", None))
        self.btn_select_data.setText(QCoreApplication.translate("MainWindow", u"\u25b6", None))
        self.btn_unselect_data.setText(QCoreApplication.translate("MainWindow", u"\u25c0", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\uc120\ud0dd\ub41c \ub370\uc774\ud130", None))

    # retranslateUi

    def process_addon_UI(self):
        """
        Plotting Widget 적용

        :return: None
        """
        self.widget = WidgetPlot(self.centralwidget)
        self.layout_visualize.addWidget(self.widget)


class UI_QuaryLogin(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(400, 125)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QSize(400, 125))
        Dialog.setMaximumSize(QSize(400, 125))
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(60, 0))
        self.label.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.label)

        self.txt_query_address = QLineEdit(Dialog)
        self.txt_query_address.setObjectName(u"txt_query_address")

        self.horizontalLayout.addWidget(self.txt_query_address)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(Dialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(60, 0))
        self.label_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_2)

        self.txt_login_account = QLineEdit(Dialog)
        self.txt_login_account.setObjectName(u"txt_login_account")

        self.horizontalLayout_3.addWidget(self.txt_login_account)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_3 = QLabel(Dialog)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(60, 0))
        self.label_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.label_3)

        self.txt_login_password = QLineEdit(Dialog)
        self.txt_login_password.setObjectName(u"txt_login_password")
        self.txt_login_password.setEchoMode(QLineEdit.Password)

        self.horizontalLayout_2.addWidget(self.txt_login_password)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        self.buttonBox.setCenterButtons(True)

        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)

    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"\ucffc\ub9ac \uc811\uc18d", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"\uc811\uc18d \uc8fc\uc18c", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"\uacc4\uc815", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"\uc554\ud638", None))
    # retranslateUi


class WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self, width=10, height=8)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

    def plot(self, data, title='', line_width=0.5):
        self.canvas.plot(data, title=title, line_width=line_width)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data, title='', line_width=0.5):
        ax = self.figure.add_subplot(111)
        ax.plot(data, linewidth=line_width)
        ax.set_title(title)
        self.draw()


class QueryLoginClass(UI_QuaryLogin, QDialog):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent_class = parent
        self.set_interaction()

    def set_interaction(self):
        """
        버튼 상호작용 처리

        :return: None
        """
        self.buttonBox.accepted.connect(self.apply_quary_information)
        self.buttonBox.rejected.connect(self.reject_quary_information)

    def apply_quary_information(self):
        """
        쿼리 적용

        :return: None
        """
        self.parent_class._query_address = self.txt_query_address.text()
        self.parent_class._query_account = self.txt_login_account.text()
        self.parent_class._query_password = self.txt_login_password.text()
        self.parent_class._import_data_source = DataType.SQLQuary
        self.close()

    def reject_quary_information(self):
        self.close()


class DataType(Enum):
    NoData = 0b00
    SQLQuary = 0b01
    LogCSVData = 0b10


class GUIClass(Ui_MainWindow, QMainWindow):
    _run_normalization: bool = True
    _run_standardization: bool = True

    # Data Information
    _import_data_source: DataType = DataType.NoData

    # Query Information
    _query_address: str = ''
    _query_account: str = ''
    _query_password: str = ''

    # Log Data Information
    _log_path: str = ''

    # Plotting Data
    _data = None
    _column_list = []

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.set_interaction()

    def set_interaction(self):
        """
        버튼 상호작용 처리

        :return: None
        """

        self.cb_run_normalization.stateChanged.connect(self.cb_run_normalization_changed)
        self.cb_run_standardization.stateChanged.connect(self.cb_run_standardization_changed)

        self.btn_connect_query.clicked.connect(self.btn_connect_query_clicked)
        self.btn_load_logfile.clicked.connect(self.btn_load_logfile_clicked)
        self.btn_visualize.clicked.connect(self.btn_visualize_clicked)
        self.btn_select_data.clicked.connect(self.btn_select_data_clicked)
        self.btn_unselect_data.clicked.connect(self.btn_unselect_data_clicked)
        return

    def cb_run_normalization_changed(self):
        self._run_normalization = self.cb_run_normalization.isChecked()

    def cb_run_standardization_changed(self):
        self._run_standardization = self.cb_run_standardization.isChecked()

    def btn_connect_query_clicked(self):
        login_dialog = QueryLoginClass(parent=self)
        login_dialog.exec()
        del login_dialog
        return

    def btn_load_logfile_clicked(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open Log CSV Data')
        if not file_path[0]:
            return
        self._log_path = file_path[0]
        self._import_data_source = DataType.LogCSVData
        return

    def btn_visualize_clicked(self):
        """
        시각화 처리

        :return: None
        """
        if self._import_data_source == DataType.NoData:
            import random
            data = [random.random() for i in range(250)]
            self.widget.plot(data, title='Example')
            return
        elif self._import_data_source == DataType.LogCSVData:
            result, data, error_message = DataProcessing.get_data_to_csv(self._log_path)
            if not result:
                return
            data.set_index(keys=data[data.columns[0]], inplace=True, drop=True)
            data = data[data.columns[2:]]
            data, self._column_list = DataProcessing.preprocessing(data=data,
                                                                   run_normalization=self._run_normalization,
                                                                   run_standardization=self._run_standardization)
            self.list_unselected_columns.clear()
            self.list_selected_columns.clear()
            for row in self._column_list:
                self.list_selected_columns.addItem(row)
            self.widget.plot(data[self._column_list], title=self._log_path.split('/')[-1])
            self._data = data
            return
        elif self._import_data_source == DataType.SQLQuary:
            result, data, error_message = DataProcessing.get_data_to_SQL(
                query_address=self._query_address,
                query_id=self._query_account,
                query_passwd=self._query_password)
            if not result:
                return
            data.set_index(keys=data[data.columns[0]], inplace=True, drop=True)
            data = data[data.columns[1:]]
            data, self._column_list = DataProcessing.preprocessing(data=data,
                                                                   run_normalization=self._run_normalization,
                                                                   run_standardization=self._run_standardization)
            self.list_unselected_columns.clear()
            self.list_selected_columns.clear()
            for row in self._column_list:
                self.list_selected_columns.addItem(row)
            self.widget.plot(data, title=self._query_address)
            self._data = data
            return
        else:
            return

    def btn_select_data_clicked(self):
        """
        출력할 데이터 선택

        :return: None
        """
        try:
            self.list_selected_columns.addItem(self.list_unselected_columns.currentItem().text())
        except AttributeError:
            return
        self.list_unselected_columns.takeItem(self.list_unselected_columns.currentRow())
        self._column_list = [self.list_selected_columns.item(index).text() for index in
                             range(self.list_selected_columns.count())]
        self.widget.plot(self._data[self._column_list])
        return

    def btn_unselect_data_clicked(self):
        """
        출력하지 않을 데이터 선택

        :return: None
        """
        try:
            self.list_unselected_columns.addItem(self.list_selected_columns.currentItem().text())
        except AttributeError:
            return
        self.list_selected_columns.takeItem(self.list_selected_columns.currentRow())
        self._column_list = [self.list_selected_columns.item(index).text() for index in
                             range(self.list_selected_columns.count())]
        self.widget.plot(self._data[self._column_list])
        return
