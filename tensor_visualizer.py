#!/bin/env python

""" -----------------
    Tensor Visualizer
    -----------------
    An Interactive GUI Application
    for Tensor Visualization based on Tkinter

    Author: K. Kawabata
    Debug Environment: Python 3.x

"""
import glob
import os
import numpy as np
import pandas as pd

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import seaborn as sns
sns.set(style='whitegrid', palette='pastel')

import ui_config as ui

# Constant values
EW = tk.E + tk.W
NS = tk.N + tk.S
NEWS = tk.N + tk.E + tk.W + tk.S


class TensorVisualizer(tk.Tk):
    def __init__(self, figsize=None, default_path=None):
        super().__init__()

        # Fundamental settings
        self.title('Tensor Visualizer')
        self.configure(
            bg='gray' #'#0B0B3B',
        )

        self.rank = 3
        self.data = None
        self.labels = None
        self.figsize = (7, 4) if figsize is None else figsize

        # Construct basic components
        path = self._init_input_form(default_path=default_path)
        xaxh = self._init_xaxis_handler()
        yaxh = self._init_yaxis_handler()
        ptls = self._init_plot_tools()
        path.grid(column=0, row=0, ipady=5, sticky=NEWS)
        xaxh.grid(column=0, row=1, ipady=5, sticky=NEWS)
        yaxh.grid(column=0, row=2, ipady=5, sticky=NEWS)
        ptls.grid(column=0, row=3, ipady=5, sticky=NEWS)

        # Construct canvas
        fig = Figure(figsize=figsize)
        fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.get_tk_widget().grid(column=0, row=5, sticky=EW)
        toolbarframe = tk.Frame(self)
        toolbarframe.grid(column=0, row=4, sticky=EW)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbarframe)
        toolbar.update()

        self.mainloop()  # Activation

    def _init_input_form(self, master=None, default_path=None):
        frame = ttk.Frame(master)

        label_input = ttk.Label(frame,
            text='DBPATH',
            font=(ui.fontname, ui.fontsize, ui.fonttype))

        self.entry_path = ttk.Entry(frame, width=50)
        if default_path is not None:
            self.entry_path.insert(tk.END, default_path)

        self.button_path = ttk.Button(frame,
            text='LOAD',
            command=self._pushed_load)

        label_input.grid(column=0, row=0)
        self.entry_path.grid(column=1, row=0)
        self.button_path.grid(column=2, row=0)

        return frame

    def _init_xaxis_handler(self, master=None, rank=3):

        frame = ttk.Frame(master)
        xaxis_label = ttk.Label(frame, text='XAxis',
            font=(ui.fontname, ui.fontsize, ui.fonttype))
        xaxis_label.grid(column=0, row=0, rowspan=rank, padx=5)

        # create radio buttons to select
        self.xaxis_ = tk.IntVar(value=0)  # default: 1st mode

        self.scale_ = []
        self.stvar_ = []
        self.edvar_ = []

        for i in range(rank):
            imode_rdbtn = ttk.Radiobutton(frame, text=f'Mode {i}',
                value=i, variable=self.xaxis_)
            imode_rdbtn.grid(column=2*i+2, row=1)

            # crate scale bars
            stvar = tk.IntVar(value=0)
            edvar = tk.IntVar(value=0)
            self.stvar_.append(stvar)
            self.edvar_.append(edvar)

            xaxis_st_label = ttk.Label(frame, text='St',
                font=(ui.fontname, ui.fontsize, ui.fonttype))
            xaxis_ed_label = ttk.Label(frame, text='Ed',
                font=(ui.fontname, ui.fontsize, ui.fonttype))

            xaxis_st_scale = tk.Scale(frame,
                                      variable=stvar,
                                      resolution=1,
                                      orient='horizontal',
                                      length=200,
                                      from_=0, to=0)
            xaxis_ed_scale = tk.Scale(frame,
                                      variable=edvar,
                                      resolution=1,
                                      orient='horizontal', 
                                      length=200,
                                      from_=0, to=0)

            xaxis_st_label.grid(column=2*i+1, row=2, padx=10, pady=1)
            xaxis_ed_label.grid(column=2*i+1, row=3, padx=10, pady=1)
            xaxis_st_scale.grid(column=2*i+2, row=2, padx=10, pady=1)
            xaxis_ed_scale.grid(column=2*i+2, row=3, padx=10, pady=1)

            self.scale_.append([xaxis_st_scale, xaxis_ed_scale])

        return frame

    def _xaxis_st_changed(self):
        self.xst = stvar.get()
    def _xaxis_ed_changed(self):
        self.xed = edvar.get()

    def _init_yaxis_handler(self, master=None):
        frame = ttk.Frame(master)

        yaxis_label = ttk.Label(frame, text='YAxis',
            font=(ui.fontname, ui.fontsize, ui.fonttype))
        yaxis_label.grid(column=0, row=0, rowspan=self.rank, padx=5)

        self.cbbox_ = []  # i-th dim's m-th mode's combobox

        for i in range(ui.max_dim):
            labeli = ttk.Label(frame,
                text=f'DIM {i+1}',
                font=(ui.fontname, ui.fontsize, ui.fonttype))
            labeli.grid(column=3*i+1, row=0, columnspan=3)

            tmp = []

            for m in range(self.rank):
                cbboxi_m = ttk.Combobox(
                    frame, width=10, state='readonly')
                mnsbtt_m = ttk.Button(frame, text='-', width=1,
                    command=self._pushed_mns_btt(i, m))
                plsbtt_m = ttk.Button(frame, text='+', width=1,
                    command=self._pushed_pls_btt(i, m))
                mnsbtt_m.grid(column=3*i+1, row=m+1)
                cbboxi_m.grid(column=3*i+2, row=m+1)
                plsbtt_m.grid(column=3*i+3, row=m+1)

                tmp.append(cbboxi_m)

            self.cbbox_.append(tmp)

        return frame

    def _pushed_pls_btt(self, dim, mode):
        def _aux():
            try:
                cursor = self.cbbox_[dim][mode].current()
                if cursor < len(self.labels[mode]) - 1:
                    cursor += 1
                self.cbbox_[dim][mode].set(
                    self.labels[mode][cursor])
            except:
                pass
            self._plot_tensor()

        return _aux
    
    def _pushed_mns_btt(self, dim, mode):
        def _aux():
            try:
                cursor = self.cbbox_[dim][mode].current()
                if cursor > -1:
                    cursor -= 1
                if cursor == -1:
                    self.cbbox_[dim][mode].set('')
                else:
                    self.cbbox_[dim][mode].set(
                        self.labels[mode][cursor])
            except:
                pass
            self._plot_tensor()

        return _aux

    def _pushed_load(self):

        filepath = self.entry_path.get()

        if not os.path.exists(filepath):
            messagebox.showerror('ERROR', 'Directory not found')
            return  # do nothing
        if not filepath[-1] == '/':
            filepath += '/'
        filelist = glob.glob(filepath + '*.csv')
        if not filelist:
            messagebox.showerror('ERROR', 'File not found')
            return 

        print(filelist)
        data = [pd.read_csv(filename) for filename in filelist]

        # check data shape
        shape = data[0].shape
        for df in data:
            if not shape == df.shape:
                messagebox.showerror('ERROR', 'Data shapes are not same')
                return

        self.data = data
        self.tensor = np.array([df.iloc[:, 1:].values for df in data])
        self.tensor = np.moveaxis(self.tensor, 0, -1)
        self.labels = []
        self.labels.append(list(self.data[0].iloc[:, 0]))
        self.labels.append(list(self.data[0].keys()[1:]))
        self.labels.append(
            [fn.split('/')[-1].split('.')[0] for fn in filelist])

        print('Loaded')
        self._refresh_tensor_info()

    def _init_plot_tools(self, master=None):
        frame = ttk.Frame(master)

        plot_bttn = ttk.Button(frame, text='PLOT', width=5,
            command=self._plot_tensor)
        reset_bttn = ttk.Button(frame, text='RESET', width=5,
            command=self._refresh)

        plot_bttn.grid(column=0, row=0)
        reset_bttn.grid(column=1, row=0)

        return frame

    def _read_tensor_info(self):

        xaxis = self.xaxis_.get()
        dims = []

        for i in range(ui.max_dim):
            tmp = []
            for m in range(self.rank):
                if m == xaxis:
                    continue

                a = self.cbbox_[i][m].current()
                if a == -1:
                    continue
                tmp.append(a)

    def _plot_tensor(self):
        # refresh canvas
        ax = self.canvas.figure.get_axes()[0]
        ax.clear()

        # x-axis
        xaxis = self.xaxis_.get()
        tensor = np.moveaxis(self.tensor, xaxis, 0)
        st = int(self.stvar_[xaxis].get())
        ed = int(self.edvar_[xaxis].get())
        if st >= ed:
            messagebox.showerror('ERROR', 'Invalid data duration')
            return

        print(tensor)
        print(tensor.shape)

        for i in range(ui.max_dim):
            dims = []
            lbls = []
            seqi = np.copy(tensor)

            for m in range(self.rank):
                if m == xaxis: continue
                dims.append(self.cbbox_[i][m].current())
                lbls.append(self.cbbox_[i][m].get())

            if any(np.array(dims) == -1):
                continue

            for d in dims:
                seqi = seqi[:, d]

            label = '-'.join(lbls)
            ax.plot(np.arange(ed - st), seqi[st:ed],
                lw=ui.linewidth, label=label)

        ax.legend()
        self.canvas.figure.tight_layout()

        self.canvas.draw()


    def _refresh(self):

        # reset duration
        for m in range(self.rank):
            self.stvar_[m].set(0)
            self.edvar_[m].set(self.tensor.shape[m])

        # reset dimensionality
        for i in range(ui.max_dim):
            for m in range(self.rank):
                self.cbbox_[i][m].set('')

        ax = self.canvas.figure.get_axes()[0]
        ax.clear()
        self.canvas.draw()


    def _refresh_tensor_info(self):

        for i, (st, ed) in enumerate(self.scale_):
            st.configure(to=self.tensor.shape[i])
            ed.configure(to=self.tensor.shape[i])

        for i in range(ui.max_dim):
            for mode, cbx in enumerate(self.cbbox_[i]):
                cbx['values'] = self.labels[mode]


if __name__ == '__main__':
    default_path = 'dat/toy/'
    tv = TensorVisualizer(default_path=default_path)