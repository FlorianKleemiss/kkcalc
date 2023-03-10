import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()

things_to_look_for = (
"CRYSTALLOGRAPHY ALPHAANGLEINDEG",
"CRYSTALLOGRAPHY BETAANGLEINDEG",
"GONIOMETER TYPE KUMA_KM4 DEF",
"ROTATION BEAM",
"ROTATION DETECTORORIENTATION",
"ROTATION ZEROCORRECTION")

def get_folders_with_par_file(path):
    folders = []
    for item in os.listdir(path):
        _ = os.path.join(path,item)
        if os.isdir(_):
            folders.append(get_folders_with_par_file(_))
        if os.path.splitext(_)[1] == ".par":
            folders.append(path)
    folder_set = set(folders)
    return list(folder_set)

file_path = filedialog.askopenfilename(defaultextension=".par",
                                       filetypes=([("Crysalis Parameter File", ".par")]),
                                       title="Please select template .par file")
binary_file = filedialog.askopenfilename(defaultextension=".proffitpars",
                                       filetypes=([("Crysalis Proffit-Parameter File", ".proffitpars")]),
                                       title="Please select Proffitpars file")
work_folder = filedialog.askdirectory(title="Please select folder to look for measurements to prepare")

all_folders = get_folders_with_par_file(work_folder)

with open(file_path) as template_par:
    lines = template_par.readlines()
    null = 0
