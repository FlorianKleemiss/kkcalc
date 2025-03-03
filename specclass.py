import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

class Spec:
    def __init__(self, path_to_spec):
        self.data_path = path_to_spec
        self.spec_count = 1
        self.column_names = []
        self.conj_data = pd.DataFrame()
        self.sep_dfs = []
         
    def __str__(self):
        return self.data_path
        
    def evaluate(self):
        self.read_inp_file()
        lengths = []
        for spec in self.sep_dfs:
            lengths.append(spec.shape[0])
        print(f"{self.spec_count} specs have been found in your input file. Your specs are of length: {lengths}")
        
        
    def read_inp_file(self):
        with open(self.data_path, "r") as inp:
            in_data_block = False
            buffer = []
            for i, line in enumerate(inp):
                if not in_data_block:
                    if line.startswith("#L energy"):
                        line = line.split("\n")[0]
                        self.column_names = line.split("  ")[0:]
                        self.column_names[0] = "energy"
                    elif line.startswith("#") or line == "\n":
                        continue
                    elif line[0].isdigit():
                        in_data_block = True
                        buffer.append(line.split("\n")[0].split(" "))
                else:
                    if line.startswith("#C") and "Scan aborted" in line:
                        continue
                    elif line == "\n":
                        continue
                    elif line[0].isdigit():
                        buffer.append(line.split("\n")[0].split(" "))
                    elif line.startswith("#S"):
                        self.spec_count += 1
                        in_data_block = False
                        self.process_buffer(buffer)
                        buffer = []
                    else:
                        continue
            self.process_buffer(buffer)
    
    def process_buffer(self, buffer):
        df_buffer = pd.DataFrame(data=buffer,columns=self.column_names,dtype=np.float32)
        self.conj_data = self.conj_data.append(df_buffer)
        self.sep_dfs.append(df_buffer)
        
    def output_all(self):
        """ returns a full pandas DataFrame of the entire .spec file with all columns"""
        return self.conj_data
        
    def output_split(self):
        """ returns a list if panda DataFrames of each spec of the .spec file with all columns"""
        return self.sep_dfs
    
    def output_E_a_array(self, a = "sca", b = None, sep = None):
        """ returns a x,y numpy array where x is "energy" and y is a parameter (default "sca") 
            if sep is present returns only the partial spec run given by index sep
            if b is present column a will be divided by b to "normalize"
        """
        if sep == None:
            if b == None:
                df = self.conj_data[["energy", a]]
                pass
            else:
                df = self.conj_data[["energy", a, b]]
                df["div"] = df[a] / df[b]
                df = df[["energy", "div"]]
            a = df.to_numpy()
            return a
        else:
            if b == None:
                df = self.sep_dfs[sep][["energy", a]]
                pass
            else:
                df = self.sep_dfs[sep][["energy", a, b]]
                df["div"] = df[a] / df[b]
                df = df[["energy", "div"]]
            a = df.to_numpy()
            return a
