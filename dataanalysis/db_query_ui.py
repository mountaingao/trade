import tkinter as tk
from tkinter import ttk
import pandas as pd
import mysql.connector
from tkinter import messagebox
from tkinter import filedialog
import json,os

class DBQueryUI:
    def __init__(self, root, db_config):
        self.root = root
        self.db_config = db_config
        self.conn = mysql.connector.connect(**self.db_config)
        self.cursor = self.conn.cursor()
        self.tables = self.get_tables()
        self.selected_table = None
        self.fields = []
        
        self.setup_ui()
    
    def get_tables(self):
        self.cursor.execute("SHOW TABLES;")
        return [table[0] for table in self.cursor.fetchall()]
    
    def setup_ui(self):
        self.root.title("数据库查询工具")
        
        # 表格选择
        self.table_label = ttk.Label(self.root, text="选择表格:")
        self.table_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.table_combobox = ttk.Combobox(self.root, values=self.tables)
        self.table_combobox.grid(row=0, column=1, padx=10, pady=10)
        self.table_combobox.bind("<<ComboboxSelected>>", self.on_table_select)
        
        # 字段选择
        self.field_label = ttk.Label(self.root, text="选择字段:")
        self.field_label.grid(row=1, column=0, padx=10, pady=10)
        
        self.field_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.field_listbox.grid(row=1, column=1, padx=10, pady=10)
        
        # 查询按钮
        self.query_button = ttk.Button(self.root, text="查询", command=self.on_query)
        self.query_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # 导出按钮
        self.export_button = ttk.Button(self.root, text="导出Excel", command=self.on_export)
        self.export_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        
        # 结果显示
        self.result_text = tk.Text(self.root, height=20, width=80)
        self.result_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
    
    def on_table_select(self, event):
        self.selected_table = self.table_combobox.get()
        self.cursor.execute(f"DESCRIBE {self.selected_table}")
        self.fields = [field[0] for field in self.cursor.fetchall()]
        self.field_listbox.delete(0, tk.END)
        for field in self.fields:
            self.field_listbox.insert(tk.END, field)
    
    def on_query(self):
        selected_fields = [self.field_listbox.get(i) for i in self.field_listbox.curselection()]
        if not selected_fields:
            messagebox.showwarning("警告", "请选择至少一个字段")
            return
        
        query = f"SELECT {', '.join(selected_fields)} FROM {self.selected_table}"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        self.result_text.delete(1.0, tk.END)
        for row in results:
            self.result_text.insert(tk.END, str(row) + "\n")
    
    def on_export(self):
        selected_fields = [self.field_listbox.get(i) for i in self.field_listbox.curselection()]
        if not selected_fields:
            messagebox.showwarning("警告", "请选择至少一个字段")
            return
        
        query = f"SELECT {', '.join(selected_fields)} FROM {self.selected_table}"
        df = pd.read_sql_query(query, self.conn)
        
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            df.to_excel(file_path, index=False)
            messagebox.showinfo("成功", f"数据已导出到 {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    # 新增代码：读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
        config = json.load(config_file)
    db_config = config['db_config']
    app = DBQueryUI(root, db_config)
    root.mainloop()