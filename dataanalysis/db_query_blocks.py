#file:D:\project\trade\dataanalysis\db_query_ui.py
import gradio as gr
import mysql.connector
import pandas as pd
import json, os
import matplotlib
matplotlib.use('Agg')  # 设置 matplotlib 后端为 Agg，避免 GUI 相关的错误


class DBQueryUI:


    def __init__(self):
        # 新增代码：读取配置文件
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        db_config = config['db_config']

        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()
        self.tables = self.get_tables()

    def get_tables(self):
        self.cursor.execute("SHOW TABLES;")
        return [table[0] for table in self.cursor.fetchall()]

    def get_fields(self, table):
        self.cursor.execute(f"DESCRIBE {table}")
        fields = [field[0] for field in self.cursor.fetchall()]
        field_types = [field[1] for field in self.cursor.fetchall()]
        return fields, field_types

    def is_numeric_or_date(self, field_type):
        numeric_types = ['int', 'float', 'decimal']
        date_types = ['date', 'datetime', 'timestamp']
        return any(t in field_type.lower() for t in numeric_types + date_types)

    def query_with_conditions(self, table, conditions):
        fields, _ = self.get_fields(table)
        query = f"SELECT * FROM {table}"
        where_clauses = []

        for field, condition in conditions.items():
            if condition:
                operator, value = condition.split(' ')
                where_clauses.append(f"{field} {operator} '{value}'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        self.cursor.execute(query)
        results = self.cursor.fetchall()
        df = pd.DataFrame(results, columns=fields)
        return df

    def on_query(self, table, conditions):
        df = self.query_with_conditions(table, conditions)
        return df

    def on_export(self, table, conditions):
        df = self.query_with_conditions(table, conditions)
        file_path = "output.xlsx"
        df.to_excel(file_path, index=False)
        return f"数据已导出到 {file_path}"

    def update_fields_and_conditions(self, selected_table):
        fields, field_types = self.get_fields(selected_table)
        new_input_components = []
        new_conditions = {}

        for field, field_type in zip(fields, field_types):
            is_numeric_or_date = self.is_numeric_or_date(field_type)
            label = f"{field} ({field_type})"
            if is_numeric_or_date:
                condition_dropdown = gr.Dropdown(
                    label=label,
                    choices=["=", "<", ">"],
                    value=None,
                    interactive=True
                )
            else:
                condition_dropdown = gr.Dropdown(
                    label=label,
                    choices=["="],
                    value=None,
                    interactive=True
                )

            value_input = gr.Textbox(label=f"{field} 值", value="", interactive=True)
            new_conditions[field] = None
            new_input_components.extend([condition_dropdown, value_input])

        return new_input_components

    def setup_ui(self):
        def query_interface(table, *conditions):
            conditions_dict = {}
            fields, _ = self.get_fields(table)
            for i, field in enumerate(fields[:3]):  # 只选择前3个字段作为示例
                conditions_dict[field] = conditions[i]
            df = self.query_with_conditions(table, conditions_dict)
            print(df)
            return df

        def export_interface(table, *conditions):
            conditions_dict = {}
            fields, _ = self.get_fields(table)
            for i, field in enumerate(fields[:3]):  # 只选择前3个字段作为示例
                conditions_dict[field] = conditions[i]
            return self.on_export(table, conditions_dict)

        fields, field_types = self.get_fields(self.tables[0])  # 默认第一个表的字段
        input_components = []
        for field, field_type in zip(fields[:3], field_types[:3]):  # 只选择前3个字段作为示例
            is_numeric_or_date = self.is_numeric_or_date(field_type)
            if is_numeric_or_date:
                input_components.append(gr.Dropdown(label=f"{field} ({field_type})", choices=["=", "<", ">"], value=None, interactive=True))
            else:
                input_components.append(gr.Dropdown(label=f"{field} ({field_type})", choices=["="], value=None, interactive=True))
            input_components.append(gr.Textbox(label=f"{field} 值", value="", interactive=True))

        query_interface = gr.Interface(
            fn=query_interface,
            inputs=[gr.Dropdown(label="选择表格", choices=self.tables, value=self.tables[0])] + input_components,
            outputs=gr.Dataframe(label="查询结果"),
            title="数据库查询工具",
            description="选择表格并输入查询条件，获取查询结果。",
            concurrency_limit=1
        )

        export_interface = gr.Interface(
            fn=export_interface,
            inputs=[gr.Dropdown(label="选择表格", choices=self.tables, value=self.tables[0])] + input_components,
            outputs=gr.Textbox(label="导出状态"),
            title="数据库导出工具",
            description="选择表格并输入查询条件，导出数据到Excel。",
            concurrency_limit=1
        )

        query_interface.launch(server_name='127.0.0.1', server_port=7929, debug=True)
        export_interface.launch(server_name='127.0.0.1', server_port=7930, debug=True)

if __name__ == "__main__":
    app = DBQueryUI()
    app.setup_ui()
