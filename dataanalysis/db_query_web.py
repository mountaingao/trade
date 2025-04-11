#file:D:\project\trade\dataanalysis\db_query_ui.py
import gradio as gr
import mysql.connector
import pandas as pd
import json, os





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

    def setup_ui(self):


        with gr.Blocks() as app:
            gr.Markdown("# 数据库查询工具")

            # 表格选择
            table_dropdown = gr.Dropdown(label="选择表格", choices=self.tables)

            # 字段选择和条件输入
            fields, field_types = self.get_fields(self.tables[0])  # 默认第一个表的字段
            conditions = {}
            input_components = []

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
                conditions[field] = None
                input_components.extend([condition_dropdown, value_input])

            # 查询按钮
            query_button = gr.Button("查询")
            query_output = gr.Dataframe(label="查询结果")

            # 导出按钮
            export_button = gr.Button("导出Excel")
            export_output = gr.Textbox(label="导出状态")

            # 更新字段和条件
            def update_fields_and_conditions(selected_table):
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

            table_dropdown.change(
                self.update_fields_and_conditions,
                inputs=table_dropdown,
                outputs=[field_dropdown, condition_dropdown, value_input]  # 确保 outputs 是 Gradio 组件列表
            )



            # 查询事件
            query_button.click(
                fn=self.on_query,
                inputs=[table_dropdown, gr.State(conditions)],
                outputs=[query_output]
            )

            # 导出事件
            export_button.click(
                fn=self.on_export,
                inputs=[table_dropdown, gr.State(conditions)],
                outputs=[export_output]
            )

        app.launch()

if __name__ == "__main__":
    app = DBQueryUI()
    app.setup_ui()
