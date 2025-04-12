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

    def update_fields_and_conditions(self, selected_table):
        fields, field_types = self.get_fields(selected_table)
        new_input_components = []
        new_conditions = {}

        # 只选择前3个字段作为示例
        for field, field_type in zip(fields[:3], field_types[:3]):
            is_numeric_or_date = self.is_numeric_or_date(field_type)
            label = f"{field} ({field_type})"
            condition_dropdown = gr.Dropdown(
                label=label,
                choices=["=", "<", ">"],
                value="=",
                interactive=True
            )

            if "date" in field_type.lower():
                value_input = gr.Textbox(label=f"{field} 值", value="", interactive=True, placeholder="YYYY-MM-DD")
            else:
                value_input = gr.Textbox(label=f"{field} 值", value="", interactive=True)

            new_conditions[field] = None
            new_input_components.extend([condition_dropdown, value_input])

        return new_input_components

    def on_query(self, table, conditions):
        # 构建查询语句
        query = f"SELECT * FROM {table}"
        if conditions:
            where_clause = " AND ".join([f"{field} {cond} %s" for field, cond in conditions.items()])
            query += f" WHERE {where_clause}"
        
        # 执行查询
        self.cursor.execute(query, tuple(conditions.values()))
        result = self.cursor.fetchall()
        
        # 将结果转换为 DataFrame
        df = pd.DataFrame(result, columns=[desc[0] for desc in self.cursor.description])
        return df

    def setup_ui(self):
        with gr.Blocks() as app:
            gr.Markdown("# 数据库查询工具")

            # 输入容器
            with gr.Row() as input_row:
                # 表格选择
                table_dropdown = gr.Dropdown(label="选择表格", choices=self.tables, value=self.tables[0])

                input_components = self.update_fields_and_conditions(self.tables[0])

                # 字段选择和条件输入
                province = gr.Dropdown(label="字段")
                operator = gr.Dropdown(label="运算")
                value = gr.Textbox(label="值")

                # province = gr.Dropdown(label="Province")
                # operator = gr.Dropdown(label="operator")
                # value = gr.Dropdown(label="value")
                #
                # province = gr.Dropdown(label="Province")
                # operator = gr.Dropdown(label="operator")
                # value = gr.Dropdown(label="value")
                # input_container = gr.Dropdown(label="选择字段",value=input_components)  # 新增：将输入组件放入一个容器中

    #     country = Dropdown(label="Country", choices=countries)
    #     province = Dropdown(label="Province")
    #     city = Dropdown(label="City")
    #     output = Textbox()
    #
    # country.change(fn=update_provinces, inputs=country, outputs=province)  # 更新省份选项基于国家选择。
    # province.change(fn=update_cities, inputs=[province, country], outputs=city)  # 更新城市选项基于国家和省份选择。注意这里需要确保省份与国家匹配。
    # city.change(fn=predict, inputs=[country, province, city], outputs=output)  # 输出最终选择。
    # Button("Submit").click(fn=predict, inputs=[country, province, city], outputs=output)  # 提交按钮触发预测函数。

    # 查询按钮
            query_button = gr.Button("查询")
            query_output = gr.Dataframe(label="查询结果")

            # 导出按钮
            export_button = gr.Button("导出Excel")
            export_output = gr.Textbox(label="导出状态")

            # 更新字段和条件
            table_dropdown.change(
                self.update_fields_and_conditions,
                inputs=table_dropdown,
                outputs=input_components
            )

            # 查询事件
            query_button.click(
                fn=self.on_query,
                inputs=[table_dropdown, gr.State({})],
                outputs=[query_output]
            )

            # 导出事件
            export_button.click(
                fn=self.on_export,
                inputs=[table_dropdown, gr.State({})],
                outputs=[export_output]
            )

        app.launch()

    def on_export(self, table, conditions):
        # 构建查询语句
        query = f"SELECT * FROM {table}"
        if conditions:
            where_clause = " AND ".join([f"{field} {cond} %s" for field, cond in conditions.items()])
            query += f" WHERE {where_clause}"
        
        # 执行查询
        self.cursor.execute(query, tuple(conditions.values()))
        result = self.cursor.fetchall()
        
        # 将结果转换为 DataFrame
        df = pd.DataFrame(result, columns=[desc[0] for desc in self.cursor.description])
        
        # 导出为 Excel 文件
        export_dir = os.path.join(os.path.dirname(__file__), '..', 'exports')
        os.makedirs(export_dir, exist_ok=True)  # 确保导出目录存在
        export_path = os.path.join(export_dir, f"{table}_export.xlsx")
        df.to_excel(export_path, index=False)
        
        return f"数据已成功导出到 {export_path}"

if __name__ == "__main__":
    app = DBQueryUI()
    app.setup_ui()
