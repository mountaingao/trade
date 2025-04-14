import math
import gradio as gr
import plotly.express as px
import numpy as np

small_and_beautiful_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#EBFAF2",
        c100="#CFF3E1",
        c200="#A8EAC8",
        c300="#77DEA9",
        c400="#3FD086",
        c500="#02C160",
        c600="#06AE56",
        c700="#05974E",
        c800="#057F45",
        c900="#04673D",
        c950="#2E5541",
        name="small_and_beautiful",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f6f7f8",
        # c100="#f3f4f6",
        c100="#F2F2F2",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        # c900="#272727",
        c900="#2B2B2B",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    # button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_600",
    # button_primary_background_fill_hover="*primary_400",
    # button_primary_border_color="*primary_500",
    button_primary_border_color_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_50",
    button_secondary_background_fill_dark="*neutral_900",
    button_secondary_text_color="*neutral_800",
    button_secondary_text_color_dark="white",
    # background_fill_primary="#F7F7F7",
    # background_fill_primary_dark="#1F1F1F",
    # block_title_text_color="*primary_500",
    block_title_background_fill_dark="*primary_900",
    block_label_background_fill_dark="*primary_900",
    input_background_fill="#F6F6F6",
    # chatbot_code_background_color="*neutral_950",
    # gradio 会把这个几个chatbot打头的变量应用到其他md渲染的地方，鬼晓得怎么想的。。。

)


test_data_1 = {'1':['11','12'],'2':['21','22']}

test_data_2 = {
    "11":'这里是{11}的输出',
    "12":'这里是{12}的输出',
    "21":'这里是{21}的输出',
    "22":'这里是{22}的输出',
}


with gr.Blocks(theme=small_and_beautiful_theme) as demo: # small_and_beautiful_theme 让页面边框变得简介
    # 控件框架
    with gr.Tab(label="对话"):
        with gr.Accordion(label="Prompt", open=True): # open可以选择下面整个模块是否显示
            with gr.Accordion(label="加载模板", open=True):# open可以选择下面整个模块是否显示
                with gr.Column():   # 模块按行排布
                    gr.Markdown("一级下拉：", elem_classes="hr-line")
                    with gr.Row():  # 模块按列排布
                        with gr.Column(scale=6):
                            templateFileSelectDropdown = gr.Dropdown( # 一级下拉菜单
                                label="选择模板集合文件",
                                choices= test_data_1.keys(),
                                multiselect=False,
                                value=list(test_data_1.keys())[0],
                                container=False,
                            )

                    with gr.Row():
                        # gr.Markdown("二级下拉：", elem_classes="hr-line")
                        with gr.Column():
                            gr.Markdown("二级下拉：", elem_classes="hr-line")
                            templateSelectDropdown = gr.Dropdown( # 二级下拉菜单
                                label="从模板中加载",
                                choices=None,
                                multiselect=False,
                                container=False,
                            )
                    templateRefreshBtn = gr.Button("🔄 刷新") # 刷新按钮
            # 内容显示栏目
            systemPromptTxt = gr.Textbox(
                show_label=True,
                placeholder="在这里输入System Prompt...",
                label="System prompt",
                value='请重新选择Prompt模版',
                lines=8
            )


    # 按钮功能1:刷新按钮的点击行为
    # get_template_dropdown 【刷新按钮】传导给【下拉菜单】 templateFileSelectDropdown
    def get_template_dropdown():
        # 输入：无输入项
        # 输出：更新【一级下拉】选项，【二级下拉】置空
        # 触发方式: click点击行为
        return gr.Dropdown(choices=test_data_1.keys()), None

    templateRefreshBtn.click(get_template_dropdown, None,
                             [templateFileSelectDropdown,templateSelectDropdown])

    # 按钮功能2:选择一级下拉 -> 二级下拉联动
    def load_template(key):
        # 输入：templateFileSelectDropdown 【一级下拉】
        # 输出：更新【二级下拉】选项  templateSelectDropdown
        # 触发方式: input当用户更改组件的值时触发
        return gr.Dropdown(choices=test_data_1[key])

    templateFileSelectDropdown.change(
        load_template,
        templateFileSelectDropdown,
        [templateSelectDropdown],
        show_progress=True,
    )

    # 按钮功能3:二级菜单的选择
    def get_template_content(selection):
        # 输入：templateSelectDropdown 【二级下拉】
        # 输出：更新【system prompt】选项  systemPromptTxt
        # 触发方式: change当组件的值发生变化时触发
        try:
            return test_data_2[selection]
        except:
            return '请重新选择模版'

    templateSelectDropdown.change(
        get_template_content,
        [templateSelectDropdown],
        [systemPromptTxt],
        show_progress=True,
    )



if __name__ == "__main__":
    demo.queue().launch()