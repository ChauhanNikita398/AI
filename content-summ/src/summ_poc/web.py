import logging
from typing import Any

import panel
import panel as pn
from py_executable_checklist.workflow import run_workflow

from summ_poc.workflow import (
    inference_workflow_steps,
    pdf_name_from,
    pdf_to_chat_archive_path,
    pdf_to_faiss_db_path,
    pdf_to_index_path,
)

css = """
:root{
  background-color: white;
  max-width: 500px;
  margin: auto;
}
.bk.bk-clearfix{
  padding-left: 10px;
  background-color: #b8d0eb;
  border-radius: 5px;
  margin-left: 5px;
  padding-right: 10px;
  margin-right:5px;
}
.bk.bk-btn.bk-btn-default{
  background-color: #a663cc;
  color: white;
}
.p{  
  marging-left: 10px;
}
input[type="file" i]{
    width: 600;
    background-color: #ff9e00;
    color: black; 
}
"""
pn.extension()
pn.extension(loading_spinner="dots", loading_color="#6f2dbd",raw_css=[css])


txt_input = pn.widgets.TextInput(value="", placeholder="Enter text here...", sizing_mode="stretch_width")
btn_ask = pn.widgets.Button(name="Ask me something!", width=100)
panel_conversations = []  # store all panel objects in a list
global_context = {}  # store workflow context


def add_qa_to_panel(question: str, answer: str) -> str:
    qa_block = f"""
🤖    **{question}**

📖    {answer}
    """
    panel_conversations.append(
        pn.Row(
            pn.pane.Markdown(
                qa_block,
                width=600,
                style={
                    "background-color": "#F6F6F6",
                    "line-height": "1.5",
                },
            ),
        )
    )
    return qa_block


def get_conversations(_: Any) -> pn.Column:
    prompt = txt_input.value_input
    logging.info("Getting conversation for prompt: %s for input %s", prompt, txt_input)
    if prompt != "":
        input_question = global_context["input_question"] = prompt
        run_workflow(global_context, inference_workflow_steps())
        openai_answer = global_context["output"]
        logging.info("Answer: %s", openai_answer)
        qa_block = add_qa_to_panel(input_question, openai_answer)
        with open(global_context["archive_file"], "a",encoding='utf-8') as f:
            f.write(qa_block)
    txt_input.value_input = ""
    return pn.Column(*(reversed(panel_conversations)))


def run_inference_workflow(context: dict) -> None:
    run_workflow(context, inference_workflow_steps())


def run_web(context: dict) -> None:
    global global_context
    context["index_path"] = pdf_to_index_path(context["app_dir"], context["input_pdf_path"])
    context["faiss_db"] = pdf_to_faiss_db_path(context["app_dir"], context["input_pdf_path"])
    context["archive_file"] = pdf_to_chat_archive_path(context["app_dir"], context["input_pdf_path"])
    global_context = context
    interactive_conversation = pn.bind(get_conversations, btn_ask)

    pdf_name = pdf_name_from(context["input_pdf_path"])
    panel_conversations.append(
        pn.pane.Markdown(f"📖 Ask me something about {pdf_name}", width=600, style={"background-color": "#a663cc"}),
    )
    
    print("global_context=")
    print(global_context)
    run_workflow(global_context, inference_workflow_steps())  ##IMP
    add_qa_to_panel(f"*Here is the summary of {pdf_name}*", global_context["output"])
    html_pane = pn.pane.HTML("""<h1>This is an HTML pane</h1>""")
    file_input = pn.widgets.FileInput(accept='.pdf', multiple=True)
    
    if file_input.value is not None:
        file_input.save('uploaded.pdf')

    dashboard = pn.Column(
        pn.Row(txt_input,btn_ask),
        pn.Row(file_input),
        pn.panel(
            interactive_conversation,
            loading_indicator=True,
            height=500,
            style={"border-radius": "5px", "border": "1px black solid"},
        )
    )
    panel.serve(dashboard, port=5006, show=True)

