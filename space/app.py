"""demodyne health check — Hugging Face Space.

Type a LeRobot dataset id; get a plain-language verdict plus the technical report.
Only tabular data is downloaded (no videos); analysis is capped at 30 episodes.
"""
import traceback

import gradio as gr

import demodyne as dd
from demodyne.audit import audit
from demodyne.cli import audit_table, ident_table
from demodyne.plain import summary

EXAMPLES = ["lerobot/svla_so100_pickplace", "lirislab/so100_demo", "maximilienroberti/so100_lego_red_box"]


def check(repo_id, progress=gr.Progress()):
    repo_id = (repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        return "Enter a dataset id like `lerobot/svla_so100_pickplace`.", ""
    try:
        progress(0.1, desc="Downloading tabular data (no videos)")
        ds = dd.load(repo_id, max_episodes=30)
        progress(0.4, desc="Measuring each joint")
        fits = dd.fit_dataset(ds)
        progress(0.7, desc="Running physics checks")
        findings = audit(ds, fits)
        tech = ident_table(ds, fits) + "\n\n" + audit_table(findings)
        return summary(ds, fits, findings), tech
    except Exception as e:  # show a readable error instead of a stack trace
        return f"Could not analyse `{repo_id}`: {type(e).__name__}: {e}", f"```\n{traceback.format_exc()[-2000:]}\n```"


with gr.Blocks(title="demodyne — robot dataset health check") as demo:
    gr.Markdown(
        "# 🦾 Robot dataset health check\n"
        "Paste the name of a **LeRobot** teleoperation dataset from the Hugging Face Hub. demodyne compares what "
        "the robot was *told* to do with what it *did*, measures how the arm responds, and flags recording "
        "problems that are invisible in the files themselves. Works best on leader–follower arms such as "
        "SO-100/SO-101, Koch or ALOHA. [Code & method](https://github.com/shoggi211/demodyne)")
    with gr.Row():
        box = gr.Textbox(label="Dataset id", placeholder="user/dataset_name", scale=4)
        btn = gr.Button("Check", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=box)
    verdict = gr.Markdown()
    with gr.Accordion("Technical report", open=False):
        tech = gr.Markdown()
    btn.click(check, box, [verdict, tech])
    box.submit(check, box, [verdict, tech])

if __name__ == "__main__":
    demo.launch()
