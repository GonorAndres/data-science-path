"""Gradio demo for the RAG QA system."""

import gradio as gr

from src.rag import answer


def qa(question: str):
    if not question.strip():
        return "Ingresa una pregunta.", ""
    result = answer(question)
    sources = "\n".join(
        f"[{i + 1}] {c['metadata']['source']}"
        for i, c in enumerate(result["chunks"])
    )
    return result["answer"], sources


with gr.Blocks(title="Local RAG QA (Llama 3.1)") as demo:
    gr.Markdown("# Local RAG QA (Ollama + Llama 3.1)")
    gr.Markdown(
        "Preguntas y respuestas ancladas en una pequena seleccion de Wikipedia en espanol. "
        "Ejecuta localmente con Ollama; sin API externa."
    )
    with gr.Row():
        q_in = gr.Textbox(label="Pregunta", placeholder="Que es la division de poderes?")
    btn = gr.Button("Responder")
    a_out = gr.Textbox(label="Respuesta", lines=6)
    src_out = gr.Textbox(label="Fuentes", lines=4)
    btn.click(qa, inputs=q_in, outputs=[a_out, src_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
