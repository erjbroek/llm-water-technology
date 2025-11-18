# Frontend Gradio UI for RAG Chatbot.

import logging
import gradio as gr
from backend import (
    evaluate_model,
    get_current_evaluation_progress,
    get_db_stats,
    upload_and_process_files,
    reindex_all_documents,
    refresh_documents,
    delete_document,
    chat_response,
    clear_all_data,
    clear_chat_only,
    list_documents,
    vector_store,
)
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, 'INFO'))
logger = logging.getLogger(__name__)


def dropdown_update(selected=None):
    # Update dropdown list with current documents.
    documents = list_documents()
    
    # Try to keep the same selection if it still exists
    value = selected if selected in documents else None
    
    return gr.Dropdown.update(choices=documents, value=value)

def toggle_send_enabled(lang):
     return gr.update(interactive=(lang in ("dutch", "english")))


def create_gradio_interface():
    # Create and return the Gradio interface.
    
    with gr.Blocks(title="Document Q&A Assistant") as demo:
        gr.Markdown("# Document-Based AI Q&A")

        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                folder_input = gr.Textbox(
                    label="Destination subfolder",
                    placeholder="e.g. GO-Fresh/Reports",
                    value="",
                )
                file_upload = gr.Files(
                    file_types=[".pdf", ".txt", ".docx"], 
                    file_count="multiple"
                )
                upload_btn = gr.Button("Process Files", variant="primary")
                upload_status = gr.Textbox(label="File Status")
                
                # Database management section
                db_stats = gr.Textbox(label="DB Stats", value=get_db_stats())
                refresh_btn = gr.Button("Refresh Index")
                
                # Document management section
                document_dropdown = gr.Dropdown(
                    label="Indexed Documents",
                    choices=list_documents(),
                    value=None,
                )
                delete_checkbox = gr.Checkbox(label="Remove source file", value=False)
                delete_btn = gr.Button("Delete Selected Document", variant="stop")
                
                # Bulk operations
                reindex_btn = gr.Button("Reindex All Documents")
                clear_btn = gr.Button("Clear Vector Store & Chat")
                gr.HTML("<br>")
                with gr.Row():
                    retrieval_eval_button = gr.Button("Evaluate chunk retrieval")
                    generation_eval_button = gr.Button("Evaluate model output")

                eval_count = gr.Number(label="Num of evaluations", value=10, precision=1)
                total_eval_button = gr.Button("Full evaluation")

            with gr.Column(scale=2):
                # Chat interface
                chatbot_ui = gr.Chatbot(label="Chatbot", height=500)
                msg_box = gr.Textbox(label="Your Question")
                lang_selector = gr.Radio(
                    choices=["dutch", "english"],
                    label="Language",
                    value=None  # default None so nothing is selected
                )
                send_btn = gr.Button("Send")
                clear_chat_btn = gr.Button("Clear Chat")

        # Wire up event handlers for UI interactions
        upload_btn.click(
            upload_and_process_files,
            [file_upload, folder_input],
            [upload_status, db_stats],
        ).then(
            lambda: dropdown_update(),
            outputs=[document_dropdown]
        )
        
        send_btn.click(
            chat_response, 
            [msg_box, chatbot_ui, lang_selector], 
            [chatbot_ui, msg_box]
        )
        
        msg_box.submit(
            chat_response, 
            [msg_box, chatbot_ui], 
            [chatbot_ui, msg_box]
        )
        
        refresh_btn.click(
            refresh_documents, 
            outputs=[db_stats]
        ).then(
            lambda: dropdown_update(),
            outputs=[document_dropdown]
        )
        
        delete_btn.click(
            delete_document,
            [document_dropdown, delete_checkbox],
            [upload_status, db_stats],
        ).then(
            lambda: dropdown_update(),
            outputs=[document_dropdown]
        )
        
        reindex_btn.click(
            reindex_all_documents,
            outputs=[upload_status, db_stats],
        ).then(
            lambda: dropdown_update(),
            outputs=[document_dropdown]
        )
        
        clear_btn.click(
            clear_all_data,
            outputs=[upload_status, chatbot_ui, db_stats],
        ).then(
            lambda: dropdown_update(),
            outputs=[document_dropdown]
        )
        
        clear_chat_btn.click(
            clear_chat_only, 
            outputs=[chatbot_ui, msg_box]
        )

        retrieval_eval_button.click(
            fn=lambda num: evaluate_model("retrieval", num),
            inputs=eval_count
        )

        generation_eval_button.click(
            fn=lambda num: evaluate_model("generation", num),
            inputs=eval_count
        )

        total_eval_button.click(
            fn=lambda num: evaluate_model("full", num),
            inputs=eval_count
        )

        lang_selector.change(
            fn=toggle_send_enabled,
            inputs=[lang_selector],
            outputs=[send_btn]
        )

    return demo


def launch_app():
    # Launch the Gradio application.
    try:
        logger.info("Creating Gradio interface...")
        demo = create_gradio_interface()
        
        logger.info("Launching Gradio...")
        demo.launch(
            share=config.GRADIO_SHARE, 
            debug=config.GRADIO_DEBUG
        )
        
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        raise


if __name__ == "__main__":
    launch_app()