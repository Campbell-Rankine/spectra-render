from src.gradio import ui

if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)