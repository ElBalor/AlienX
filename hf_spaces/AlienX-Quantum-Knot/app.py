import os
os.environ.setdefault('ALIENX_KNOT_EPOCHS', '0')
os.environ.setdefault('ALIENX_KNOT_NO_LAUNCH', '1')
import importlib.util
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qknot_engine', ROOT / 'AlienX-Qknot.py')
q = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(q)
q.model.load_state_dict(__import__('torch').load(ROOT / 'alienx_quantum_knot_checkpoint.pt', map_location=q.device, weights_only=True)['model'])
q.model.eval()

with gr.Blocks(title='AlienX Quantum Knot') as demo:
    gr.Markdown('# ALIENX QUANTUM KNOT\n### SO(3) frame · relevance-gated complex QSA · temporal trefoil response')
    gr.Markdown('Interactive inference uses the released checkpoint. The slider changes a deterministic perturbation trajectory; it does not retrain the model.')
    strength = gr.Slider(0, 1, value=.5, step=.05, label='Perturbation strength')
    run = gr.Button('Run temporal knot readout', variant='primary')
    plot = gr.Plot()
    run.click(q.predict_reconnection, strength, plot)
    demo.load(q.predict_reconnection, strength, plot)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)), show_error=True)
