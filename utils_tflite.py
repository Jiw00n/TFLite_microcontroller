import numpy as np
import tensorflow as tf
from PIL import Image


def get_interpreter(model_path):
    try:
        it = tf.lite.Interpreter(model_path=model_path)
    except ValueError:
        from tensorflow.lite.python import schema_py_generated as schema
        import flatbuffers
        m = schema.ModelT.InitFromObj(
            schema.Model.GetRootAsModel(bytearray(open(model_path, 'rb').read()), 0))
        for sg in m.subgraphs:
            for t in sg.tensors:
                if t.quantization and t.shape is not None and len(t.shape) <= 1:
                    t.quantization.quantizedDimension = 0
        b = flatbuffers.Builder(1024)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        open('person_detect_fixed.tflite', 'wb').write(b.Output())
        it = tf.lite.Interpreter(model_path='person_detect_fixed.tflite')
    it.allocate_tensors()
    return it

def run_inference(model_path, bmp_path):
    
    interpreter = get_interpreter(model_path)
    in_idx  = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    img = np.array(Image.open(bmp_path).convert('L'))      # grayscale 96x96, 0~255
    x = img.astype(np.int8).reshape(1, 96, 96, 1)          # uint8 -> int8 재해석
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_idx)[0]               # (2,) int8
    return img, int(out[1]), int(out[0])                   # person=idx1, no_person=idx0
