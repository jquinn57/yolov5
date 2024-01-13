import onnxruntime
import numpy as np
import cv2
import glob
from memryx import AsyncAccl
import time
from IPython import embed

coco_val = '/home/jquinn/datasets/coco/images/val2017/'
dfp_filename = 'yolov5s-leakyrelu3.dfp'
nframes = 2000
count = 0
time_start = []
time_stop = []

# define the callback that will return model input data
def data_source():
    global time_start
    res = 640
    img_filenames = glob.glob(coco_val + '*.jpg')
    for img_filename in img_filenames[:nframes]:
        print(img_filename)
        img = cv2.imread(img_filename)
        img = cv2.resize(img, (res, res)).astype(np.float32)
        time_start.append(time.perf_counter())
        yield img

# define the callback that will process the outputs of the model
def output_processor(*outputs):
    global count
    global time_stop

    count += 1
    time_stop.append(time.perf_counter())


# Accelerate using the MemryX hardware
accl = AsyncAccl(dfp_filename)
onnx_filename = 'model_0_' + dfp_filename.replace('.dfp', '_post.onnx')
accl.set_postprocessing_model(onnx_filename)
accl.connect_input(data_source) # starts asynchronous execution of input generating callback
accl.connect_output(output_processor) # starts asynchronous execution of output processing callback

t0 = time.perf_counter()
accl.wait() # wait for the accelerator to finish execution
t1 = time.perf_counter()

fps = count / (t1 - t0)
inference_time_ms = 1000 * (t1 - t0) / count
print(f'inference time ms: {inference_time_ms}')
print(f'FPS: {fps}')

delta_t = np.array(time_stop) - np.array(time_start)
mean_latency_ms = 1000*np.mean(delta_t)
print(f'mean latency (ms): {mean_latency_ms}')

