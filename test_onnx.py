import onnxruntime
import numpy as np
import cv2
import glob
import time
import argparse

nframes = 2000
coco_val = '/home/jquinn/datasets/coco/images/val2017/'
res = 640

def run_test(use_tensorrt, batch_size, onnx_file):

    providers = ['TensorrtExecutionProvider'] if use_tensorrt else ['CUDAExecutionProvider']
    print(providers)
    session = onnxruntime.InferenceSession(onnx_file, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    print(output_names)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(input_name)
    print(input_shape)

    t0 = time.perf_counter()
    count = 0
    warm = False
    img_batch = np.zeros((batch_size, 3, res, res), dtype=np.float32)
    n = 0 
    img_filenames = glob.glob(coco_val + '*.jpg')
    for img_filename in img_filenames[:nframes]:
        img = cv2.imread(img_filename)
        img = cv2.resize(img, (res, res)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_batch[n, :, :, :] = img
        n += 1
        if n == batch_size:
            if warm:
                y = session.run(output_names, {input_name: img_batch})[0]
                count += batch_size
            else:
                y = session.run(output_names, {input_name: img_batch})[0]
                warm = True
                t0 = time.perf_counter()
                count = 0 

            print(img_filename)
            print(y.shape)
            n = 0 

    dt = time.perf_counter() - t0
    time_per_image = 1000 * dt / count
    print(f'{dt}, {count}, Time per image: {time_per_image} ms')
    return time_per_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tensorrt', action='store_true', help='Use TensorRT EP (or else use CUDA)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--onnx', default='yolov5s-leakyrelu3-dynamic.onnx', help='Path to ONNX model')
    args = parser.parse_args()
    time_per_image = run_test(args.use_tensorrt, args.batch, args.onnx)


if __name__ == '__main__':
    main()