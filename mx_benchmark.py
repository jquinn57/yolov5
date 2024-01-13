from memryx import Benchmark


dfp_filename = 'yolov5s-leakyrelu3.dfp'


with Benchmark(dfp=dfp_filename) as accl:
    # 1000 frames, get FPS
    outputs,_,fps = accl.run(frames=1000)

    # single frame, get latency
    outputs, latency,_ = accl.run(threading=False)

    print(f'latency_ms: {latency}')
    print(f'FPS {fps}')
