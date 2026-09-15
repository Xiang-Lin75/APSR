# Efficiency measurements

`benchmark.py` measures whole-utterance P inference with R4, FP32 and batch one. Each timed pass includes enrollment processing, CPU-to-device input transfer and device-to-CPU output transfer. CUDA is synchronized around the measurement. File reads, model construction and weight loading are excluded; warmup passes are discarded. Raw pass times, RTF values, input lengths, host thread count, device and software version are saved.

The default input is deterministic synthetic noise (4-s mixture and 3-s enrollment). This checks executable throughput; it does not reproduce the paper's duration figure or a speech-quality experiment. Whole-utterance RTF is not a per-frame deadline, algorithmic delay or full streaming latency measurement.

The MAC profiler counts Conv2d, Linear, GRU, attention QK/AV and enrollment pooling products by actual module calls. The four shared refinement calls are each counted. P materializes dense attention score matrices before masking, so the masked entries still contribute to this estimate. STFT/iSTFT, normalization, nonlinearities, softmax and elementwise arithmetic are excluded. Report the mixture/enrollment lengths alongside MAC/s; the enrollment cost and dense attention make this length-dependent.

Source-integration testing compares the per-operation counts against the frozen source profiler. The public forward removes unused historical diagnostics and auxiliary sink-waveform reconstruction; target output and neural MACs are preserved. Runtime measured using this package should be identified as this implementation, rather than silently substituted into historical runtime results.
