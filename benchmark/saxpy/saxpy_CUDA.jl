import CUDA

function _saxpy_cuda!(z, a, x, y)
    maxPossibleThreads = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    size = length(z)
    threads = min(size, maxPossibleThreads)
    blocks = ceil(Int32, size / threads)
    CUDA.@sync CUDA.@cuda threads = threads blocks = blocks _saxpy_cuda_kernel!(z, a, x, y)
end


function _saxpy_cuda_kernel!(z, a, x, y)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if i <= length(z)
        @inbounds z[i] = a * x[i] + y[i]
    end
    return nothing
end

function run_native()

    # warm up step
    n = 1024
    x = CUDA.randn(Float32, n)
    y = CUDA.randn(Float32, n)
    z = CUDA.zeros(Float32, n)
    a = Float32(2.5)
    _saxpy_cuda!(z, a, x, y)

    for n in [131_072, 262_144, 524_288, 1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216]
        x = CUDA.randn(Float32, n)
        y = CUDA.randn(Float32, n)
        z = CUDA.zeros(Float32, n)
        a = Float32(2.5)
        time = @elapsed _saxpy_cuda!(z, a, x, y)
        println("n: ", n, " time: ", time)
    end
end

run_native()