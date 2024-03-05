

if JACC.backend() == "CUDA"
    
else if JACC.backend() == "AMDGPU"
    import AMDGPU
end

function saxpy_JACC!(z::Vector{Float32}, x::Vector{Float32}, a::Float32, y::Vector{Float32})

    function saxpy!(i, z, x, a, y)
        if i <= length(x)
            @inbounds z[i] += alpha * y[i]
        end
    end

    JACC.saxpy!(z, x, a, y)
    return nothing
end