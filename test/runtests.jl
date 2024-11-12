using Test

include("test-comms.jl")

@testset verbose=true "channel estimation" begin
  @testset "LMS" begin
    channel_estimation(Float32, [(0,0.8f0), (7,0.3f0), (11,-0.5f0)], LMS())
    channel_estimation(Float64, [(0,0.8), (7,0.3), (11,-0.5)], LMS())
    channel_estimation(ComplexF32, [(0,0.8f0+0im), (7,0+0.3f0im), (11,-0.2f0-0.5f0im)], LMS())
    channel_estimation(ComplexF64, [(0,0.8+0im), (7,0+0.3im), (11,-0.2-0.5im)], LMS())
  end
  @testset "NLMS" begin
    channel_estimation(Float32, [(0,0.8f0), (7,0.3f0), (11,-0.5f0)], NLMS(1f0))
    channel_estimation(Float64, [(0,0.8), (7,0.3), (11,-0.5)], NLMS(1f0))
    channel_estimation(ComplexF32, [(0,0.8f0+0im), (7,0+0.3f0im), (11,-0.2f0-0.5f0im)], NLMS(1f0))
    channel_estimation(ComplexF64, [(0,0.8+0im), (7,0+0.3im), (11,-0.2-0.5im)], NLMS(1f0))
  end
  @testset "options" begin
    channel_estimation(Float64, [(0,0.8), (7,0.3), (11,-0.5)], LMS(); ts=1:1000)
    channel_estimation(Float64, [(0,0.8), (7,0.3), (11,-0.5)], LMS(); saveat=1)
    channel_estimation(Float64, [(0,0.8), (7,0.3), (11,-0.5)], LMS(); ts=1:1000, saveat=5)
  end
end

@testset verbose=true "channel equalization" begin
  @testset "LMS" begin
    channel_equalization([(0,0.8+0im), (7,0+0.3im), (11,-0.2-0.5im)], LMS(0.01))
  end
  @testset "NLMS" begin
    channel_equalization([(0,0.8+0im), (7,0+0.3im), (11,-0.2-0.5im)], NLMS())
  end
end