# dmrg3
3-site dmrg based on ITensors.jl

Usage:
```julia
using ITensors
include("./dmrg3.jl")
```
at the beginning of the script and call
```julia
dmrg3(H, psi0, sweeps)
```
to do 3-site dmrg
