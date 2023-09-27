import ITensors.AbstractMPS
import ITensors.@debug_check
import ITensors.@printf
import ITensors.replacebond!
import ITensors.@timeit_debug

function Permute(
    M::AbstractMPS, ::Tuple{typeof(linkind),typeof(siteinds),typeof(linkind)}
)::typeof(M)
    M̃ = typeof(M)(length(M))
    for n in 1:length(M)
        lₙ₋₁ = linkind(M, n - 1)
        lₙ = linkind(M, n)
        s⃗ₙ = sort(Tuple(siteinds(M, n)); by=plev)
        M̃[n] = permute(M[n], filter(!isnothing, (lₙ₋₁, s⃗ₙ..., lₙ)))
    end
    set_ortho_lims!(M̃, ortho_lims(M))
    return M̃
end

function dmrg3(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
    ITensors.check_hascommoninds(siteinds, H, psi0)
    ITensors.check_hascommoninds(siteinds, H, psi0')
    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = Permute(H, (linkind, siteinds, linkind))
    PH = ProjMPO(H)
    ITensors.set_nsite!(PH, 3)
    return dmrg3(PH, psi0, sweeps; kwargs...)
end

function dmrg3(PH, psi0::MPS, sweeps::Sweeps; kwargs...)
    if length(psi0) == 1
        error(
            "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
        )
    end

    @debug_check begin
        # Debug level checks
        # Enable with ITensors.enable_debug_checks()
        checkflux(psi0)
        checkflux(PH)
    end

    which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
    svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
    obs = get(kwargs, :observer, NoObserver())
    outputlevel::Int = get(kwargs, :outputlevel, 1)

    write_when_maxdim_exceeds::Union{Int,Nothing} = get(
        kwargs, :write_when_maxdim_exceeds, nothing
    )
    write_path = get(kwargs, :write_path, tempdir())

    # eigsolve kwargs
    eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)
    eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 3)
    eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
    eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

    ishermitian::Bool = get(kwargs, :ishermitian, true)

    # TODO: add support for targeting other states with DMRG
    # (such as the state with the largest eigenvalue)
    # get(kwargs, :eigsolve_which_eigenvalue, :SR)
    eigsolve_which_eigenvalue::Symbol = :SR

    # TODO: use this as preferred syntax for passing arguments
    # to eigsolve
    #default_eigsolve_args = (tol = 1e-14, krylovdim = 3, maxiter = 1,
    #                         verbosity = 0, ishermitian = true,
    #                         which_eigenvalue = :SR)
    #eigsolve = get(kwargs, :eigsolve, default_eigsolve_args)

    # Keyword argument deprecations
    if haskey(kwargs, :maxiter)
        error("""maxiter keyword has been replaced by eigsolve_krylovdim.
                 Note: compared to the C++ version of ITensor,
                 setting eigsolve_krylovdim 3 is the same as setting
                 a maxiter of 2.""")
    end

    if haskey(kwargs, :errgoal)
        error("errgoal keyword has been replaced by eigsolve_tol.")
    end

    if haskey(kwargs, :quiet)
        error("quiet keyword has been replaced by outputlevel")
    end

    psi = copy(psi0)
    N = length(psi)

    if !isortho(psi) || ITensors.orthocenter(psi) != 1
        psi = orthogonalize!(PH, psi, 1)
    end
    @assert isortho(psi) && ITensors.orthocenter(psi) == 1

    if !isnothing(write_when_maxdim_exceeds)
        if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
           (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
            PH = disk(PH; path=write_path)
        end
    end
    PH = position!(PH, psi, 1)
    energy = 0.0

    for sw in 1:nsweep(sweeps)
        sw_time = @elapsed begin
            maxtruncerr = 0.0

            if !isnothing(write_when_maxdim_exceeds) &&
               maxdim(sweeps, sw) > write_when_maxdim_exceeds
                if outputlevel >= 2
                    println(
                        "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
                    )
                end
                PH = disk(PH; path=write_path)
            end

            for (b, ha) in sweepnext(N; ncenter=3)
                @debug_check begin
                    checkflux(psi)
                    checkflux(PH)
                end

                @timeit_debug timer "dmrg: position!" begin
                    PH = position!(PH, psi, b)
                end

                @debug_check begin
                    checkflux(psi)
                    checkflux(PH)
                end

                #----------------------
                @timeit_debug timer "dmrg: psi[b]*psi[b+1]*psi[b+2]" begin
                    phi = psi[b] * psi[b+1] * psi[b+2]
                end

                @timeit_debug timer "dmrg: eigsolve" begin
                    vals, vecs = ITensors.eigsolve(
                        PH,
                        phi,
                        1,
                        eigsolve_which_eigenvalue;
                        ishermitian=ishermitian,
                        tol=eigsolve_tol,
                        krylovdim=eigsolve_krylovdim,
                        maxiter=eigsolve_maxiter
                    )
                end

                energy = vals[1]
                phi::ITensor = vecs[1]

                ortho = ha == 1 ? "left" : "right"

                drho = nothing
                if noise(sweeps, sw) > 0.0
                    @timeit_debug timer "dmrg: noiseterm" begin
                        # Use noise term when determining new MPS basis
                        drho = noise(sweeps, sw) * noiseterm(PH, phi, ortho)
                    end
                end

                @debug_check begin
                    checkflux(phi)
                end

                @timeit_debug timer "dmrg: replacebond3!" begin
                    spec = replacebond3!(
                        PH,
                        psi,
                        b,
                        phi;
                        maxdim=ITensors.maxdim(sweeps, sw),
                        mindim=ITensors.mindim(sweeps, sw),
                        cutoff=ITensors.cutoff(sweeps, sw),
                        eigen_perturbation=drho,
                        ortho=ortho,
                        normalize=true,
                        which_decomp=which_decomp,
                        svd_alg=svd_alg
                    )
                end
                maxtruncerr = max(maxtruncerr, spec.truncerr)

                # add this line for dmrg3. Shift ortho center to the last site
                if (b == N - 2 && ha == 1)
                    orthogonalize!(psi, N)
                end

                @debug_check begin
                    checkflux(psi)
                    checkflux(PH)
                end

                if outputlevel >= 2
                    @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                    @printf(
                        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                        ITensors.cutoff(sweeps, sw),
                        ITensors.maxdim(sweeps, sw),
                        ITensors.mindim(sweeps, sw)
                    )
                    @printf(
                        "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                    )
                    flush(stdout)
                end

                sweep_is_done = (b == 1 && ha == 2)
                measure!(
                    obs;
                    energy=energy,
                    psi=psi,
                    projected_operator=PH,
                    bond=b,
                    sweep=sw,
                    half_sweep=ha,
                    spec=spec,
                    outputlevel=outputlevel,
                    sweep_is_done=sweep_is_done
                )
            end
        end
        if outputlevel >= 1
            @printf(
                "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psi),
                maxtruncerr,
                sw_time
            )
            flush(stdout)
        end
        isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
        isdone && break
    end
    return (energy, psi)
end

function replacebond3!(PH, M::MPS, b::Int, phi::ITensor; kwargs...)
    return replacebond3!(M, b, phi; kwargs...)
end

"""
    replacebond3!(M::MPS, b::Int, phi::ITensor; kwargs...)

Factorize the ITensor `phi` and replace the ITensors
`b`, `b+1` and `b+2` of MPS `M` with the factors. Choose
the orthogonality with `ortho="left"/"right"`.
"""

function replacebond3!(M::MPS, b::Int, phi::ITensor; kwargs...)
    ortho::String = get(kwargs, :ortho, "left")
    swapsites::Bool = get(kwargs, :swapsites, false)
    which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
    normalize::Bool = get(kwargs, :normalize, false)

    # Deprecated keywords
    if haskey(kwargs, :dir)
        error(
            """dir keyword in replacebond3! has been replaced by ortho.
            Note that the options are now the same as factorize, so use `left` instead of `fromleft` and `right` instead of `fromright`.""",
        )
    end

    indsMb = inds(M[b])
    if swapsites
        sb = siteind(M, b)
        sbp1 = siteind(M, b + 1)
        indsMb = replaceind(indsMb, sb, sbp1)
    end
    # ===========================assign new MPS tensors================
    L, tmp, spec1 = ITensors.factorize(
        phi, indsMb; which_decomp=which_decomp, tags=tags(linkind(M, b)), kwargs..., ortho="left"
    )
    err1 = spec1.truncerr
    mid, R, spec2 = ITensors.factorize(
        tmp, (siteind(M, b + 1), commonind(L, tmp)); which_decomp=which_decomp, tags=tags(linkind(M, b + 1)), kwargs..., ortho="right"
    )
    err2 = spec2.truncerr
    M[b] = L
    M[b+1] = mid
    M[b+2] = R
    # ==========================set MPS info============================
    if ortho == "left"
        ITensors.leftlim(M) == b - 1 && ITensors.setleftlim!(M, ITensors.leftlim(M) + 1)
        ITensors.rightlim(M) == b + 1 && ITensors.setrightlim!(M, ITensors.rightlim(M) + 1)
        normalize && (M[b+1] ./= norm(M[b+1]))
    elseif ortho == "right"
        ITensors.leftlim(M) == b + 1 && ITensors.setleftlim!(M, ITensors.leftlim(M) - 1)
        ITensors.rightlim(M) == b + 3 && ITensors.setrightlim!(M, ITensors.rightlim(M) - 1)
        normalize && (M[b+1] ./= norm(M[b+1]))
    else
        error(
            "In replacebond3!, got ortho = $ortho, only currently supports `left` and `right`."
        )
    end
    if err1 > err2
        return spec1
    else
        return spec2
    end
end