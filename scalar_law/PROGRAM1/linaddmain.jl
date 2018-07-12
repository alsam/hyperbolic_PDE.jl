#***********************************************************************
#  Copyright 2006 John A. Trangenstein
#
#  This software is made available for research and instructional use
#  only.
#  You may copy and use this software without charge for these
#  non-commercial purposes, provided that the copyright notice and
#  associated text is reproduced on all copies.
#  For all other uses (including distribution of modified versions),
#  please contact the author at
#    John A. Trangenstein
#    Department of Mathematics
#    Duke University
#    Durham, NC 27708-0320
#    USA
#  or
#    johnt@math.duke.edu
#
#  This software is made available "as is" without any assurance that it
#  is completely correct, or that it will work for your purposes.
#  Use the software at your own risk.
#***********************************************************************
__precompile__()
using Printf, DocOpt
using OffsetArrays

include("consts.jl")

using .consts

include("riemprob.jl")
include("linearad.jl")
include("consdiff.jl")
include("upwind.jl")

function do_computation(ncells, nsteps, verbose, print_solution)

    u    = OffsetArray{Float64}(undef, -2:ncells+1)
    x    = OffsetArray{Float64}(undef,  0:ncells)
    flux = OffsetArray{Float64}(undef,  0:ncells)
    dfdu = OffsetArray{Float64}(undef, -2:ncells+1)

    tmax     =  0.8
    cfl      =  0.9

    # array bounds:
    fc=-2
    lc=ncells+1
    fm=0
    lm=ncells-1
    fs=0
    ls=ncells-1
    ifirst=0
    ilast=ncells-1

    initsl(ncells,fc,lc,fm,lm,ifirst,ilast, u,x)

    if verbose
        println("x : $x u : $u")
    end

    bcmesh(fm,lm,ncells, x)
    bccells(fc,lc,ncells, u)
    fluxderv(fc,lc,fc,lc, u, dfdu)

    istep=0
    t=0.0
    while istep < nsteps && t < tmax
        bccells(fc,lc,ncells, u)
        fluxderv(fc,lc,fc,lc, u, dfdu)
        dt=cfl*stabledt(fc,lc,fm,lm,ifirst,ilast, u, dfdu,x)
        method(dt,fc,lc,fm,lm,fs,ls,ifirst,ilast, u, flux)
        consdiff(fc,lc,fs,ls,fm,lm,ifirst,ilast, x,flux, u)

        t=t+dt
        istep=istep+1
    end

    # write final results (plot later)
    if print_solution
        @inbounds for ic=0:ncells-1
            xc = (x[ic]+x[ic+1])*0.5
            uc = u[ic]
            @printf("%e %e\n",xc,uc)
        end
    end

end

function main()
    script_name = basename(Base.source_path())
    doc = """$script_name

Usage:
  $script_name -h | --help
  $script_name [-v | --verbose] [-p | --print] [--cells=<cells>] [--steps=<steps>] [--runs=<runs>]

Options:
  -h --help                  Show this screen.
  -v --verbose               Adds verbosity.
  -p --print                 Print solution.
  --cells=<cells>            Specify a number of cells [default: 100].
  --steps=<steps>            Specify a number of steps [default: 10000].
  --runs=<runs>              Specify a number of runs  [default: 2].
"""

    arguments = docopt(doc)

    verbose   = arguments["--verbose"]
    print_sol = arguments["--print"]
    ncells    = parse(Int, arguments["--cells"])
    nsteps    = parse(Int, arguments["--steps"])
    nruns     = parse(Int, arguments["--runs"])

    if verbose
        println("ncells: $ncells")
        println("nsteps: $nsteps")
    end

    for r = 1:nruns
        @time do_computation(ncells, nsteps, verbose, print_sol)
    end

end # main

main()



