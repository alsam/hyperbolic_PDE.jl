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

#@inline getindex(V::SubArray, I::Int64...) = Base.unsafe_getindex(V, I...)
#@inline getindex(V::SubArray, I::Union(Range{Int64},Array{Int64,1},Int64,Colon)...) = Base.unsafe_getindex(V, I...)
#@inline getindex{T,N,P,IV}(V::SubArray{T,N,P,IV}, I::Union(Real, AbstractVector, Colon)...) = Base.unsafe_getindex(V, to_index(I)...)
#@inline setindex!(V::SubArray, v, I::Int64...) = Base.unsafe_setindex!(V, v, I...)

using OffsetArrays # for @inbounds

macro shifted_array(T, r)
    :(view(Array{$T}(undef, length($r)), -minimum($r) + 2 : length($r)))
end

macro shifted_array2(T, r1, r2)
    :(view(Array{$T}(undef, length($r1), length($r2)), -minimum($r1) + 2 : length($r1), -minimum($r2) + 2 : length($r2)))
end

macro shifted_array3(T, r1, r2, r3)
    :(view(Array{$T}(undef, length($r1), length($r2), length($r3)), -minimum($r1) + 2 : length($r1), -minimum($r2) + 2 : length($r2), -minimum($r3) + 2 : length($r3)))
end


function do_computation(nsteps, ncells, tmax, ifirst, ilast, statelft, statergt, velocity, dt, fc, lc, flux, x, u)
    istep=0
    t=0.0
    # loop over timesteps
    while istep < nsteps && t < tmax
        # right boundary condition: outgoing wave
        @inbounds for ic=ncells:lc
            u[ic]=u[ncells-1]
        end
        # left boundary condition: specified value
        @inbounds for ic=fc:-1
            u[ic]=statelft
        end

        # upwind fluxes times dt (ie, flux time integral over cell side)
        # assumes velocity > 0
        vdt=velocity*dt
        @inbounds for ie=ifirst:ilast+1
            flux[ie]=vdt*u[ie-1]
        end

        # conservative difference
        @inbounds for ic=ifirst:ilast
            u[ic] -= (flux[ic+1]-flux[ic]) / (x[ic+1]-x[ic])
        end

        # update time and step number
        t=t+dt
        istep=istep+1
    end
    u
end

function main()

    ncells = 10000

    #   integer nsteps
    #   double precision cfl,tmax
    #   double precision jump,x_left,x_right,statelft,statergt,velocity
    #   double precision
    #  &  u(-2:ncells+1),
    #  &  x(0:ncells),
    #  &  flux(0:ncells)

    u    = @eval @shifted_array(Float64, -2:$ncells+1)
    x    = @eval @shifted_array(Float64,  0:$ncells)
    flux = @eval @shifted_array(Float64,  0:$ncells)

    #   integer fc,lc,ifirst,ilast
    #   integer ic,ie,ijump,istep
    #   double precision dt,dx,frac,mindx,t,vdt
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # problem-specific parameters:
    #      tic()
    jump     =  0.0
    x_left   = -0.2
    x_right  =  1.0
    statelft =  2.0
    statergt =  0.0
    velocity =  1.0

    nsteps   =  10000
    tmax     =  0.8
    cfl      =  0.9

    # array bounds:
    fc=-2
    lc=ncells+1
    ifirst=0
    ilast=ncells-1

    #  uniform mesh:
    dx=(x_right-x_left)/ncells
    @inbounds for ie in ifirst:ilast+1
        x[ie]=x_left+ie*dx
    end

    # initial values for diffential equation:
    ijump=max(ifirst-1,min(convert(Int,round(ncells*(jump-x_left)/(x_right-x_left))),ilast+1))
    # left state to left of jump
    @inbounds for ic=ifirst:ijump-1
        u[ic]=statelft
    end
    # volume-weighted average in cell containing jump
    frac=(jump-x_left-ijump*dx)/(x_right-x_left)
    u[ijump]=statelft*frac+statergt*(1.0-frac)
    # right state to right of jump
    @inbounds for ic=ijump+1:ilast
        u[ic]=statergt
    end

    # stable timestep (independent of time for linear advection):
    mindx=1.0e300
    @inbounds for ic=ifirst:ilast
        mindx=min(mindx,x[ic+1]-x[ic])
    end

    dt = cfl*mindx/abs(velocity)

    u = do_computation(nsteps, ncells, tmax, ifirst, ilast, statelft, statergt, velocity, dt, fc, lc, flux, x, u)

    # write final results (plot later)
    @inbounds for ic=0:ncells-1
        xc = (x[ic]+x[ic+1])*0.5
        uc = u[ic]
        #@printf("%e %e\n",xc,uc)
    end
    # toc()
end # main

# jit compile main
#main()

@time main()
@time main()

#@profile main()
#Profile.print()
##
## #code_typed(main,())
##
## #code_native(main,())
