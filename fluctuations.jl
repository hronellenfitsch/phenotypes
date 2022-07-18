using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Statistics

using VoronoiDelaunay

function ss_solve(f, x0; atol=1e-8, rtol=1e-6, Δt=0.1, maxint=100000)
    """ Steady state solver using Euler method
    """
    x = copy(x0)
    fx = f(x, 0.0)

    converged = false

    for i=1:maxint
        t = i*Δt
        fx_new = f(x, t)

        Δfx = norm(fx - fx_new)
        if (Δfx < rtol*norm(fx)) | (Δfx < atol)
            converged = true
            break
        end

        fx = fx_new

        # time step
        x = x + Δt*fx
    end

    return x, converged
end

function adaptation_ode(K, t, netw, currents_function, κ, β, ρ)
    F = currents_function(K, netw)
    return F.^β .- K .+ κ*exp(-t/ρ)
end

function static_currents(K, netw; i=1)
    E = netw.E[2:end,:]

    L = E*spdiagm(0 => K)*E'
    q = -ones(netw.N_v)./(netw.N_v - 1)
    q[i] = 1.0

    q_reduced = @view q[2:end]

    p = cholesky(L) \ q_reduced
    F = K.*(E'*p)

    return F.^2
end

function correlated_currents_fun(netw, f; ind=1)
    pos = netw.pos
    D = [norm(pos[i,:] - pos[j,:]) for i=1:netw.N_v, j=1:netw.N_v]
    Q = -f.(D)

    # normalize
    Q[ind,:] .= 0.0
    sums = sum(Q, dims=1)
    Q ./= -sums
    Q[ind,:] .= 1.0

    Q_reduced = Q[2:end,:]

    function currents_fun(K, netw)
        E = netw.E[2:end,:]
        L = E*spdiagm(0 => K)*E'

        p = cholesky(L) \ Q_reduced
        F = K.*(E'*p)

        return mean(F.^2, dims=2)[:,1]
    end
end

function corson_currents_fun(ee, netw; ind=1)
    pos = netw.pos
    Q = zeros(netw.N_v, netw.N_v)

    # fluctuation part
    Q = Q .- (1.0 - ee)
    for i=1:netw.N_v
        Q[i,i] -= ee
    end

    # remove ind
    Q = Q[:,1:end .!= ind]

    Q[ind,:] .= 0.0
    sums = sum(Q, dims=1)
    Q ./= -sums
    Q[ind,:] .= 1.0
    Q_reduced = Q[2:end,:]

    function currents_fun(K, netw)
        E = netw.E[2:end,:]
        L = E*spdiagm(0 => K)*E'

        p = cholesky(L) \ Q_reduced
        F = K.*(E'*p)

        return mean(F.^2, dims=2)[:,1]
    end
end

function random_currents_fun(p, netw; ind=1)
    Q = [rand() < p ? 1.0 : 0.0 for i=1:netw.N_v, j=1:netw.N_v]

    # normalize
    Q[ind,:] .= 0.0
    sums = sum(Q, dims=1)
    Q ./= -sums
    Q[ind,:] .= 1.0

    Q_reduced = Q[2:end,:]

    function currents_fun(K, netw)
        E = netw.E[2:end,:]
        L = E*spdiagm(0 => K)*E'

        p = cholesky(L) \ Q_reduced
        F = K.*(E'*p)

        return mean(F.^2, dims=2)[:,1]
    end
end

struct Network
    pos
    edgelist
    E
    N_e
    N_v
    mean_len
    lengths
end

function network_from_txt(edges, nodes)
    edgelist = Int64.(readdlm(edges, ' ') .+ 1.0)
    pos = readdlm(nodes, ' ')

    # remove overly long edges
    lengths = [norm(pos[edgelist[i,1],:] - pos[edgelist[i,2],:]) for i=1:size(edgelist)[1]]
    mean_len = mean(lengths)

    # sparse adjacency matrix
    N_e = size(edgelist)[1]
    N_v = size(pos)[1]

    J = vcat(1:N_e, 1:N_e)
    I = vcat(edgelist[:,1], edgelist[:,2])
    V = vcat(ones(N_e), -ones(N_e))

    E = sparse(I, J, V)

    Network(pos, edgelist, E, N_e, N_v, mean_len, lengths)
end

function network_from_txt(fname; zoom=1.0)
    positions = readdlm(fname, ',') .+ 1.0
    positions = vcat([[positions[i,1] positions[i,2]] for i = 1:size(positions)[1]
            if (positions[i,1] - 1.5)^2 + (positions[i,2] - 1.5)^2 <= (zoom*0.5)^2]...)

    points = [Point2D(positions[i,1], positions[i,2]) for i=1:size(positions)[1]]

    # Delaunay triangulation
    tess = DelaunayTessellation()
    push!(tess, points)

    edgelist = []
    for edge in delaunayedges(tess)
        a, b = geta(edge), getb(edge)

        ind_a = findall(x -> x==a, points)[1]
        ind_b = findall(x -> x==b, points)[1]

        push!(edgelist, [ind_a ind_b])
    end

    # real positions because tesselation changes the points list
    pos = vcat([[p._x p._y] for p in points]... )

    # remove overly long edges
    lengths = [norm(pos[a,:] - pos[b,:]) for (a, b) in edgelist]
    mean_len = mean(lengths)
    edgelist = [e for (e, l) in zip(edgelist, lengths) if l < 1.5mean_len]

    # new mean length
    lengths = [norm(pos[a,:] - pos[b,:]) for (a, b) in edgelist]
    mean_len = mean(lengths)

    # sparse adjacency matrix
    N_e = length(edgelist)
    N_v = length(points)

    edgelist = vcat(edgelist...)

    J = vcat(1:N_e, 1:N_e)
    I = vcat(edgelist[:,1], edgelist[:,2])
    V = vcat(ones(N_e), -ones(N_e))

    E = sparse(I, J, V)

    Network(pos, edgelist, E, N_e, N_v, mean_len, lengths)
end

# analysis functions
function steady_state_dissipation(K, netw; ind=1)
    F2 = static_currents(K, netw; i=ind)
    nonz = (K .> 1e-8)

    return sum(netw.lengths[nonz].*F2[nonz]./K[nonz])
end

function area_penalty(K, netw; ind=1)
    K = copy(K)
    K[K .< 1e-7] .= 0.0

    E = netw.E[2:end,:]
    # flows = areas
    L = E*spdiagm(0 => K)*E'
    L_chol = cholesky(L)

    q = -ones(netw.N_v)
    q[ind] = netw.N_v - 1

    q_reduced = @view q[2:end]

    p = L_chol \ q_reduced
    F = abs.(K.*(E'*p))

    # compute diagonal elements of S directly
    S = (L_chol \ E).*E
    s = K.*sum(S, dims=2)[1,:]

    # find edges where denominator diverges
    divergent = s .> 1 - 1e-5

    # areas
    penalty = mean(F[divergent]/netw.N_v)
    return isnan(penalty) ? 0.0 : penalty
end

function cost(K, netw; γ=0.5)
    return sum(netw.lengths.*K.^γ)
end

function network_indices(netw)
    """ Return the index of the leftmost and center node
    """
    i_left = argmin(netw.pos[:,1])
    com = mean(netw.pos, dims=1)[1,:]
    i_center = argmin([norm(netw.pos[i,:] .- com) for i=1:netw.N_v])

    return Dict(:center => i_center, :left => i_left)
end
