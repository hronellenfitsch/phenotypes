using Distributed
using BSON
using Random

# seed is process id.
Random.seed!(myid())

@everywhere include("fluctuations.jl")

@everywhere function run_simulations_corson(source)

    if source == "center"
        src = :center
    elseif source == "left"
        src = :left
    else
        println("Unknown source!")
    end

    params = Dict(
        :N_samples => 2,
        :N_kappa => 5,
        :N_rho => 5,
        :N_e => 20,
        :e_min => log10(0.9),
        :e_max => 0.0,
        :fluctuations => :corson,
        :source => src,
        :beta => 1.0/(1 + 0.5)
    )
    name = "corson_$(abs(rand(Int32)))"

    N_samples = params[:N_samples]

    combinations = []
    for κ=10.0.^LinRange(-2, 0, params[:N_kappa])
        for ρ=10.0.^LinRange(0, 2, params[:N_rho])
            for e=10.0.^LinRange(params[:e_min], params[:e_max], params[:N_e])
                for i=1:N_samples
                    push!(combinations, [κ, ρ, e])
                end
            end
        end
    end

    # netw = network_from_txt("lattices/points2000_1.txt"; zoom=0.8)
    netw = network_from_txt("lattices/paper_edges.txt", "lattices/paper_nodes.txt")
    inds = network_indices(netw)

    i = inds[params[:source]]
    maxi = length(combinations)

    # Parallel compute
    results = @distributed (hcat) for (j, c) in collect(enumerate(combinations))
        κ, ρ, e = c
        # initial conditions
        K0 = -log10.(rand(netw.N_e))

        # fluctuation model
        if params[:fluctuations] == :gauss
            currents = correlated_currents_fun(netw,
                x->exp(-0.5x^2/(σ*netw.mean_len)^2); ind=i)
        elseif params[:fluctuations] == :exponential
            currents = correlated_currents_fun(netw,
                x->exp(abs.(x)/(σ*netw.mean_len)); ind=i)
        elseif params[:fluctuations] == :random
            currents = random_currents_fun(σ, netw; ind=i)
        elseif params[:fluctuations] == :corson
            currents = corson_currents_fun(e, netw; ind=i)
        else
            println("Error: unknown fluctuations.")
        end

        #
        f(K, t) = adaptation_ode(K, t, netw, currents, κ, params[:beta], ρ)
        K, converged = ss_solve(f, K0; Δt=1.0)

        P_ss = steady_state_dissipation(K, netw; ind=i)
        A = area_penalty(K, netw; ind=i)
        C = cost(K, netw)

        println("$(j)/$(maxi)  κ=$(κ), ρ=$(ρ), e=$(e). converged: $(converged)")
        println("P_ss = $(P_ss), A = $(A), C=$(C)")

        Dict(:K => K,
            :converged => converged,
            :P_ss => P_ss,
            :A => A,
            :C => C,
            :κ => κ,
            :ρ => ρ,
            :e => e)
    end

    bson("new_data/results_$(source)_$(name).bson", Dict(:network => netw,
                             :results => results,
                             :parameters => params))
end

run_simulations_corson(ARGS[1])
