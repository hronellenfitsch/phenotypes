using Distributed
using BSON
using Random

include("fluctuations.jl")

function run_simulations_gauss(γ)
    println("γ = $(γ)")
    params = Dict(
        :N_samples => 2,
        :N_kappa => 5,
        :N_rho => 5,
        :N_σ => 40,
        :σ_min => 0.1,
        :σ_max => 5.0,
        :fluctuations => :gauss,
        :source => :left,
        :beta => 1.0/(1 + γ)
    )
    name = "gauss_gamma_$(γ)_$(abs(rand(Int32)))"
    println(name)

    N_samples = params[:N_samples]

    combinations = []
    for κ=10.0.^LinRange(-2, 0, params[:N_kappa])
        for ρ=10.0.^LinRange(0, 2, params[:N_rho])
            for σ=LinRange(params[:σ_min], params[:σ_max], params[:N_σ])
                for i=1:N_samples
                    push!(combinations, [κ, ρ, σ])
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
    results = []
    for (j, c) in collect(enumerate(combinations))
        κ, ρ, σ = c
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
        C = cost(K, netw; γ=γ)
        C_half = cost(K, netw; γ=0.5)

        println("$(j)/$(maxi)  κ=$(κ), ρ=$(ρ), σ=$(σ). converged: $(converged)")
        println("P_ss = $(P_ss), A = $(A), C=$(C)")

        res = Dict(:K => K,
            :converged => converged,
            :P_ss => P_ss,
            :A => A,
            :C => C,
            :C_half => C_half,
            :κ => κ,
            :ρ => ρ,
            :σ => σ)
        push!(results, res)
    end

    bson("ben_data/results_$(name).bson", Dict(:network => netw,
                             :results => results,
                             :parameters => params))
end

run_simulations_gauss(parse(Float64, ARGS[1]))
