using ProductionScheduling
import ProductionScheduling: generate_expert_schedules
using DataFrames
using Plots
using PyCall
import Base.Threads: @threads, nthreads
import Graphs: outdegree, outneighbors
import Impute: interp

@pyimport numpy
@pyimport eta_utility as etautility

gr()

"""
Environment class representing the ETA Factory environment
"""
mutable struct Environment <: ProductionEnvironment
    "Settings for the optimization run."
    settings::EnvironmentSettings
    "Status of the optimization run."
    status::EnvironmentStatus

    "Structure of the production system."
    structure::ProductionSystem
    "Events for the production scheduling problem."
    events::EventsMaps

    "Buffer of the rewards for the last set of solutions."
    buffer_rewards::Matrix{Float64}
    "Buffer of the events for the last set of solutions."
    buffer_events::Matrix{Int}
    "Buffer of the variables for the last set of solutions."
    buffer_variables::Matrix{Int}
    "Buffer of the solutions sorted by front."
    buffer_fronts::Union{Vector{Vector{Int}}, Matrix{Int}}

    "DataFrame containing some scenario data like energy prices."
    scenario_data::DataFrame
    "Efficiency factor for cooling."
    cooling_efficiency::Float64
    "Efficiency factor for heating."
    heating_efficiency::Float64
    "Wait time in seconds before machines are switched to standby mode."
    wait_before_standby::Int
    "Average starting temperature of machine when the optimization starts."
    starting_temps::Vector{Float64}
    "Conversion factor for the energy prices from scenario data (converts prices to €/Ws)."
    price_conversion::Float64

    function Environment(pyenv::PyObject)
        # Read general settings from the python environment.
        settings = EnvironmentSettings(pyenv)

        # Read the production system structure from configuration files.
        productionsystem =
            ProductionSystem(settings.pathscenarios, pyenv.machines_file, pyenv.products_file, pyenv.orders_file)

        # Create the event and variables mappings.
        events = EventsMaps(productionsystem, pyenv.varsecondmap)

        scenario_data = py_import_scenario(settings, pyenv.scenario_paths)
        scenario_data[:, :ambient_halltemperature] = interp(scenario_data[!, :ambient_halltemperature])

        starting_temps = Vector{Float64}(undef, length(productionsystem.machines))
        for (machine_id, temp) in pairs(pyenv.starting_temps)
            machine_idx = nothing
            for (m_idx, m) in pairs(productionsystem.machines)
                if m.id == parse(Int, machine_id)
                    machine_idx = m_idx
                    break
                end
            end
            starting_temps[machine_idx] = temp
        end

        # Instantiate the environment
        env = new(
            settings,
            EnvironmentStatus(pyenv),
            productionsystem,
            events,
            Matrix{Float64}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Vector{Vector{Int}}(undef, 0),
            scenario_data,
            pyenv.cooling_efficiency,
            pyenv.heating_efficiency,
            pyenv.wait_before_standby,
            starting_temps,
            pyenv.price_conversion,
        )

        # Set the action and observation spaces of the python environment.
        action_space = py_dict_space(
            Dict(
                "events" => py_discrete_space(length(env.events.to_schedule)),
                "variables" =>
                    py_multidiscrete_space(fill(length(env.events.varsecondmap), length(env.events.to_schedule))),
            ),
        )
        observation_space = py_box_space(0, numpy.inf, shape=(length(env.events.events),), dtype=numpy.float32)
        pyenv_set_spaces!(env.settings, action_space, observation_space)

        return env
    end

    function Environment(
        settings::EnvironmentSettings,
        status::EnvironmentStatus,
        machines_file::AbstractString,
        products_file::AbstractString,
        orders_file::AbstractString,
        scenario_paths::Vector{Dict{String, T}},
        varsecondmap::Vector{Int},
        cooling_efficiency::Number,
        heating_efficiency::Number,
        wait_before_standby::Number,
        starting_temps::Dict{String, N},
        price_conversion::Number,
    ) where {T <: Any, N <: Number}
        productionsystem = ProductionSystem(settings.pathscenarios, machines_file, products_file, orders_file)
        events = EventsMaps(productionsystem, varsecondmap)
        scenario_data = py_import_scenario(settings, scenario_paths)
        scenario_data[:, :ambient_halltemperature] = interp(scenario_data[!, :ambient_halltemperature])

        _starting_temps = Vector{Float64}(undef, length(productionsystem.machines))
        for (machine_id, temp) in pairs(starting_temps)
            machine_idx = nothing
            for (m_idx, m) in pairs(productionsystem.machines)
                if m.id == parse(Int, machine_id)
                    machine_idx = m_idx
                    break
                end
            end
            _starting_temps[machine_idx] = temp
        end

        new(
            settings,
            status,
            productionsystem,
            events,
            Matrix{Float64}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Matrix{Int}(undef, 0, 0),
            Vector{Vector{Int}}(undef, 0),
            scenario_data,
            cooling_efficiency,
            heating_efficiency,
            wait_before_standby,
            _starting_temps,
            price_conversion,
        )
    end
end

"""
Perform an environment step.

:param env: The environment

:param actions: Actions as determined by the optimization algorithm.

:return: Tuple of observations, reward, done and info.
"""
function step!(env::Environment, actions)
    env.status.nsteps += 1
    nactions = size(actions)[1]

    if nactions > 1
        observations = Matrix{Float64}(undef, (nactions, length(env.events.events)))
        rewards = Matrix{Float64}(undef, nactions, 2)
        dones = trues(nactions)
        infos = Vector{Dict{String, Any}}(undef, nactions)

        # Copy events and variables to make them threadsafe
        events = Matrix{Int}(undef, (nactions, length(env.events.to_schedule)))
        variables = Matrix{Int}(undef, (nactions, length(env.events.to_schedule)))
        for i in 1:nactions
            events[i, :], variables[i, :] = actions[i]
        end

        @threads for i in 1:nactions
            observations[i, :], rewards[i, :], dones[i], infos[i] = calcreward(env, events[i, :], variables[i, :])
        end
    else
        events, variables = actions
        observations, rewards, dones, infos = calcreward(env, events, variables)
    end

    @debug log_step_errors(infos)

    return observations, rewards, dones, infos
end

"""
Update the environment state. Implemented here only for compatibility with the eta_utility interface.

:param env: The environment.
"""
function update!(env::Environment) end

"""
Calcuate the reward for a solution that is returned to the agent.

:param env: The environment
:param events: Array of actions as determined by the agent.
:param variables: Array of variables as determined by the agent.
"""
function calcreward(env::Environment, events, variables)
    schedule, machinegraph, machinestartingnodes, error = buildschedule(env.events, env.structure, events, variables)

    # Store info object
    info = Dict{String, Any}()
    if !isnothing(error)
        info["error"] = error
        info["valid"] = false

        return zeros(Float64, length(env.events.events)), [Inf, Inf], false, info
    end

    # Get the separate schedules for all machines.
    machineschedules = machineschedule(env.events, schedule, machinegraph, machinestartingnodes)

    # Calculate makespan
    makespan = schedule_makespan(machineschedules)

    # If the makespan exceeds the scenario duration, the solution is not allowable because in that case, 
    # the energy consumption cannot be calculated.
    if makespan >= env.settings.scenarioduration
        info["error"] = "Exceeded maximum allowable makespan (scenario_duration)."
        info["valid"] = true

        return ones(Float64, length(env.events.events)), [makespan, Inf], false, info
    end

    electricity_cost = sum(
        map(
            c -> sum(
                electric_power_consumption(env, c) .* env.scenario_data[1:makespan, :electrical_energy_price] .*
                env.price_conversion,
            ),
            values(machineenergy(env, machineschedules, makespan)),
        ),
    )

    return ones(Float64, length(env.events.events)), [makespan, electricity_cost], false, info
end

"""
Calculate the makespan of a production schedule

:param machineschedules: Schedules for all machines.
:return: Makespan
"""
schedule_makespan(machineschedules) = maximum(map(x -> length(x) > 0 ? last(x).endtime : 0, machineschedules))

"""
Calculate the energy consumption of a machine over the course of a production schedule.

:param env: The Environment
:param machineschedules: Schedules for all machines
:param makespan: Total duration of the production schedule.
:return: Vector of vectors of energy consumptions for all machines.
"""
function machineenergy(env::Environment, machineschedules, makespan)
    # Calculate the energy consumption of the solution
    energy_consumption = Vector{ModelResults}(undef, length(machineschedules))

    for (machine, machineschedule) in pairs(machineschedules)
        a_st, a_op, a_wk, z_proc =
            machinestates(env, env.structure.machines[machine], machineschedule, env.structure.products, makespan)

        energy_consumption[machine] = forward_model(
            env.structure.machines[machine],
            a_st=a_st,
            a_op=a_op,
            a_wk=a_wk,
            z_proc=z_proc,
            T_u=view(env.scenario_data, 1:makespan, :ambient_halltemperature),
            T_M1=env.starting_temps[machine],
        )
    end

    return energy_consumption
end

"""
Calculate the states of a machine over the course of a production schedule.

:param env: The environment.
:param machine: The machine object.
:param machineschedule: A schedule for that machine.
:param products: Products produced in the production system.
:param makespan: Total duration of the production schedule.
"""
function machinestates(
    env::Environment,
    machine::Machine,
    machineschedule::Vector{ScheduleItem},
    products::Vector{Product},
    makespan,
)
    a_st = falses(makespan)
    a_op = falses(makespan)
    a_wk = falses(makespan)
    z_proc = zeros(Float64, makespan)

    previousendtime = 1
    for item in machineschedule
        # Skip item if another coinciding item was already considered.
        if item.starttime < previousendtime
            continue
        end

        if item.starttime - previousendtime >= env.wait_before_standby
            a_st[previousendtime:item.starttime] .= true
        else
            a_op[previousendtime:item.starttime] .= true
        end

        a_op[item.starttime:item.starttime+item.setuptime] .= true
        a_wk[item.starttime+item.setuptime:item.endtime] .= true
        z_proc[item.starttime+item.setuptime:item.endtime] .= process_param(machine, products, item)

        previousendtime = item.endtime
    end

    return a_st, a_op, a_wk, z_proc
end
"""
Get the process-dependent parameter for a machine tool.

:param MachineTool: MachineTool object.
:param products: List of products produced in the production system.
:param item: ScheduleItem for which the process-dependent parameter is needed.
:return: Process-dependent parameter.
"""
process_param(::MachineTool{T}, products::Vector{Product}, item::ScheduleItem) where {T <: ParameterData} =
    products[item.product].operations[item.operation].processparam

"""
Get the process-dependent parameter for a cleaning machine.

:param CleaningMachine: CleaningMachine object.
:param products: List of products produced in the production system.
:param item: ScheduleItem for which the process-dependent parameter is needed.
:return: Process-dependent parameter.
"""
process_param(::CleaningMachine{T}, products::Vector{Product}, item::ScheduleItem) where {T <: ParameterData} =
    length(item.coinciding)

"""
Calculate the total electric energy consumption of a machine tool from the energy model results object.

:param env: The environment.
:param data: Energy model results.
:return: Timeseries of electric energy consumption.
"""
electric_power_consumption(env::Environment, data::MachineToolResults) =
    data.P_el + data.P_th_c * 1 / env.cooling_efficiency

"""
Calculate the total electric energy consumption of a cleaning machine from the energy model results object.

:param env: The environment.
:param data: Energy model results.
:return: Timeseries of electric energy consumption.
"""
function electric_power_consumption(env::Environment, data::CleaningMachineResults)
    if data.electric
        data.P_el
    else
        (data.P_el + data.P_th * 1 / env.heating_efficiency)
    end
end

"""
Render the optimization results in a scatter plot, a Gannt chart and some energy plots.

:param env: The environment.
:param mode: Parameter to determine rendering mode (only here for compatibility with eta_utility).
"""
function render(
    env::Environment,
    mode="human";
    path=nothing,
    filename=nothing,
    fileextension="png",
    debug_annotations=true,
)
    path = isnothing(path) ? env.settings.pathresults : path
    if isnothing(filename)
        env.status.nepisodes += 1
        filename::String =
            etautility.eta_x.common.episode_name_string(env.settings.runname, env.status.nepisodes, env.settings.envid)
    end

    plotargs = Dict(
        :plot_titlefontsize => 11,
        :annotation_fontsize => 9,
        :formatter => :plain,
        :fontfamily => "Palatino Roman",
        :margin => (1, :mm),
        :legend_background_color => RGBA(1, 1, 1, 0.7),
        :size => (700, 800),
        :dpi => 150,
        :size => (
            321.51616 * 120 * 0.01384, # width in pt * dpi * inches per pt
            469.4704 * 120 * 0.01384 * 0.95, # heigth in pt * dpi * inches per pt * 0.95
        ),
    )

    if typeof(env.buffer_fronts) <: Matrix
        env.buffer_fronts = [env.buffer_fronts[i, :] for i in 1:size(env.buffer_fronts)[1]]
    end

    if length(env.buffer_rewards) > 0
        thisplotargs = copy(plotargs)
        thisplotargs[:size] = (first(plotargs[:size]) * 0.7, last(plotargs[:size]) * 0.3)
        plot_solspace = render_solspace(env, debug_annotations; thisplotargs...)
        savefig(plot_solspace, joinpath(path, "$(filename)_solutionspace.$fileextension"))

        if mode == "all"
            for front in env.buffer_fronts, solution in front .+ 1
                plot_schedule = render_schedule(env, solution, debug_annotations; plotargs...)
                savefig(plot_schedule, joinpath(path, "$(filename)_$solution.$fileextension"))
                println(
                    "$(filename)_$solution - MKSP: $(env.buffer_rewards[solution, 1]), ERC: $(env.buffer_rewards[solution, 2])",
                )
            end

        else
            solution = typeof(mode) <: String ? 1 : mode
            plot_schedule = render_schedule(env, solution, debug_annotations; plotargs...)
            savefig(plot_schedule, joinpath(path, "$(filename)_$solution.$fileextension"))
            println(
                "$(filename)_$solution - MKSP: $(env.buffer_rewards[solution, 1]), ERC: $(env.buffer_rewards[solution, 2])",
            )
        end
    end
end

"""
Render the solution space determined by the agent.

:param env: The environment.
"""
function render_solspace(env::Environment, debug_annotations::Bool; kwargs...)
    maxfront = 8
    plt = plot(; xlabel="Makespan in s", ylabel="Energy cost in €", palette=:okabe_ito, kwargs...)

    for (num, front) in enumerate(env.buffer_fronts)
        # Only show the first eight fronts
        num > maxfront && break
        scatter!(plt, env.buffer_rewards[front.+1, 1], env.buffer_rewards[front.+1, 2], label="front $num")
    end

    if debug_annotations
        for (num, front) in enumerate(env.buffer_fronts)
            # Only show the first eight fronts
            num > maxfront && break
            # Plot only annotations, moved up slightly compared to the actual markers.
            scatter!(
                plt,
                env.buffer_rewards[front.+1, 1],
                env.buffer_rewards[front.+1, 2] .+ ((last(ylims(plt)) - first(ylims(plt))) / 60),
                markeralpha=0,
                color=:lightgrey,
                label=false,
                series_annotations=text.(front .+ 1, :bottom, 7, "Palatino Roman"),
            )
        end
    end

    return plt
end

"""
Render a specific schedule as a gannt chart.

:param env: The environment.
:param solutionidx: Index of the solution to be rendered.
"""
function render_schedule(env::Environment, solutionidx::Int, debug_annotations::Bool; kwargs...)
    schedule, machinegraph, machinestartingnodes, err = buildschedule(
        env.events,
        env.structure,
        env.buffer_events[solutionidx, :],
        env.buffer_variables[solutionidx, :],
    )
    if !isnothing(err)
        errtext = "The requested solution '$solutionidx' is not valid and cannot be rendered: $err"
        @error errtext

        plt = plot(; kwargs...)
        annotate!(plt, [(0.5, 0.5, (errtext, 11, :red, :center))])
        return plt
    end

    machineschedules = machineschedule(env.events, schedule, machinegraph, machinestartingnodes)
    makespan = schedule_makespan(machineschedules)

    plot_schedule = plot(legend_columns=3)
    yticklabels = String[]

    # Setup legend
    bar!(plot_schedule, (-10, -10), orientation=:h, color=palette(:Pastel2)[3], label="Pause Time")
    bar!(plot_schedule, (-10, -10), orientation=:h, color=palette(:Pastel2)[4], label="Setup Time")
    labels = falses(length(env.structure.products))

    for (machine, schedule) in pairs(machineschedules)
        push!(yticklabels, env.structure.machines[machine].name)
        starts = Int[]
        texts = String[]
        for item in schedule
            # Preceding pause
            bar!(
                plot_schedule,
                (machine, item.starttime),
                fillrange=item.starttime - item.pausetime,
                bar_width=0.2,
                orientation=:h,
                color=palette(:Pastel2)[3],
                label=false,
            )

            # Actual operation
            bar!(
                plot_schedule,
                (machine, item.endtime),
                fillrange=item.starttime + item.setuptime,
                bar_width=0.7,
                orientation=:h,
                color=palette(:okabe_ito)[item.product],
                label=labels[item.product] ? false : env.structure.products[item.product].name,
            )
            labels[item.product] = true
            if length(item.coinciding) > 1
                push!(starts, item.starttime + item.setuptime)
                push!(texts, "$(length(item.coinciding))")
            end

            # Smaller line for preparation time 
            bar!(
                plot_schedule,
                (machine, item.starttime + item.setuptime),
                fillrange=item.starttime,
                bar_width=0.2,
                orientation=:h,
                color=palette(:Pastel2)[4],
                label=false,
            )
        end

        if !isempty(starts)
            annotate!(
                plot_schedule,
                starts .+ 0.005 * makespan,
                machine + 0.32,
                text.(texts, :left, :top, 7, "Palatino Roman"),
            )
        end
    end
    yaxis!(
        plot_schedule,
        ylims=(0.4, length(yticklabels) + 0.6 + (0.2 * length(yticklabels))),
        yticks=(1:length(yticklabels), yticklabels),
    )

    plot_energy = plot(ylabel="Electric power in W", palette=:okabe_ito, legend=false)
    if makespan <= size(env.scenario_data)[1]
        electric_power = zeros(Float64, length(machineschedules), makespan)
        for (machine, energy) in pairs(machineenergy(env, machineschedules, makespan))
            electric_power[machine, :] = electric_power_consumption(env, energy)
        end
        plot!(plot_energy, vec(sum(electric_power, dims=1)))

        plot_prices = plot(
            env.scenario_data[1:makespan, :electrical_energy_price],
            ylabel="Electric energy price in €/MWh",
            xlabel="Time in s",
            label="energy price",
            palette=:okabe_ito,
            legend_position=:topleft,
        )
        plot!(plot_prices, -10, -10, label="cumulated cost")

        plot_energycost = twinx(plot_prices)
        plot!(
            plot_energycost,
            cumsum(
                sum(electric_power, dims=1)[1, :] .* env.scenario_data[1:makespan, :electrical_energy_price] .*
                env.price_conversion,
            ),
            ylabel="Electric energy cost in €",
            label=false,
        )

    else
        annotate!(
            plot_energy,
            [(0.5, 0.5, ("could not plot energy, scenario data vector too short.", 11, :red, :center))],
        )
        plot_prices = plot(
            ylabel="Electric energy price in €/MWh",
            xlabel="Time in s",
            palette=:okabe_ito,
            legend_position=:topleft,
        )
    end

    return plot(plot_schedule, plot_energy, plot_prices; layout=@layout[°; °; °], kwargs...)
end

"""
Reset the environment state.

:param env: The environment.
"""
function reset!(env::Environment)
    ProductionScheduling.reset!(env.status, env.settings)
    env.buffer_rewards = Matrix{Float64}(undef, 0, 0)
    env.buffer_events = Matrix{Int}(undef, 0, 0)
    env.buffer_variables = Matrix{Int}(undef, 0, 0)
    env.buffer_fronts = Vector{Vector{Int}}(undef, 0)

    observations = zeros(Float64, env.settings.pyenv."observation_space"."shape"[1])

    return observations
end

"""
Reset the environment state.

:param env: The environment.
"""
function reset!(env::Environment)
    ProductionScheduling.reset!(env.status, env.settings)
    env.buffer_rewards = Matrix{Float64}(undef, 0, 0)
    env.buffer_events = Matrix{Int}(undef, 0, 0)
    env.buffer_variables = Matrix{Int}(undef, 0, 0)
    env.buffer_fronts = Vector{Vector{Int}}(undef, 0)

    observations = zeros(Float64, env.settings.pyenv."observation_space"."shape"[1])

    return observations
end

"""
Update the environment state with observations if it interacts with another environment. Not implemented.
"""
function first_update!(env::Environment) end

"""
Close the environment after the optimization is done.

:param env: The environment.
"""
function close!(env::Environment) end

generate_expert_schedules(env::Environment, method::String, count::Int=1) =
    generate_expert_schedules(env.settings, env.events, env.structure, method, count)
