using JuMP
using JSON
using Plots
using Statistics
using Dates

"""
Data used for the parameter estimation for machine tool models.
"""
struct MachineToolRegressionData <: RegressionData
    "Name of the electric power consumption timeseries."
    P_el::Symbol
    "Name of the thermal power consumption timeseries."
    P_th::Symbol
    "Name of the standby mode active indicator (1 or 0) timeseries."
    a_st::Symbol
    "Name of the operational mode active indicator (1 or 0) timeseries."
    a_op::Symbol
    "Name of the working mode active indicator (1 or 0) timeseries."
    a_wk::Symbol
    "Name of the material removal rate timeseries."
    z_proc::Symbol
    "Name of the external temperature timeseries."
    T_u::Symbol
    "Name of the coolant tempreature timeseries."
    T_c::Symbol
    "Name of the machine temperature timeseries."
    T_M::Symbol
    "Regression Periods"
    periods::Union{Nothing, Function}
end

"""
Parameter data for machine tool models, available after parameter estimation.
"""
struct MachineToolParameterData <: ParameterData
    "Regression parameter for electric energy."
    β_el::Vector{Float64}
    "Regression parameter for cooling energy."
    β_th_c::Float64
    "Regression parameter for external energy losses."
    β_th_o::Float64
    "Regression parameter for machine heat capacity."
    β_c_M::Float64
    "Average coolant temperature."
    T_c::Float64
end

"""
Object describing a machine tool in the factory.
"""
struct MachineTool{T} <: Machine where {T <: ModelData}
    "Name of the machine."
    name::String
    "Unique identifier of the machine."
    id::Int
    "Capacity of the machine (how many parts it can produce at once)."
    capacity::Int
    "Unique job flag (true if all coinciding jobs have to be the same operation.)"
    unique_job::Bool
    "Either parameter or regression data object."
    data::T
end

"""
Instantiate MachineTool object with the names timeseries names for regression.
"""
MachineTool(name; id, P_el, P_th, a_st, a_op, a_wk, z_proc, T_u, T_c, T_M, periods) = MachineTool(
    name,
    id,
    1,
    true,
    MachineToolRegressionData(P_el, P_th, a_st, a_op, a_wk, z_proc, T_u, T_c, T_M, periods),
)

"""
Instantiate MachineTool object with parameter data from a file.
"""
function MachineTool(id::Int, filename::AbstractString)
    objects = JSON.parsefile(filename)
    for obj in objects
        if obj["resource_id"] == id
            return MachineTool(
                obj["name"],
                obj["resource_id"],
                obj["capacity"],
                obj["unique_job"],
                MachineToolParameterData(
                    obj["parameters"]["beta_el"],
                    obj["parameters"]["beta_th_c"],
                    obj["parameters"]["beta_th_o"],
                    obj["parameters"]["beta_c_M"],
                    obj["parameters"]["T_c"],
                ),
            )
        end
    end
    return nothing
end

"""
Create a regression model for the machine.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:return: JuMP model.
"""
function regression_model(machine::MachineTool{T}, data::DataFrame) where {T <: RegressionData}
    model = Model()
    names = machine.data

    values = dropmissing(
        (isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods)),
        regression_colnames(names),
    )
    timesteps = size(values)[1]

    # Parameters
    a_st = Vector{Bool}(values[:, names.a_st])
    a_op = Vector{Bool}(values[:, names.a_op])
    a_wk = Vector{Bool}(values[:, names.a_wk])
    z_proc = Vector{Float64}(values[:, names.z_proc])
    P_el = Vector{Float64}(values[:, names.P_el])
    P_th_c = Vector{Float64}(values[:, names.P_th])
    T_u = Vector{Float64}(values[:, names.T_u])
    T_c = Vector{Float64}(values[:, names.T_c])
    T_M = Vector{Float64}(values[:, names.T_M])

    # Variables for electric power consumption
    @variables(model, begin
        β_el[1:4]
        ϵ_el[1:timesteps]
    end)

    @constraints(
        model,
        begin
            c_el[t=1:timesteps],
            P_el[t] == (
                (a_st[t] + a_op[t] + a_wk[t]) * β_el[1] +
                (a_op[t] + a_wk[t]) * β_el[2] +
                a_wk[t] * (z_proc[t] * β_el[3] + β_el[4]) +
                ϵ_el[t]
            )
        end
    )

    # Variables and constraints for thermal cooling power
    @variables(model, begin
        β_th_c
        ϵ_th_c[1:timesteps]
    end)
    @constraints(model, begin
        c_th_c[t=1:timesteps], P_th_c[t] == β_th_c * (T_M[t] - T_c[t]) + ϵ_th_c[t]
    end)

    # Variables and constraints for heat flux to environment
    @variables(model, begin
        P_th_o[1:timesteps]
        β_th_o
        ϵ_th_o[1:timesteps]
    end)
    @constraint(model, c_th_o[t=1:timesteps], P_th_o[t] == β_th_o * (T_M[t] - T_u[t]) + ϵ_th_o[t])

    # Constraint to connect thermal power and electric power of the machine
    @variables(model, begin
        β_c_M
        ϵ_c_M[1:timesteps]
    end)
    @constraints(
        model,
        begin
            c_th_M[t=1:timesteps-1],
            β_c_M * T_M[t+1] ==
            P_el[t] + (β_c_M - β_th_c - β_th_o) * T_M[t] + β_th_c * T_c[t] + β_th_o * T_u[t] + ϵ_c_M[t]
        end
    )

    @constraints(model, begin
        β_el[1:4] .>= 0
        β_th_c >= 0
        β_th_o >= 0
        β_c_M >= 0
    end)

    # Objective for the regression model is to reduce the quadratic loss function.
    @objective(model, Min, sum(ϵ_el[t]^2 + ϵ_th_c[t]^2 + ϵ_th_o[t]^2 + ϵ_c_M[t]^2 for t in 1:timesteps))
    return model
end

"""
Plot data collected to perform the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:return: A plot object.
"""
function plot_data(machine::MachineTool{T}, data::DataFrame; kwargs...) where {T <: RegressionData}
    l = @layout([°; °; °; °; °])
    names = machine.data

    values =
        coalesce.(
            (isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods))[
                !,
                union(regression_colnames(names), (:Timestamp,)),
            ],
            NaN64,
        )

    x_ticks = range(ceil(values[1, :Timestamp], Dates.Hour), floor(values[end, :Timestamp], Dates.Hour), step=Hour(2))
    x_tickformat = Dates.format.(x_ticks, "dd. u HH:MM")

    plt_power = plot(
        values[:, :Timestamp],
        values[:, names.P_el],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Electric power in W",
        label="electric power",
        palette=:okabe_ito,
        legend=false,
    )
    plt_thermal = plot(
        values[:, :Timestamp],
        values[:, names.P_th],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0, Inf),
        ylabel="Cooling power in W",
        label="cooling power",
        color=palette(:okabe_ito)[2],
        legend=false,
    )

    plt_act = plot(
        ylabel="Energy state",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0.4, 3.6),
        yticks=([1, 2, 3], [:standby, :operational, :working]),
        legend=false,
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_st],
        Dict(true => 1),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[1],
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_op],
        Dict(true => 2),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[2],
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_wk],
        Dict(true => 3),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[3],
    )

    ylimzproc = maximum(skipmissing(values[!, names.z_proc])) + 0.1 * maximum(skipmissing(values[!, names.z_proc]))
    plt_timevolume = plot(
        values[:, :Timestamp],
        values[:, names.z_proc],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Mat. removal rate in g/s",
        label="material removal rate",
        ylims=(0, ylimzproc),
        legend=false,
    )

    plt_temperature = plot(
        DateTime.(values[:, :Timestamp]),
        values[:, names.T_M],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        label="machine",
        ylabel="Temperature in °C",
        xlabel="Time",
        palette=:okabe_ito,
        ylims=(0, Inf),
    )
    plot!(plt_temperature, values[:, :Timestamp], values[:, names.T_c], label="coolant")
    plot!(plt_temperature, values[:, :Timestamp], values[:, names.T_u], label="air")

    plt = plot(plt_power, plt_thermal, plt_act, plt_timevolume, plt_temperature, layout=l; kwargs...)
    return plt
end

"""
Export the parameters identified during the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param model: The solved JuMP model containing the estimated parameters.
:return: Dictionary to be exported to a JSON file.
"""
function export_parameters(
    machine::MachineTool{T},
    model::JuMP.AbstractModel,
    data::DataFrame,
) where {T <: RegressionData}
    vars = object_dictionary(model)

    Dict(
        "type" => "machinetool",
        "name" => machine.name,
        "resource_id" => machine.id,
        "unique_job" => machine.unique_job,
        "capacity" => machine.capacity,
        "parameters" => Dict(
            "beta_el" => value.(vars[:β_el]),
            "beta_th_c" => value.(vars[:β_th_c]),
            "beta_th_o" => value.(vars[:β_th_o]),
            "beta_c_M" => value.(vars[:β_c_M]),
            "T_c" => mean(skipmissing(data[!, machine.data.T_c])),
        ),
    )
end

"""
Results vectors from a forward execution of the machine tool model.
"""
struct MachineToolResults <: ModelResults
    "Predicted electric power consumption."
    P_el::Vector{Union{Missing, Float64}}
    "Predicted cooling power consumption."
    P_th_c::Vector{Union{Missing, Float64}}
    "Perdicted thermal energy loss to the environment."
    P_th_o::Vector{Union{Missing, Float64}}
    "Predicted machine temperature."
    T_M::Vector{Union{Missing, Float64}}
end

"""
Prediction model for power consumption of the machine tool.

:param machine: The machine object containing the estimated parameters.
:param a_st: Standby mode active indicator (1 or 0) timeseries.
:param a_op: Operational mode active indicator (1 or 0) timeseries.
:param a_wk: Working mode active indicator (1 or 0) timeseries.
:param z_proc: Material removal rate timeseries.
:param T_u: External temperature timeseries.
:param T_M1: Machine temperature at the beginning.
"""
function forward_model(
    machine::MachineTool{T};
    a_st,
    a_op,
    a_wk,
    z_proc,
    T_u,
    T_M1::AbstractFloat,
) where {T <: ParameterData}
    β_el = machine.data.β_el
    β_th_c = machine.data.β_th_c
    β_th_o = machine.data.β_th_o
    β_c_M = machine.data.β_c_M
    T_c = machine.data.T_c

    P_el = (a_st + a_op + a_wk) .* β_el[1] + (a_op + a_wk) .* β_el[2] + a_wk .* (z_proc .* β_el[3] .+ β_el[4])

    T_M = Vector{Union{Missing, Float64}}(missing, length(a_st))
    T_M[1] = T_M1
    for t in 1:length(a_st)-1
        T_M[t+1] =
            (P_el[t] / β_c_M) +
            ((β_c_M - β_th_c - β_th_o) / β_c_M) * T_M[t] +
            (β_th_c / β_c_M) * T_c +
            (β_th_o / β_c_M) * T_u[t]
    end

    P_th_c = β_th_c .* (T_M .- T_c)
    P_th_o = β_th_o .* (T_M - T_u)

    return MachineToolResults(P_el, P_th_c, P_th_o, T_M)
end

"""
Results plot for a prediction made using the machine energy model.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:param results: Results vectors from a forward execution of the machine tool model.
"""
function plot_result(
    machine::MachineTool{T},
    data::DataFrame,
    result::MachineToolResults;
    kwargs...,
) where {T <: RegressionData}
    l = @layout([°; °; °; °; °])
    names = machine.data
    values = dropmissing(
        isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods),
        union(regression_colnames(names), (:Timestamp,)),
    )

    errors = Dict{String, Float64}(
        "RMSE_P_el" => sqrt(sum((result.P_el .- values[:, names.P_el]) .^ 2) / length(result.P_el)),
        "RMSE_P_th_c" => sqrt(sum((result.P_th_c .- values[:, names.P_th]) .^ 2) / length(result.P_th_c)),
        "RMSE_smooth_P_el" => sqrt(
            sum(skipmissing(result.P_el .- moving_average(values[:, names.P_el], 30)) .^ 2) / length(result.P_el),
        ),
        "RMSE_smooth_P_th_c" => sqrt(
            sum(skipmissing(result.P_th_c .- moving_average(values[:, names.P_th], 30)) .^ 2) / length(result.P_th_c),
        ),
        "RMSE_T_M" => sqrt(sum((result.T_M .- values[:, names.T_M]) .^ 2) / length(result.T_M)),
        "MAE_P_el" => sum(abs.(result.P_el .- values[:, names.P_el])) / length(result.P_el),
        "MAE_P_th_c" => sum(abs.(result.P_th_c .- values[:, names.P_th])) / length(result.P_th_c),
        "MAE_smooth_P_el" =>
            sum(abs.(skipmissing(result.P_el .- moving_average(values[:, names.P_el], 30)))) / length(result.P_el),
        "MAE_smooth_P_th_c" =>
            sum(abs.(skipmissing(result.P_th_c .- moving_average(values[:, names.P_th], 30)))) / length(result.P_th_c),
        "MAE_T_M" => sum(abs.(result.T_M .- values[:, names.T_M])) / length(result.T_M),
        "TotalEnergy_P_el" => (sum(result.P_el) - sum(values[:, names.P_el])) / sum(values[:, names.P_el]),
        "TotalEnergy_P_th_c" => (sum(result.P_th_c) - sum(values[:, names.P_th])) / sum(values[:, names.P_th]),
    )
    errors["RMSE"] =
        (
            errors["RMSE_P_el"] * length(result.P_el) +
            errors["RMSE_P_th_c"] * length(result.P_th_c) +
            errors["RMSE_T_M"] * length(result.T_M)
        ) / (length(result.P_el) + length(result.P_th_c) + length(result.T_M))
    errors["MAE"] =
        (
            errors["MAE_P_el"] * length(result.P_el) +
            errors["MAE_P_th_c"] * length(result.P_th_c) +
            errors["MAE_T_M"] * length(result.T_M)
        ) / (length(result.P_el) + length(result.P_th_c) + length(result.T_M))
    errors["RMSE_smooth"] =
        (
            errors["RMSE_smooth_P_el"] * length(result.P_el) +
            errors["RMSE_P_th_c"] * length(result.P_th_c) +
            errors["RMSE_T_M"] * length(result.T_M)
        ) / (length(result.P_el) + length(result.P_th_c) + length(result.T_M))
    errors["MAE_smooth"] =
        (
            errors["MAE_smooth_P_el"] * length(result.P_el) +
            errors["MAE_P_th_c"] * length(result.P_th_c) +
            errors["MAE_T_M"] * length(result.T_M)
        ) / (length(result.P_el) + length(result.P_th_c) + length(result.T_M))

    x_ticks = range(ceil(values[1, :Timestamp], Dates.Hour), floor(values[end, :Timestamp], Dates.Hour), step=Hour(2))
    x_tickformat = Dates.format.(x_ticks, "dd. u HH:MM")

    plt_power = plot(
        values[:, :Timestamp],
        values[:, names.P_el],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Electric power in W",
        label="actual",
        palette=:okabe_ito,
    )
    plot!(plt_power, values[:, :Timestamp], result.P_el, ylabel="Power in W", label="predicted")

    plt_thermal = plot(
        values[:, :Timestamp],
        values[:, names.P_th],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Thermal power in W",
        label="actual",
        palette=:okabe_ito,
        legend=false,
    )
    plot!(
        plt_thermal,
        values[:, :Timestamp],
        result.P_th_c,
        label="predicted",
        legend=false,
    )
    # plot!(plt_thermal, values[:, :Timestamp], result.P_th_o, label="predicted power-loss", color=palette(:okabe_ito)[3])

    plt_act = plot(
        ylabel="Energy state",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0.4, 3.6),
        yticks=([1, 2, 3], [:standby, :operational, :working]),
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_st],
        Dict(true => 1),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[1],
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_op],
        Dict(true => 2),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[2],
    )
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_wk],
        Dict(true => 3),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[3],
    )

    ylimzproc = maximum(skipmissing(values[!, names.z_proc])) + 0.1 * (maximum(skipmissing(values[!, names.z_proc])))
    plt_timevolume = plot(
        values[:, :Timestamp],
        values[:, names.z_proc],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Mat. removal rate in g/s",
        label="material removal rate",
        ylims=(0, ylimzproc),
        legend=false,
    )

    plt_temperature = plot(
        values[:, :Timestamp],
        values[:, names.T_M],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        label="machine",
        ylabel="Temperature in °C",
        xlabel="Time",
        palette=:okabe_ito,
        ylims=(0, Inf),
    )
    plot!(plt_temperature, values[:, :Timestamp], result.T_M, label="predicted machine")
    plot!(plt_temperature, values[:, :Timestamp], values[:, names.T_c], label="coolant")
    plot!(plt_temperature, values[:, :Timestamp], values[:, names.T_u], label="air")

    plt = plot(plt_power, plt_thermal, plt_act, plt_timevolume, plt_temperature, layout=l; kwargs...)

    return plt, errors
end
