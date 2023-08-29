using JuMP
using JSON
using Plots

"""
Data used for the parameter estimation for thermal cleaning machine models.
"""
struct ThermalCleaningMachineRegressionData <: RegressionData
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
    "Name of the heater active indicator (1 or 0) timeseries."
    a_heater::Symbol
    "Name of the machine temperature timeseries."
    T_M::Symbol
    "Name of the external temperature timeseries."
    T_u::Symbol
    "Name of the timeseries describing the number of workpieces processed concurrently."
    z_proc::Symbol
    "Regression Periods"
    periods::Union{Nothing, Function}
end

"""
Data used for the parameter estimation for thermal cleaning machine models.
"""

struct ElectricCleaningMachineRegressionData <: RegressionData
    "Name of the electric power consumption timeseries."
    P_el::Symbol
    "Name of the standby mode active indicator (1 or 0) timeseries."
    a_st::Symbol
    "Name of the operational mode active indicator (1 or 0) timeseries."
    a_op::Symbol
    "Name of the working mode active indicator (1 or 0) timeseries."
    a_wk::Symbol
    "Name of the heater active indicator (1 or 0) timeseries."
    a_heater::Symbol
    "Name of the machine temperature timeseries."
    T_M::Symbol
    "Name of the external temperature timeseries."
    T_u::Symbol
    "Name of the timeseries describing the number of workpieces processed concurrently."
    z_proc::Symbol
    "Regression Periods"
    periods::Union{Nothing, Function}
end

"""
Parameter data for cleaning machine models, available after parameter estimation.
"""
struct CleaningMachineParameterData <: ParameterData
    "Flag to indicate whether the machine uses an electric heater."
    electric::Bool
    "Regression parameter for electric energy."
    β_el::Vector{Float64}
    "Regression parameter for thermal energy."
    β_th_h::Float64
    "Regression parameter for thermal energy loss to the environment."
    β_th_o::Float64
    "Regression parameter for thermal energy loss to workpieces."
    β_th_wp::Float64
    "Regression parameter for thermal energy loss due to spray cleaning."
    β_th_s::Float64
    "Regression parameter for heat capacity of the machine."
    β_c_M::Float64
end

"""
Object describing a cleaning machine in the factory.
"""
struct CleaningMachine{T} <: Machine where {T <: ModelData}
    "Name of the machine."
    name::String
    "Unique identifier of the machine."
    id::Int
    "Capacity of the machine (how many parts it can produce at once)."
    capacity::Int
    "Unique job flag (true if all coinciding jobs have to be the same operation.)"
    unique_job::Bool
    "Lower temperature limit for the cleaning medium tank."
    t_lower::Float64
    "Upper temperature limit for the cleaning medium tank."
    t_upper::Float64
    "Either parameter or regression data object."
    data::T
end

"""
Instantiate ThermalCleaningMachine object with the names timeseries names for regression.
"""
ThermalCleaningMachine(
    name;
    id,
    capacity,
    t_lower,
    t_upper,
    P_el,
    P_th,
    a_st,
    a_op,
    a_wk,
    a_heater,
    T_M,
    T_u,
    z_proc,
    periods,
) = CleaningMachine(
    name,
    id,
    capacity,
    false,
    t_lower,
    t_upper,
    ThermalCleaningMachineRegressionData(P_el, P_th, a_st, a_op, a_wk, a_heater, T_M, T_u, z_proc, periods),
)

"""
Instantiate ElectricCleaningMachine object with the names timeseries names for regression.
"""
ElectricCleaningMachine(
    name;
    id,
    capacity,
    t_lower,
    t_upper,
    P_el,
    a_st,
    a_op,
    a_wk,
    a_heater,
    T_M,
    T_u,
    z_proc,
    periods,
) = CleaningMachine(
    name,
    id,
    capacity,
    false,
    t_lower,
    t_upper,
    ElectricCleaningMachineRegressionData(P_el, a_st, a_op, a_wk, a_heater, T_M, T_u, z_proc, periods),
)

"""
Instantiate CleaningMachine object with parameter data from a file.
"""
function CleaningMachine(id::Int, filename::AbstractString)
    objects = JSON.parsefile(filename)
    for obj in objects
        if obj["resource_id"] == id
            return CleaningMachine(
                obj["name"],
                obj["resource_id"],
                obj["capacity"],
                obj["unique_job"],
                obj["t_lower"],
                obj["t_upper"],
                CleaningMachineParameterData(
                    obj["electric"],
                    obj["parameters"]["beta_el"],
                    obj["parameters"]["beta_th_h"],
                    obj["parameters"]["beta_th_o"],
                    obj["parameters"]["beta_th_wp"],
                    obj["parameters"]["beta_th_s"],
                    obj["parameters"]["beta_c_M"],
                ),
            )
        end
    end
    return nothing
end

"""
Create a regression model for the machine.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the CleaningMachine object.
:return: JuMP model.
"""
function regression_model(machine::CleaningMachine{T}, data::DataFrame) where {T <: RegressionData}
    model = Model()
    names = machine.data
    electric = typeof(names) == ElectricCleaningMachineRegressionData

    values = dropmissing(
        isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods),
        regression_colnames(names),
    )
    timesteps = size(values)[1]

    # Parameters
    a_st = Vector{Bool}(values[:, names.a_st])
    a_op = Vector{Bool}(values[:, names.a_op])
    a_wk = Vector{Bool}(values[:, names.a_wk])
    a_h = Vector{Bool}(values[:, names.a_heater])
    z_proc = Vector{Float64}(values[:, names.z_proc])
    P_el = Vector{Float64}(values[:, names.P_el])
    T_u = Vector{Float64}(values[:, names.T_u])
    T_M = Vector{Float64}(values[:, names.T_M])

    # Variables for electric power consumption
    @variables(model, begin
        β_el[1:3]
        ϵ_el[1:timesteps]
        β_th_h
        β_th_o
        β_th_wp
        β_th_s
        β_c_M
        ϵ_th_h[1:timesteps]
        ϵ_th_m[1:timesteps]
    end)

    @constraints(model, begin
        β_el[1:3] .>= 0
        β_th_h >= 0
        β_th_o >= 0
        β_th_wp >= 0
        β_th_s >= 0
        β_c_M >= 0
    end)

    if electric
        @variable(model, P_th_h[1:timesteps])
        @constraints(
            model,
            begin
                c_th_h[t=1:timesteps], P_th_h[t] == a_h[t] * β_th_h + ϵ_th_h[t]
                # c_T_M[t=1:timesteps-1],
                # β_c_M * T_M[t+1] == a_h[t] * β_th_h - a_wk[t] * (z_proc[t] * β_th_wp) * (T_M[t] - T_u[t]) + ϵ_th_m[t]
                c_T_M[t=1:timesteps-1],
                β_c_M * T_M[t+1] ==
                a_h[t] * β_th_h + β_c_M * T_M[t] - β_th_o * (T_M[t] - T_u[t]) -
                a_wk[t] * (β_th_s + z_proc[t] * β_th_wp) * (T_M[t] - T_u[t]) + ϵ_th_m[t]
                c_el[t=1:timesteps],
                P_el[t] == (
                    (a_st[t] + a_op[t] + a_wk[t]) * β_el[1] +
                    (a_op[t] + a_wk[t]) * β_el[2] +
                    a_wk[t] * β_el[3] +
                    ϵ_el[t] +
                    P_th_h[t]
                )
            end
        )
    else
        P_th_h = Vector{Float64}(values[:, names.P_th])
        @constraints(
            model,
            begin
                c_th_h[t=1:timesteps], P_th_h[t] == a_h[t] * β_th_h + ϵ_th_h[t]
                # c_T_M[t=1:timesteps-1],
                # β_c_M * T_M[t+1] == a_h[t] * β_th_h - a_wk[t] * (z_proc[t] * β_th_wp) * (T_M[t] - T_u[t]) + ϵ_th_m[t]
                c_T_M[t=1:timesteps-1],
                β_c_M * T_M[t+1] ==
                a_h[t] * β_th_h + β_c_M * T_M[t] - β_th_o * (T_M[t] - T_u[t]) -
                a_wk[t] * (β_th_s + z_proc[t] * β_th_wp) * (T_M[t] - T_u[t]) + ϵ_th_m[t]
                c_el[t=1:timesteps],
                P_el[t] ==
                ((a_st[t] + a_op[t] + a_wk[t]) * β_el[1] + (a_op[t] + a_wk[t]) * β_el[2] + a_wk[t] * β_el[3] + ϵ_el[t])
            end
        )
    end

    # Objective for the regression model is to reduce the quadratic loss function.
    @objective(model, Min, sum(ϵ_el[t]^2 + ϵ_th_h[t]^2 + ϵ_th_m[t]^2 for t in 1:timesteps))
    return model
end

"""
Plot data collected to perform the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the CleaningMachine object.
:return: A plot object.
"""
function plot_data(machine::CleaningMachine{T}, data::DataFrame; kwargs...) where {T <: RegressionData}
    names = machine.data

    values =
        coalesce.(
            (isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods))[
                !,
                union(regression_colnames(names), (:Timestamp,)),
            ],
            NaN64,
        )

    x_ticks = range(ceil(values[1, :Timestamp], Dates.Hour), floor(data[end, :Timestamp], Dates.Hour), step=Hour(2))
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

    if typeof(names) == ThermalCleaningMachineRegressionData
        plt_thermal = plot(
            values[:, :Timestamp],
            values[:, names.P_th],
            xticks=(x_ticks, x_tickformat),
            xlims=(-Inf, Inf),
            ylabel="Thermal power in W",
            label="thermal power",
            color=palette(:okabe_ito)[2],
            legend=false,
        )
    end

    plt_act = plot(
        ylabel="Energy state",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0.4, 4.6),
        yticks=([1, 2, 3, 4], [:standby, :operational, :working, :heater]),
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
    barmap!(
        plt_act,
        values[:, :Timestamp],
        values[:, names.a_heater],
        Dict(true => 4),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[4],
    )

    ylimzproc = maximum(skipmissing(values[!, names.z_proc])) + 0.1 * maximum(skipmissing(values[!, names.z_proc]))
    plt_workpieces = plot(
        values[:, :Timestamp],
        values[:, names.z_proc],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Number of workpieces",
        label="number of workpieces",
        ylims=(0, ylimzproc),
        legend=false,
    )

    plt_temperature = plot(
        values[:, :Timestamp],
        values[:, names.T_M],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        label="cleaning medium",
        ylabel="Temperature in °C",
        xlabel="Time",
        palette=:okabe_ito,
        ylims=(0, Inf),
    )
    plot!(plt_temperature, values[:, :Timestamp], values[:, names.T_u], label="air")

    if typeof(names) == ThermalCleaningMachineRegressionData
        plt = plot(
            plt_power,
            plt_thermal,
            plt_act,
            plt_workpieces,
            plt_temperature,
            layout=@layout([°; °; °; °; °]);
            kwargs...,
        )
    else
        plt = plot(plt_power, plt_act, plt_workpieces, plt_temperature, layout=@layout([°; °; °; °]); kwargs...)
    end

    return plt
end

"""
Export the parameters identified during the regression.

:param machine: The machine object containing the timeseries names (RegressionData).
:param model: The solved JuMP model containing the estimated parameters.
:return: Dictionary to be exported to a JSON file.
"""
function export_parameters(
    machine::CleaningMachine{T},
    model::JuMP.AbstractModel,
    data::DataFrame,
) where {T <: RegressionData}
    vars = object_dictionary(model)
    Dict(
        "type" => "cleaningmachine",
        "name" => machine.name,
        "resource_id" => machine.id,
        "capacity" => machine.capacity,
        "unique_job" => false,
        "t_lower" => machine.t_lower,
        "t_upper" => machine.t_upper,
        "electric" => typeof(machine.data) == ElectricCleaningMachineRegressionData,
        "parameters" => Dict(
            "beta_el" => value.(vars[:β_el]),
            "beta_th_h" => value.(vars[:β_th_h]),
            "beta_th_o" => value.(vars[:β_th_o]),
            "beta_th_wp" => value.(vars[:β_th_wp]),
            "beta_th_s" => value.(vars[:β_th_s]),
            "beta_c_M" => value.(vars[:β_c_M]),
        ),
    )
end

"""
Results vectors from a forward execution of the cleaning machine model.
"""
struct CleaningMachineResults <: ModelResults
    "Flag to indicate whether the machine uses an electric heater."
    electric::Bool
    "Predicted electric power consumption."
    P_el::Vector{Union{Missing, Float64}}
    "Predicted thermal power consumption."
    P_th::Vector{Union{Missing, Float64}}
    "Predicted machine temperature"
    T_M::Vector{Union{Missing, Float64}}
    "Predicted operation of tank heating system"
    a_heater::Vector{Union{Missing, Bool}}
end

"""
Prediction model for power consumption of the cleaning machine

:param machine: The machine object containing the estimated parameters.
:param a_st: Standby mode active indicator (1 or 0) timeseries.
:param a_op: Operational mode active indicator (1 or 0) timeseries.
:param a_wk: Working mode active indicator (1 or 0) timeseries.
:param z_proc: Timeseries describing the number of workpieces processed at once.
:param T_u: External temperature timeseries.
:param T_M1: Machine temperature at the beginning.
"""
function forward_model(
    machine::CleaningMachine{T};
    a_st,
    a_op,
    a_wk,
    z_proc,
    T_u,
    T_M1::Number,
) where {T <: ParameterData}
    β_el = machine.data.β_el
    β_th_h = machine.data.β_th_h
    β_c_M = machine.data.β_c_M
    β_th_o = machine.data.β_th_o
    β_th_s = machine.data.β_th_s
    β_th_wp = machine.data.β_th_wp

    T_M = Vector{Union{Missing, Float64}}(missing, length(a_st))
    P_th_h = Vector{Union{Missing, Float64}}(missing, length(a_st))
    a_h = Vector{Union{Missing, Bool}}(missing, length(a_st))

    T_M[1] = T_M1
    a_h[1] = false
    for t in 1:length(a_st)-1
        P_th_h[t] = a_h[t] * β_th_h

        # 2-point control for the tank heater
        if machine.t_lower <= T_M[t] <= machine.t_upper
            a_h[t+1] = a_h[t]
        elseif T_M[t] < machine.t_lower
            a_h[t+1] = true
        else
            a_h[t+1] = false
        end
        # T_M[t+1] = a_h[t] * (β_th_h / β_c_M) - a_wk[t] * ((z_proc[t] * β_th_wp) / β_c_M) * (T_M[t] - T_u[t])
        T_M[t+1] =
            a_h[t] * (β_th_h / β_c_M) + T_M[t] - (β_th_o / β_c_M) * (T_M[t] - T_u[t]) -
            a_wk[t] * ((β_th_s + z_proc[t] * β_th_wp) / β_c_M) * (T_M[t] - T_u[t])
    end

    P_th_h[end] = a_h[end] * β_th_h

    if machine.data.electric
        P_el = (a_st + a_op + a_wk) .* β_el[1] + (a_op + a_wk) .* β_el[2] + a_wk .* β_el[3] + P_th_h
    else
        P_el = (a_st + a_op + a_wk) .* β_el[1] + (a_op + a_wk) .* β_el[2] + a_wk .* β_el[3]
    end

    return CleaningMachineResults(machine.data.electric, P_el, P_th_h, T_M, a_h)
end

"""
Results plot for a prediction made using the machine energy model.

:param machine: The machine object containing the timeseries names (RegressionData).
:param data: Actual timeseries identified by the names in the MachineTool object.
:param results: Results vectors from a forward execution of the machine tool model.
"""
function plot_result(
    machine::CleaningMachine{T},
    data::DataFrame,
    result::CleaningMachineResults;
    kwargs...,
) where {T <: RegressionData}
    names = machine.data
    values = dropmissing(
        isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods),
        union(regression_colnames(names), (:Timestamp,)),
    )

    errors = Dict{String, Float64}(
        "RMSE_P_el" => sqrt(sum((result.P_el .- values[:, names.P_el]) .^ 2) / length(result.P_el)),
        "RMSE_smooth_P_el" => sqrt(
            sum(skipmissing(result.P_el .- moving_average(values[:, names.P_el], 30)) .^ 2) / length(result.P_el),
        ),
        "RMSE_T_M" => sqrt(sum((result.T_M .- values[:, names.T_M]) .^ 2) / length(result.T_M)),
        "MAE_P_el" => sum(abs.(result.P_el .- values[:, names.P_el])) / length(result.P_el),
        "MAE_T_M" => sum(abs.(result.T_M .- values[:, names.T_M])) / length(result.T_M),
        "MAE_smooth_P_el" =>
            sum(abs.(skipmissing(result.P_el .- moving_average(values[:, names.P_el], 30)))) / length(result.P_el),
        "TotalEnergy_P_el" => (sum(result.P_el) - sum(values[:, names.P_el])) / sum(values[:, names.P_el]),
    )
    if typeof(names) == ThermalCleaningMachineRegressionData
        errors["RMSE_P_th"] = sqrt(sum((result.P_th .- values[:, names.P_th]) .^ 2) / length(result.P_th))
        errors["RMSE_smooth_P_th"] =
            sqrt(sum(skipmissing(result.P_th .- moving_average(values[:, names.P_th], 30)) .^ 2) / length(result.P_th))

        errors["MAE_P_th"] = sum(abs.(result.P_th .- values[:, names.P_th])) / length(result.P_th)
        errors["MAE_smooth_P_th"] =
            sum(abs.(skipmissing(result.P_th .- moving_average(values[:, names.P_th], 30)))) / length(result.P_th)
        errors["TotalEnergy_P_th"] = (sum(result.P_th) - sum(values[:, names.P_th])) / sum(values[:, names.P_th])

        errors["RMSE"] =
            (
                errors["RMSE_P_el"] * length(result.P_el) +
                errors["RMSE_P_th"] * length(result.P_th) +
                errors["RMSE_T_M"] * length(result.T_M)
            ) / (length(result.P_el) + length(result.P_th) + length(result.T_M))
        errors["MAE"] =
            (
                errors["MAE_P_el"] * length(result.P_el) +
                errors["MAE_P_th"] * length(result.P_th) +
                errors["MAE_T_M"] * length(result.T_M)
            ) / (length(result.P_el) + length(result.P_th) + length(result.T_M))
        errors["RMSE_smooth"] =
            (
                errors["RMSE_smooth_P_el"] * length(result.P_el) +
                errors["RMSE_P_th"] * length(result.P_th) +
                errors["RMSE_T_M"] * length(result.T_M)
            ) / (length(result.P_el) + length(result.P_th) + length(result.T_M))
        errors["MAE_smooth"] =
            (
                errors["MAE_smooth_P_el"] * length(result.P_el) +
                errors["MAE_P_th"] * length(result.P_th) +
                errors["MAE_T_M"] * length(result.T_M)
            ) / (length(result.P_el) + length(result.P_th) + length(result.T_M))
    else
        errors["RMSE"] =
            (errors["RMSE_P_el"] * length(result.P_el) + errors["RMSE_T_M"] * length(result.T_M)) /
            (length(result.P_el) + length(result.T_M))
        errors["MAE"] =
            (errors["MAE_P_el"] * length(result.P_el) + errors["MAE_T_M"] * length(result.T_M)) /
            (length(result.P_el) + length(result.T_M))
        errors["RMSE_smooth"] =
            (errors["RMSE_smooth_P_el"] * length(result.P_el) + errors["RMSE_T_M"] * length(result.T_M)) /
            (length(result.P_el) + length(result.T_M))
        errors["MAE_smooth"] =
            (errors["MAE_smooth_P_el"] * length(result.P_el) + errors["MAE_T_M"] * length(result.T_M)) /
            (length(result.P_el) + length(result.T_M))
    end

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
    plot!(plt_power, values[:, :Timestamp], result.P_el, label="predicted")

    if typeof(names) == ThermalCleaningMachineRegressionData
        plt_thermal = plot(
            values[:, :Timestamp],
            values[:, names.P_th],
            xticks=(x_ticks, x_tickformat),
            xlims=(-Inf, Inf),
            ylabel="Thermal power in W",
            label="actual",
            legend=false,
            palette=:okabe_ito,
        )
        plot!(
            plt_thermal,
            values[:, :Timestamp],
            result.P_th,
            label="predicted",
            legend=false,
        )
    end

    plt_act = plot(
        ylabel="Energy state",
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylims=(0.4, 4.6),
        yticks=([1, 2, 3, 4], [:standby, :operational, :working, :heater]),
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
    barmap!(
        plt_act,
        values[:, :Timestamp],
        result.a_heater,
        Dict(true => 4),
        orientation=:h,
        bar_width=0.2,
        color=palette(:okabe_ito)[4],
    )

    ylimzproc = maximum(skipmissing(values[!, names.z_proc])) + 0.1 * maximum(skipmissing(values[!, names.z_proc]))
    plt_workpieces = plot(
        values[:, :Timestamp],
        values[:, names.z_proc],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        ylabel="Number of workpieces",
        label="number of workpieces",
        palette=:okabe_ito,
        ylims=(0, ylimzproc),
        legend=false,
    )

    plt_temperature = plot(
        values[:, :Timestamp],
        values[:, names.T_M],
        xticks=(x_ticks, x_tickformat),
        xlims=(-Inf, Inf),
        label="cleaning medium",
        ylabel="Temperature in °C",
        xlabel="Time",
        palette=:okabe_ito,
        ylims=(0, Inf),
    )
    plot!(plt_temperature, values[:, :Timestamp], result.T_M, label="predicted cleaning medium")
    plot!(plt_temperature, values[:, :Timestamp], values[:, names.T_u], label="air")

    if typeof(names) == ThermalCleaningMachineRegressionData
        plt = plot(
            plt_power,
            plt_thermal,
            plt_act,
            plt_workpieces,
            plt_temperature,
            layout=@layout([°; °; °; °; °]);
            kwargs...,
        )
    else
        plt = plot(plt_power, plt_act, plt_workpieces, plt_temperature, layout=@layout([°; °; °; °]); kwargs...)
    end

    return plt, errors
end
