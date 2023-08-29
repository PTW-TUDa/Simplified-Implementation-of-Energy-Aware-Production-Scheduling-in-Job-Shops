import Plots: gr, plot, savefig, RGBA
import Impute: interp, locf

using CPLEX
using DataFrames
using JSON
using JuMP
using Logging
using Statistics
using Dates

using ProductionScheduling

function main()
    # Path for training data
    data_path = "<local_path>"
    # Export path for the parametrized model
    model_export_path = "trials/test/results/data/machines.json"
    # Export path for plots
    plot_path = "trials/test/results/plots/"
    plot_suffix = "png"

    # Additional arguments for plotters
    plots_kwargs = (
        yformatter=:plain,
        fontfamily="Palatino Roman",
        legend_background_color=RGBA(1, 1, 1, 0.7),
        margin=(1, :mm),
        dpi=150,
        size=(
            321.51616 * 120 * 0.01384, # width in pt * dpi * inches per pt
            469.4704 * 120 * 0.01384 * 0.95, # heigth in pt * dpi * inches per pt * 0.95
        ),
    )

    # Weights of parts after each processing step
    weights = Dict(
        :raw => [456],
        :op10 => [345, 346], # After first operation
        :op11 => [234, 235], # After second operation
    )
    durations = Dict(:op10 => 123, :op11 => 456,) # Duration per operation in s

    # Use gr plotting backend
    gr()

    plot_modeldata(
        weights,
        durations,
        NamedTuple(),
        plot_path,
        "<subfolder_name>",
        plots_kwargs,
        plot_suffix,
        data_path,
        "<file1>",
        "<file2>",
    )

    train_models(
        weights,
        durations,
        (
            machine=ts ->
                DateTime("13.12.2022 10:10", dateformat"d.m.y H:M") .<=
                ts .<=
                DateTime("13.12.2022 18:00", dateformat"d.m.y H:M") .||
                DateTime("14.12.2022 10:00", dateformat"d.m.y H:M") .<=
                ts .<=
                DateTime("14.12.2022 18:00", dateformat"d.m.y H:M"),
        ),
        plot_path,
        plots_kwargs,
        plot_suffix,
        model_export_path,
        data_path,
        "<file1>",
    )

    test_models(
        weights,
        durations,
        NamedTuple(),
        plot_path,
        plots_kwargs,
        plot_suffix,
        model_export_path,
        data_path,
        "<file2>",
    )
end

"""
Import data and create plots with it.
"""
function plot_modeldata(
    weights,
    durations,
    periods,
    plot_path,
    plot_prefix,
    plots_kwargs,
    plot_suffix,
    data_root_path,
    data_files...,
)
    # Import Data from experiments
    data = import_experimentdata(data_root_path, data_files...)

    models, data = create_models!(data, weights, durations, periods)

    # Plot measured data
    map(
        m -> savefig(
            plot_data(m, data; plots_kwargs...),
            joinpath(plot_path, "$(plot_prefix)_$(m.id)_$(m.name).$plot_suffix"),
        ),
        values(models),
    )
end

"""
Estimate machine energy model parameters with provided data.
"""  
function train_models(
    weights,
    durations,
    periods,
    plot_path,
    plots_kwargs,
    plot_suffix,
    model_export_file,
    data_root_path,
    data_files...,
)
    # Import Data from experiments
    data = import_experimentdata(data_root_path, data_files...)
    rename_data!(data)

    models, data = create_models!(data, weights, durations, periods)

    # Plot measured data
    map(
        m -> savefig(
            plot_data(m, data; plots_kwargs...),
            joinpath(plot_path, "training_$(m.id)_$(m.name).$plot_suffix"),
        ),
        values(models),
    )

    # Create and optimize regression models 
    regression_models = Dict(map(m -> m.id => regression_model(m, data), values(models)))

    optimizer = prepare_optimization()
    optimizemodels!(regression_models, optimizer; display=true) # , outfile="model.txt")

    # Export model data as resource file for scheduling
    open(model_export_file, "w") do f
        JSON.print(f, collect(map(m -> export_parameters(m, regression_models[m.id], data), values(models))), 4)
    end
end

"""
Test model fit compared to provided test data.
"""
function test_models(
    weights,
    durations,
    periods,
    plot_path,
    plots_kwargs,
    plot_suffix,
    model_import_file,
    data_root_path,
    data_files...,
)
    # Import Data from experiments
    data = import_experimentdata(data_root_path, data_files...)
    rename_data!(data)
    # describe_data(data)

    models, data = create_models!(data, weights, durations, periods)
    parametrized_models = map(m -> convert_model(m, model_import_file), values(models))
    results = forward_models(parametrized_models, models, data)

    # Plot measured data and print errors measures.
    for m in values(models)
        plt, errors = plot_result(m, data, results[m.id]; plots_kwargs...)
        savefig(plt, joinpath(plot_path, "test_$(m.id)_$(m.name).$plot_suffix"))
        for (name, value) in pairs(errors)
            println("$(m.id)_$(m.name) $name: $value")
        end
    end
end

"""
Create models that can be parametrized
"""
function create_models!(data, weights, durations, periods)
    # Create Machine models
    m1, data = model_machine1!(data, name="Machine1", id=1, weights=weights, durations=durations, periods=periods)
    models = Dict(map(m -> m.id => m, [m1]))

    return models, data
end

"""
Create models from stored parameters (for testing).
"""
convert_model(model::MachineTool, filename) = MachineTool(model.id, filename)

"""
Create models from stored parameters (for testing).
"""
convert_model(model::CleaningMachine, filename) = CleaningMachine(model.id, filename)

"""
Execute predictions with parametrized models.
"""
function forward_models(parametrized_models::Vector{Machine}, regression_models::Dict{Int, Machine}, data::DataFrame)
    results = Dict{Int, ModelResults}()
    for model in parametrized_models
        names = regression_models[model.id].data
        values = dropmissing(
            isnothing(names.periods) ? data : subset(data, :Timestamp => names.periods),
            union(regression_colnames(names), (:Timestamp,)),
        )

        if typeof(model) <: MachineTool
            results[model.id] = forward_model(
                model,
                a_st=values[:, names.a_st],
                a_op=values[:, names.a_op],
                a_wk=values[:, names.a_wk],
                z_proc=values[:, names.z_proc],
                T_u=values[:, names.T_u],
                T_M1=20.0,
            )
        elseif typeof(model) <: CleaningMachine
            results[model.id] = forward_model(
                model,
                a_st=values[:, names.a_st],
                a_op=values[:, names.a_op],
                a_wk=values[:, names.a_wk],
                z_proc=values[:, names.z_proc],
                T_u=values[:, names.T_u],
                T_M1=35.0,
            )
        else
            error("Unknown model type: $typeof(model).")
        end
    end
    return results
end

"""
Preprocess data and create a model for parameter estimation.
"""
function model_machine1!(data; name, id, weights, durations, periods)
    rates = Dict(
        "OP10" => (mean(weights[:raw]) - mean(weights[:op10])) / durations[:op10],
        "OP20" => (mean(weights[:op10]) - mean(weights[:op11])) / durations[:op11],
    )
    data[:, :mat_removal_rate] = locf(encode_str2num(data[:, :program_id], rates))

    return MachineTool(
        name,
        id=id,
        P_el=:power_elec,
        P_th=:power_thermal,
        a_st=:mode_standby,
        a_op=:mode_operational,
        a_wk=:mode_working,
        z_proc=:mat_removal_rate,
        T_u=:ambient_temperature,
        T_c=:coolant_temperature,
        T_M=:machine_temperature,
        periods=periods,
    ),
    data
end

"""
Prepare the model parameter estimation optimization.
"""
function prepare_optimization()
    global_logger(ConsoleLogger(stdout, Logging.Info; show_limited=true))

    # CPLEX Optimizer 
    optimizer_with_attributes(CPLEX.Optimizer)
end

main()
