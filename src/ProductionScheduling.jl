module ProductionScheduling

import Base.Filesystem: abspath, joinpath, splitdir
import Random: seed!
using PyCall
using JuMP

# Import python utilities for all sub-files
const etautility = PyNULL()
const spaces = PyNULL()
const np = PyNULL()

function __init__()
    copy!(etautility, pyimport("eta_utility"))
    copy!(spaces, pyimport("gym.spaces"))
    copy!(np, pyimport("numpy"))
end

export Byteorder, LittleEndian, BigEndian, SYS_BYTEORDER, °

# Determine system byteorder for modbus data conversion
@enum Byteorder begin
    LittleEndian = 1
    BigEndian = 2
end
const SYS_BYTEORDER::Byteorder = ENDIAN_BOM == 0x04030201 ? LittleEndian : BigEndian

# Set ° as nothing for plot layouts
const ° = nothing

################################################
#                                              #
#     Data Import and Processing Functions     #
#                                              #
################################################

export import_experimentdata, describe_data, decode_uint16_bitarray, encode_str2num, encode_onehot, barmap, barmap!, moving_average
include("DataProcessing/DataImport.jl")
include("DataProcessing/DataPreprocess.jl")
include("DataProcessing/DataPlotting.jl")

################################################
#                                              #
#          Production Machine Models           #
#                                              #
################################################

export Machine, ModelData, RegressionData, ParameterData, ModelResults
export regression_colnames, writemodel, optimizemodels!

export MachineTool, MachineToolResults
export CleaningMachine, ThermalCleaningMachine, ElectricCleaningMachine, CleaningMachineResults
export regression_model, plot_data, export_parameters, forward_model, plot_result

"""
Machine represents any kind of production machine model.
"""
abstract type Machine end

"""
ModelData contains data for the production machine models. This is either RegressionData or ParameterData.
"""
abstract type ModelData end

"""
RegressionData contains data used for the model parameter estimation in the regression process.
"""
abstract type RegressionData <: ModelData end

"""
ParameterData contains the parameters identified during the regression.
"""
abstract type ParameterData <: ModelData end

"""
ModelResults contains the results of executing a machine energy model for a specific production schedule.
"""
abstract type ModelResults end

function regression_colnames(data::T) where {T <: RegressionData}
    fields = Vector{Symbol}()
    for name in fieldnames(typeof(data))
        field = getfield(data, name)
        if typeof(field) <: Symbol
            push!(fields, field)
        end
    end
    return fields
end


include("MachineModels/MachineTool.jl")
include("MachineModels/CleaningMachine.jl")

"""
Write the JuMP Model to a file for analysing its structure.

:param model: The JuMP model
:param filename: Name of the file to store the model.
"""
writemodel(model::JuMP.AbstractModel, filename) =
    open(filename, "w") do f
        print(f, model)
    end

"""
Take some JuMP models and optimize them.

:param models: Dictionary of JuMP models.
:param optimizer: A JuMP optimizer to be used for the model.
:param outfile: Name of a file where the model should be stored for structural analysis.
The default is nothing. Nothing means that no output will be written.
:param display: Display a model summary and an overview of the solution in console.
:param add_bridges: Determine whether the model needs JuMP bridges to be solved correctly by
the specified optimizer.
"""
function optimizemodels!(
    models::AbstractDict{Int, T},
    optimizer;
    outfile::Union{AbstractString, Nothing}=nothing,
    display=false,
    add_bridges=false,
) where {T <: JuMP.AbstractModel}
    for (m_id, model) in models
        set_optimizer(model, optimizer, add_bridges=add_bridges)
        if !isnothing(outfile)
            dir, file = splitdir(outfile)
            writemodel(model, abspath(joinpath(dir, "$(m_id)_$file")))
        end
        display && println("Optimizing model: ", m_id)
        optimize!(model)
        if display
            @show model
            print(solution_summary(model))
            print("\n\n\n\n")
        end
    end
end

################################################
#                                              #
#        Production System Environment         #
#                                              #
################################################

export ProductionEnvironment, EnvironmentSettings, EnvironmentStatus
export pyobject_to_environment,
    py_box_space,
    py_discrete_space,
    py_multidiscrete_space,
    py_dict_space,
    pyenv_set_spaces!,
    py_import_scenario,
    reset!,
    seed!

export ProductionSystem, Product, Job, Operation, Event, EventsMaps, ScheduleItem

export import_resources,
    import_orders,
    import_machine,
    mapevents,
    mapvars,
    step!,
    reset!,
    close!,
    buildschedule,
    machineschedule,
    schedule!,
    schedule_times,
    generate_expert_schedules,
    log_step_errors

"""
ProductionEnvironment is a super class for any Environment class representing a specific production system.
"""
abstract type ProductionEnvironment end

include("Scheduling/SettingsStatus.jl")
include("Scheduling/ProductionSystem.jl")
include("Scheduling/Scheduler.jl")

"""
Seed the random number generator of the environment.

:param env: The environment.
:param seed: An integer used for seeding the RNG.
"""
seed!(env::ProductionEnvironment, seed) = seed!(env.settings.rng, seed)

end
