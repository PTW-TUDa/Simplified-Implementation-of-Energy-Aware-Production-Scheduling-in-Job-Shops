import Dates
import Random: Xoshiro, seed!
import Impute: declaremissings
using PyCall

"Settings for an eta_utility BaseEnv environment"
struct EnvironmentSettings
    "Pyenv is the original python environment object managing the interaction."
    pyenv::PyObject

    "Path where optimization results should be stored."
    pathresults::String
    "Path where imported scenario data can be found."
    pathscenarios::String
    "Callback for the step function."
    callback::Function

    "ID of this environment in case multiple environments are being used."
    envid::Int
    "Name of the optimization run, for storing optimization results."
    runname::String

    "Duration of an episode (used to determine completion of the episode)."
    episodeduration::Float64
    "Interval between taking sample of scenario data."
    samplingtime::Float64
    "Number of steps in an episode."
    nepisodesteps::Int

    "Duration to read from scenario data."
    scenarioduration::Float64
    "Beginning time of a scenario."
    scenariotimebegin::Dates.DateTime
    "Ending time of a scenario."
    scenariotimeend::Dates.DateTime
    
    "Seed used for the random generator."
    seed::UInt64
    "The random generator itself."
    rng::Xoshiro

    function EnvironmentSettings(pyenv::PyObject)
        pyenv.is_multithreaded = true

        # Initialize and seed the random number generator
        rng = Xoshiro()
        seed!(rng, convert(UInt64, pyenv._seed))

        new(
            pyenv,
            pycall(pyenv.path_results.as_posix, String),
            pycall(pyenv.path_scenarios.as_posix, String),
            pyenv.callback,
            pyenv.env_id,
            pyenv.run_name,
            pyenv.episode_duration,
            pyenv.sampling_time,
            pyenv.n_episode_steps,
            pyenv.scenario_duration,
            pyenv.scenario_time_begin,
            pyenv.scenario_time_end,
            pyenv._seed,
            rng
        )
    end

    function EnvironmentSettings(
        pathresults::String,
        pathscenarios::String,
        envid::Int,
        runname::String,
        episodeduration::Number,
        samplingtime::Number,
        scenariotimebegin::Dates.DateTime,
        scenariotimeend::Dates.DateTime,
        seed::Int64,
    )
        rng = Xoshiro()
        seed!(rng, seed)
        new(
            nothing,
            pathresults,
            pathscenarios,
            x -> x,
            envid,
            runname,
            episodeduration,
            samplingtime,
            episodeduration / samplingtime,
            (scenariotimeend - scenariotimebegin).value / 1000,
            scenariotimebegin,
            scenariotimeend,
            seed,
            rng,
        )
    end
end

"Status of the optimization using the Environment."
mutable struct EnvironmentStatus
    "Number of completed episodes."
    nepisodes::Int
    "Number of completed steps in the current episode."
    nsteps::Int
    "Number of completed steps over all episodes."
    nstepslongtime::Int
    
    function EnvironmentStatus(pyenv::PyObject)
        new(
            pyenv.n_episodes,
            pyenv.n_steps,
            pyenv.n_steps_longtime
        )
    end

    function EnvironmentStatus()
        new(0, 0, 0)
    end
end

"""
Create the EnvironmentSettings and EnvironmentStatus objects from a python environment object.
""" 
pyobject_to_environment(pyenv::PyObject) = EnvironmentSettings(pyenv), EnvironmentStatus(pyenv)

"""
Create a spaces.Box space for gym envrionments.

:param lows: Low value for all values in the box.
:param highs: High value for all values in the box.
:param shape: Shape of the box (vector of dimensions).
:param dtype: Datatype (has to be a python type).
""" 
py_box_space(lows, highs; shape, dtype) = pycall(spaces.Box, PyObject, lows, highs; shape=shape, dtype=dtype)

"""
Create a spaces.Discrete space for gym environments.

:param length: Length of the space
"""
py_discrete_space(length) = pycall(spaces.Discrete, PyObject, length)

"""
Create a spaces.MultiDiscrete space for gym environments.

:param lengths: Vector of lengths for all elements in the space.
"""
py_multidiscrete_space(lengths) = pycall(spaces.MultiDiscrete, PyObject, lengths)

"""
Create a spaces.Dict space for gym environments.

:param dict: Dictionary mapping values to spaces.
"""
py_dict_space(dict) = pycall(spaces.Dict, PyObject, dict)

"""
Set the observation_space and action_space of the gym environment.

:param envset: EnvironmentSettings object containing the python environment.
:param action_space: The Space object to be used as the action space.
:param observation_space: The Space object to be used as the observation space.
"""
function pyenv_set_spaces!(envset::EnvironmentSettings, action_space::PyObject, observation_space::PyObject)
    envset.pyenv."observation_space" = observation_space
    envset.pyenv."action_space" = action_space
end

"""
Import scenario data from .csv files using eta_utility functionality.

:param envset: EnvironmentSettings object containing settings information.
:param scenario_paths: Path from the scenario files are to be loaded.
:param prefix_renamed: Set to true to rename columns which have been prefixed.
"""
function py_import_scenario(envset::EnvironmentSettings, scenario_paths; prefix_renamed = false)
    paths = String[]
    prefix = Union{String, Nothing}[]
    int_methods = Union{String, Nothing}[]
    scale_factors = Union{Dict{String, Float64}, Nothing}[]
    rename_cols = Dict{String, String}()
    infer_datetime_from = Union{String, Int}[]
    time_conversion_str = String[]

    for path in scenario_paths
        push!(paths, joinpath(envset.pathscenarios, path["path"]))
        push!(prefix, get(path, "prefix", nothing))
        push!(int_methods, get(path, "interpolation_method", nothing))
        push!(scale_factors, get(path, "scale_factors", nothing))
        merge!(rename_cols, get(path, "rename_cols", Dict()))
        push!(infer_datetime_from, get(path, "infer_datetime_cols", "string"))
        push!(time_conversion_str, get(path, "time_conversion_str", "%Y-%m-%d %H:%M"))
    end

    timeseries = pyimport("eta_utility.timeseries")
    scenario = timeseries.scenario_from_csv(
        paths=paths,
        resample_time = envset.samplingtime,
        start_time = envset.scenariotimebegin,
        end_time = envset.scenariotimeend,
        total_time = envset.scenarioduration,
        random=haskey(envset.pyenv, "np_random") ? envset.pyenv."np_random" : nothing,
        interpolation_method = int_methods,
        scaling_factors = scale_factors,
        rename_cols = rename_cols,
        prefix_renamed = prefix_renamed,
        infer_datetime_from = infer_datetime_from,
        time_conversion_str = time_conversion_str
    )

    # Convert the python pandas DataFrame to a julia DataFrame.
    declaremissings(DataFrame(scenario.values, scenario.columns.to_list()); values=(NaN, "NULL", nothing))
end

"""
Reset the environment status.

:param status: EnvironmentStatus object containing the actual status.
:param envset: Settings object containing a reference to the python environment.
"""
function reset!(status::EnvironmentStatus, envset::EnvironmentSettings)
    status.nepisodes = envset.pyenv.n_episodes
    status.nstepslongtime += envset.pyenv.n_steps_longtime
    status.nsteps = 0
end
