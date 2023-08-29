using DataFrames

import CSV
import Dates
import Base.Filesystem: normpath, joinpath

"""
Import data from csv files created during experiments.

:param root_path: Root path where all files can be found.
:param files: List of files to import.
"""
function import_experimentdata(root_path::AbstractString, files::AbstractString...)
    data = DataFrame()
    for file in files
        filedata = CSV.read(
            normpath(joinpath(root_path, file)),
            DataFrame;
            delim=',',
            normalizenames=true,
            missingstring=["", "NaN", "NAN", "NA", "nan"],
            decimal='.',
            types=Dict(1 => Dates.DateTime),
            dateformat=Dict(1 => "y-m-d H:M:S.s"),
            strict=true,
            validate=true,
        )
        append!(data, filedata, cols=:union)
    end
    sort!(data, :Timestamp)
    return data
end

"""
Output description of a DataFrame to the console.

:param data: DataFrame to be described.
"""
function describe_data(data::DataFrame)
    println(size(data))
    println(typeof(data))
    display(names(data))
    display(describe(data))
end
