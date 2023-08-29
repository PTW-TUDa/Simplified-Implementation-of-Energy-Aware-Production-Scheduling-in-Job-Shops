using Dates

"""
Create a bar plot, similar to a gannt chart, showing when a specific value is active or not.

:param plt: A plot object which this plot is added to.
:param data: Data to be plotted (should be one hot encoded data).
:param categories: Each category will have a separate line on the bar chart
:param args: arguments to the plotting function.
:param kwargs: keyword arguments to the plotting function.
"""
function barmap!(plt, data, categories::AbstractDict, args...; kwargs...)
    previous = data[firstindex(data)]
    barstart = -1

    for (index, value) in enumerate(data)
        index == firstindex(data) && continue  # Skip first index

        if ismissing(previous) && ismissing(value)
            continue
        elseif ismissing(previous) || (!ismissing(value) && previous != value && barstart == -1)
            previous = value
            barstart = index
        elseif ismissing(value) || (!ismissing(previous) && previous != value && barstart != -1)
            if previous in keys(categories)
                bar!(plt, (categories[previous], index - 1), args...; fillrange=barstart, label=nothing, kwargs...)
            end

            previous = value
            barstart = value in keys(categories) ? index : -1
        end
    end
end

"""
Create a bar plot, similar to a gannt chart, showing when a specific value is active or not.

:param plt: A plot object which this plot is added to.
:param data_x: Data for the x-axis.
:param data_y: Data to be plotted on the y-axis (should be one hot encoded data).
:param categories: Each category will have a separate line on the bar chart
:param args: arguments to the plotting function.
:param kwargs: keyword arguments to the plotting function.
"""
function barmap!(plt, data_x, data_y, categories::AbstractDict, args...; kwargs...)
    previous_y = data_y[firstindex(data_y)]
    barstart = nothing

    for (index, value) in enumerate(data_y)
        index == firstindex(data_y) && continue  # Skip first index

        if ismissing(previous_y) && ismissing(value)
            continue
        elseif ismissing(previous_y) || (!ismissing(value) && previous_y != value && isnothing(barstart))
            previous_y = value
            barstart = Dates.value(data_x[index])
        elseif ismissing(value) || (!ismissing(previous_y) && previous_y != value && !isnothing(barstart))
            if previous_y in keys(categories)
                bar!(
                    plt,
                    (categories[previous_y], Dates.value(data_x[index])),
                    args...;
                    fillrange=barstart,
                    label=nothing,
                    kwargs...,
                )
            end

            previous_y = value
            barstart = value in keys(categories) ? Dates.value(data_x[index]) : nothing
        end
    end
    
    xlims!(plt, (Dates.value(first(data_x)), Dates.value(last(data_x))))
end

"""
Create a bar plot, similar to a gannt chart, showing when a specific category is active or not.


:param: data: Data to be plotted (should be one hot encoded data).
:args: arguments to the plotting function.
"""
barmap(data, args...) = barmap!(plot(), data, args...)

"""
Create a bar plot, similar to a gannt chart, showing when a specific category is active or not.

:param data_x: Data for the x-axis.
:param data_y: Data for the y-axis.
:param args: arguments to the plotting function.
"""
barmap(data_x, data_y, args...) = barmap!(plot(), data_x, data_y, args...)
