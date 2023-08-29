"""
Decode bit arrays like those supplied by modbus devices.

:param Type: Type to interpret the value as.
:param value: Bit array value (string or integer number).
:param byteorder: Byte order (endianness) of the value.
:return: Converted value.
"""
function decode_uint16_bitarray(
    ::Type{T},
    value::Union{Missing, S};
    byteorder::Byteorder=LittleEndian
) where {T <: Number, S <: AbstractString}
    if ismissing(value)
        return missing
    end

    strings = bitstring.(parse.(UInt16, strip.(s -> s ∈ ['[', ']', ' '], split(value, r"[,;]"))))
    t = Dict(1 => UInt16, 2 => UInt32, 4 => UInt64)
    if !(length(strings) ∈ keys(t))
        error("The modbus value array has an invalid length. Lengths 1, 2 and 4 are supported. Got $(length(strings))")
    end

    if byteorder == SYS_BYTEORDER
        integer = parse(t[length(strings)], join(strings), base=2)
    else
        integer = parse(t[length(strings)], join(reverse(strings)), base=2)
    end
    reinterpret(T, integer)::T
end

"""
Decode multiple bit arrays like those supplied by modbus devices.

:param Type: Type to interpret the values as.
:param value: Bit array values (strings or integer numbers).
:param byteorder: Byte order (endinanness) of the value.
:return: List of converted values.
"""
function decode_uint16_bitarray(::Type{T}, value::AbstractArray; byteorder::Byteorder=LittleEndian) where {T <: Number}
    new = Array{Union{T, Missing}}(undef, size(value))
    for (i, value) in enumerate(value)
        new[i] = decode_uint16_bitarray(T, value, byteorder=byteorder)
    end
    return new
end

"""
Encode an array of strings as numbers.

:param values: Array of strings.
:param mapping: Dictionary of mappings.
:return: Mapped values.
"""
function encode_str2num(
    values::AbstractArray{Union{Missing, S}},
    mapping::Dict{S, N},
) where {S <: AbstractString, N <: Number}
    programs = keys(mapping)
    new = Array{Union{Missing, N}}(missing, size(values))
    for (i, value) in enumerate(values)
        if ismissing(value)
            continue
        end

        for p in programs
            if occursin(p, value)
                new[i] = mapping[p]
                break
            end
        end
    end
    return new
end

"""
Create a onehote encoding for a vector of values.

:param values: Vector of values (strings).
:return: Array of encoded values (each value in the original array has a separate column in the output).
"""
function encode_onehot(values::AbstractVector{Union{Missing, S}}) where {S <: AbstractString}
    names = Set(values)
    if missing ∈ names
        pop!(names, missing)
    end

    out = DataFrame([Symbol(name) => Vector{Union{Missing, Bool}}(missing, length(values)) for name in names])
    for (i, value) in enumerate(values)
        if ismissing(value)
            out[i, :] .= missing
            continue
        end

        out[i, :] .= false
        out[i, Symbol(value)] = true
    end
    return out
end

function moving_average(x::Union{Vector{T}, Vector{Union{Missing, T}}}, window::Int) where {T <: Real}
    y = Vector{Union{Missing, T}}(undef, length(x))
    for i in 1:length(x)
        y[i] = i < window ? missing : mean(skipmissing(x[i - window + 1:i]))
    end
    return y
end
