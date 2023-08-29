import Graphs: DiGraph, add_edge!, inneighbors, outneighbors, indegree, outdegree, sum, nv
import Cairo, Fontconfig
import Random: shuffle, rand
using GraphPlot
using Compose

"""
An Event maps an array entry to a job id and a corresponding operation.
"""
struct Event
    "Reference to the job this event refers to."
    job::Int
    "Reference to the product the job of this event refers to."
    product::Int
    "Reference to the operation for the product this event refers to."
    operation::Int
    "Reference to the machine this event should be performed on."
    machine::Int
    "Duration of the event in seconds."
    duration::Int
    "Setup time (in case it is required) for this event in seconds."
    setup::Int
end

"""
Contains the mappings of events needed to perform the schedule building functions.
"""
struct EventsMaps
    "All events containing info about the job and operation they belong to."
    events::Vector{Event}
    "Indices of events that need to be scheduled."
    to_schedule::Vector{Int}
    "Indices of events that have corresponding workpieces in storage."
    in_storage::Vector{Int}
    "The product graph determining preceding and successive operations for each event."
    productgraph::DiGraph{Int}
    "Mapping to keep track of variables for each machine."
    machine_variables::Vector{Int}
    "Mapping of variables to seconds."
    varsecondmap::Vector{Int}

    function EventsMaps(prodsystem::ProductionSystem, varsecondmap)
        events, events_to_schedule, events_in_storage, productgraph = mapevents(prodsystem)
        machine_variables = mapvars(prodsystem, events, events_to_schedule)

        return new(events, events_to_schedule, events_in_storage, productgraph, machine_variables, varsecondmap)
    end
end

"""
Map the jobs and their operations into an array of events and the productgraph.
The productgraph interconnects all operations of each product. For example, if
there are two jobs, where the first job has a quantity of two and the second job
has a quantity of 1, the productgraph will be:

Job 1: 1 -->  2 -->  3 -->  4
X     X
Job 1: 5 -->  6 -->  7 -->  8

Job 2: 9 --> 10 --> 11

:param prodsystem: Object containing the structure of the production system.
:param debug: Set debug to true to save an image of the created product graph.
:return events: A list of events that occur during the scheduling process.
:return events_to_schedule: A list of events that actually need to be scheduled.
:return events_in_storage: A list of events that have been produced before (are in storage).
:return productgraph: The product graph described above.
"""
function mapevents(prodsystem::ProductionSystem; debug=false)
    # Determine the total number of events
    nevents = 0
    for job in prodsystem.jobs
        nevents += job.quantity * length(prodsystem.products[job.product].operations)
    end

    events = Event[]
    events_to_schedule = Int[]
    events_in_storage = Int[]

    productgraph = DiGraph{Int}(nevents)

    for (j_idx, job) in pairs(prodsystem.jobs), (o_idx, op) in pairs(prodsystem.products[job.product].operations)
        # Create a mapping of event numbers to steps in a specific job.
        for _ in 1:job.quantity
            push!(events, Event(j_idx, job.product, o_idx, op.machine, op.duration, op.setup))
        end

        # Create a mapping of pieces to be produced to events 
        produced_pieces = job.quantity - prodsystem.storeditems[j_idx][o_idx]
        if produced_pieces > 0
            append!(events_to_schedule, collect(1:produced_pieces) .+ (length(events) - job.quantity))
        end

        # Create a mapping of stored pieces to events
        stored_pieces = prodsystem.storeditems[j_idx][o_idx]
        if stored_pieces > 0
            append!(events_in_storage, collect(1:stored_pieces) .+ (length(events) - stored_pieces))
        end

        # Add edges to the product graph to connect the production steps of
        # each job. Each prior operation on a product must be connected to
        # all operations which could follow it.
        if o_idx > 1
            priorsteps = length(events)-2*job.quantity+1:length(events)-job.quantity
            currentsteps = length(events)-job.quantity+1:length(events)
            for p in priorsteps, c in currentsteps
                add_edge!(productgraph, p, c)
            end
        end
    end

    # Store the product graph as an image if debug is true.
    if debug
        draw(
            PNG("productgraph.png", 16cm, 16cm),
            gplot(productgraph, layout=shell_layout, nodelabel=1:nv(productgraph), arrowlengthfrac=0.05),
        )
    end

    return events, events_to_schedule, events_in_storage, productgraph
end

"""
To ensure that each machine always uses the same variables for scheduling the pauses
between its events, the machine_variables map is needed. It determines, which resource should
use which starting points for scheduling its events.

:param prodsystem: The production system object.
:param events: Vector of all events for scheduling.
:param events_to_schedule: Indices of those events that actually need to be scheduled (are not in storage).

:return: Mapping of resource id to the starting variable.
"""
function mapvars(prodsystem::ProductionSystem, events::Vector{Event}, events_to_schedule::Vector{Int})
    nmachineevents = zeros(Int, length(prodsystem.machines))

    for e_idx in events_to_schedule
        nmachineevents[events[e_idx].machine] += 1
    end

    machine_variables = ones(Int, length(prodsystem.machines))
    for r in 1:length(nmachineevents)-1
        machine_variables[r+1] = machine_variables[r] + nmachineevents[r]
    end

    return machine_variables
end

"""
Get the operation object for a specific event.

:param prodsystem: ProductionSystem object containing the production system structure.
:param envevts: Events mapping for scheduling.
:param event: Index of the actual event for which this function returns the operation object.
:return: The operation object.
"""
eventoperation(prodsystem::ProductionSystem, envevts::EventsMaps, event) =
    prodsystem.products[prodsystem.jobs[envevts.events[event].job].product].operations[envevts.events[event].operation]

"""
An item in the production schedule.
"""
struct ScheduleItem
    "Id of the job for this schedule item."
    job::Int
    "If of the product which this job represents."
    product::Int
    "Id of the operation for this schedule item."
    operation::Int
    "Id of the event for this schedule item."
    event::Int
    "Starting time in seconds after the beginning of the schedule."
    starttime::Int
    "Ending time in seconds after the beginning of the schedule."
    endtime::Int
    "Pause time before the schedule item begins."
    pausetime::Int
    "Setup time (after starttime) before the schedule item begins."
    setuptime::Int
    "Other coinciding schedule items (length of this vector is equal to number of items being processed concurrently)."
    coinciding::Vector{Int}
    "Boolean flag which is true if the item was in storage before the schedule starts."
    done::Bool
end

"""
An entire production schedule is a vector of schedule items.
"""
Schedule = Vector{ScheduleItem}

"""
Build a schedule using the events and variables determined by an optimization algorithm.

:param envevts: Event mappings known to the production system environment.
:param prodsystem: Structure of the production system with machines and products.
:param events: Events vector determined by the optimization.
:param variables: Variables vector determined by the optimization.
:param debug: If set to true this function will produce an image of the machine graph generated
              for the specific events vector given by the optimization.
:return: Tuple of schedule, machinegraph, machinestartingnodes, error. The schedule is an unsorted list
         of ScheduleItems. The machinegraph determines the order of these schedule items for each machine.
         The vector machinestartingnodes contains information about the starting nodes for each machine to
         speed up creating schedules for a single machine. Finally, error contains information about a
         potential error if one occured.
"""
function buildschedule(envevts::EventsMaps, prodsystem::ProductionSystem, events, variables; debug=false)
    schedule = Vector{ScheduleItem}(undef, length(envevts.events))
    currentresourcevars = copy(envevts.machine_variables)

    # Create a graph which contains a path for each machine according to the schedule.
    machinegraph = DiGraph(length(envevts.events))
    machinepriornode = zeros(Int, length(prodsystem.machines))
    machinestartingnodes = zeros(Int, length(prodsystem.machines))

    for event in events
        machine = envevts.events[envevts.to_schedule[event+1]].machine

        if machinepriornode[machine] == 0
            machinestartingnodes[machine] = envevts.to_schedule[event+1]
        else
            add_edge!(machinegraph, machinepriornode[machine], envevts.to_schedule[event+1])
        end
        machinepriornode[machine] = envevts.to_schedule[event+1]
    end

    if debug
        draw(
            PNG("machinegraph.png", 16cm, 16cm),
            GraphPlot.gplot(machinegraph, layout=spring_layout, nodelabel=1:nv(machinegraph), arrowlengthfrac=0.05),
        )
    end

    # keep track of nodes which have been visited before
    touchedfrom = falses(length(envevts.events), length(envevts.events) + 1)
    # keep track of nodes which have been used by antecessors
    usednodes = Set{Int}()
    sizehint!(usednodes, length(envevts.events))
    # keep track of nodes which have already been scheduled
    schedulednodes = Int[]
    sizehint!(schedulednodes, length(envevts.events))
    schedulednodes_bits = falses(length(envevts.events))
    # add scheduled nodes to schedulednodes and schedulednodes_bits
    for event in envevts.in_storage
        schedulednodes_bits[event] = true
        push!(schedulednodes, event)
        schedule!(schedule, envevts, event, true)
    end

    # Find starting nodes of both graphs. This means nodes which have indegree
    # of 0 in both the machinegraph and productgraph.
    starters = Vector{Tuple{Int, Int}}()
    sizehint!(starters, length(envevts.events))
    indegreesproducts = sum(envevts.productgraph, 1)
    indegreesmachines = sum(machinegraph, 1)
    for n in envevts.to_schedule
        if indegreesmachines[n] == 0
            push!(starters, (n, 1))
        end
    end

    while length(starters) > 0
        node, fromnode = popfirst!(starters)
        # If the node has already been touched, the graph contains a cycle (is invalid).
        if touchedfrom[node, fromnode]
            return schedule, machinegraph, machinestartingnodes, "Solution contains at least one cycle."
        else
            touchedfrom[node, fromnode] = true
            # If the node is already scheduled, jump to the next iteration.
            if schedulednodes_bits[node]
                continue
            end
        end

        # There should always be exactly one preceding node on the machine. We need to check
        # whether this node has been scheduled, yet.
        # If the preceding machine node has not been scheduled, the loop continues because the
        # current node cannot be scheduled, yet.
        if indegreesmachines[node] == 0
            reqmachinenode = 0
        else
            reqmachinenodes = schedulednodes ∩ inneighbors(machinegraph, node)
            if length(reqmachinenodes) > 0
                reqmachinenode = first(reqmachinenodes)
            else
                continue
            end
        end

        # Check whether any preceding nodes are required by the product (preceding producting operations
        # on the product). If this is the case, these operations must have been scheduled in previous runs
        # of the loop. Otherwise the loop continues, because the current node cannot be scheduled, yet.
        if indegreesproducts[node] == 0
            reqproductnode = 0
        else
            # The difference between incoming neighbor and usednodes is the set of nodes which
            # could still be available for scheduling. The intersection between these nodes and
            # and scheduled nodes is the set of nodes which can be used as predecessors of the
            # current node (are predecessors and have been scheduled already).
            reqproductnodes = schedulednodes ∩ setdiff(inneighbors(envevts.productgraph, node), usednodes)
            if length(reqproductnodes) > 0
                reqproductnode = first(reqproductnodes)
            else
                continue
            end
        end

        # Store the used preceding product node in usednodes.
        if reqproductnode != 0
            push!(usednodes, reqproductnode)
        end

        # Get the correct (next) variable for the resource which is being utilized by the node
        # being scheduled and advance the current resource vars.
        variable = variables[currentresourcevars[envevts.events[node].machine]]
        currentresourcevars[envevts.events[node].machine] += 1

        # Schedule the current node with the parameters determined above, store the node in
        # schedulednodes to denote that it could be used as a predecessor for later nodes.
        schedule!(schedule, envevts, prodsystem, node, reqmachinenode, reqproductnode, variable)
        push!(schedulednodes, node)
        schedulednodes_bits[node] = true

        # Push additional starting nodes. Outgoing edges of the scheduled node or the current
        # node, if it could not be scheduled, yet.
        for n in outneighbors(machinegraph, node) ∪ outneighbors(envevts.productgraph, node)
            push!(starters, (n, node + 1))
        end
    end

    # Check whether all events have actually been scheduled.
    if sum(schedulednodes_bits) != length(envevts.events)
        return schedule,
        machinegraph,
        machinestartingnodes,
        "Building the schedule failed - there were unreachable events."
    end

    return schedule, machinegraph, machinestartingnodes, nothing
end

"""
Generate schedules based on "expert" methods. Currently only dispatching rules are implemented. The
method to be used for generating the schedules is specified using the method parameter.

:param envset: The EnvironmentSettings object.
:param envevts: The EventsMaps object.
:param prodsystem: Production system structure.
:param method: Method to use for generation of the schedules. One of {'spt': shortest processing time, ...}
:param count: Number of Schedules to be generated
"""
function generate_expert_schedules(
    envset::EnvironmentSettings,
    envevts::EventsMaps,
    prodsystem::ProductionSystem,
    method::String,
    count::Int=1,
)
    # Create an array to keep track of all starting nodes.
    starters = Int[]
    sizehint!(starters, length(envevts.events))
    # Keep track of nodes which have been used by antecessors
    usednodes = Set{Int}()
    sizehint!(usednodes, length(envevts.events))
    # Find starting nodes on the product graph (starting nodes have no inneighbors)
    indegreesproducts = sum(envevts.productgraph, 1)
    for node in envevts.to_schedule
        if indegreesproducts[node] == 0
            push!(starters, node)
        elseif length(inneighbors(envevts.productgraph, node) ∩ setdiff(envevts.in_storage, usednodes)) > 0
            push!(usednodes, first(setdiff(envevts.in_storage, usednodes)))
            push!(starters, node)
        end
    end

    machinenodes = [Int[] for _ in prodsystem.machines]
    for (idx, event) in pairs(envevts.events)
        push!(machinenodes[event.machine], idx)
    end

    # Get maximum values for all variables
    var_maxvalues = fill(length(envevts.varsecondmap) - 1, length(envevts.to_schedule))

    events = Matrix{Int}(undef, (count, length(envevts.to_schedule)))
    variables = Matrix{Int}(undef, (count, length(envevts.to_schedule)))

    @info "Generating $count expert schedules with $method method."
    for i in 1:count
        if lowercase(method) == "spt"
            events[i, :], variables[i, :] =
                _generate_spt_schedule(envevts, prodsystem, envset.rng, starters, machinenodes, var_maxvalues)
        else
            error("Specified method '$method' for expert schedule creation is unavailable.")
        end
    end

    # Convert output to a python recarray
    var_dtype = typeof(variables[1]) <: AbstractFloat ? np.float64 : np.int64
    dtype = pycall(
        np.dtype,
        PyObject,
        [("events", (np.int64, length(events[1, :]))), ("variables", (var_dtype, length(variables[1, :])))],
    )

    output =
        @pycall np.core.records.fromrecords([(events[i, :], variables[i, :]) for i in 1:count]; dtype=dtype)::PyObject
    return output
end

"""
Generate schedules based on the shortest processing time dispatching rule

:param envevts: The EventsMaps object.
:param prodsystem: Production system structure.
:param rng: Random number generator.
:param starters: Possible starting nodes for the schedule.
:param machinenodes: Sets of nodes for each machine.
:param var_maxvalues: Maximum values for each variable.
"""
function _generate_spt_schedule(
    envevts::EventsMaps,
    prodsystem::ProductionSystem,
    rng,
    starters::Vector{Int},
    machinenodes::Vector{Vector{Int}},
    var_maxvalues::Vector{Int},
)
    # Make sure we have an independent copy of the starters and current resource variables vectors.
    starters = copy(starters)
    currentresourcevars = copy(envevts.machine_variables)
    # Create the schedule.
    schedule = Vector{ScheduleItem}(undef, length(envevts.events))

    # Create a vector of random variables.
    variables = [rand(rng, 0:maxval) for maxval in var_maxvalues]

    # Provide a location for the durations of each event.
    durations = fill(typemax(Int), length(envevts.events))
    # Keep track of nodes which have been used by antecessors.
    usednodes = Set{Int}()
    sizehint!(usednodes, length(envevts.events))
    # Keep track of nodes which have already been scheduled.
    schedulednodes = Int[]
    sizehint!(schedulednodes, length(envevts.events))
    # Add scheduled nodes to schedulednodes and schedule them.
    for event in envevts.in_storage
        push!(schedulednodes, event)
        schedule!(schedule, envevts, event, true)
    end

    # Keep track of the preceding nodes used on each machine.
    precedingmachinenodes = zeros(Int, length(prodsystem.machines))
    new_precedingmachinenodes = copy(precedingmachinenodes)
    # Find starting nodes on the product graph (starting nodes have no inneighbors)
    indegreesproducts = sum(envevts.productgraph, 1)

    while length(starters) > 0
        # Find the duration for each node in the starter array. This has to be repeated for each 
        # iteration because the setup time depends on the preceding node on that machine.
        for node in starters
            _, setup, duration, coincides_with = schedule_times(
                schedule,
                envevts,
                prodsystem,
                node,
                precedingmachinenodes[envevts.events[node].machine],
                0,
            )
            durations[node] = coincides_with == 0 ? setup + duration : 0
        end

        # Determine the nodes with the shortest processing time for each machine and insert
        # them into the nextnodes_possible array.
        nextnodes_possible = [Int[] for _ in prodsystem.machines]
        nextdurations = fill(typemax(Int), length(prodsystem.machines))
        for (m_id, machine) in pairs(machinenodes), node in machine
            !(node in starters) && continue

            if durations[node] < nextdurations[m_id]
                nextdurations[m_id] = durations[node]
                nextnodes_possible[m_id] = Int[node]
            elseif durations[node] == nextdurations[m_id]
                push!(nextnodes_possible[m_id], node)
            end
        end

        # Randomly select the next nodes for each machine, then shuffle the resulting array to
        # ensure that generated schedules are different.
        nextnodes = Int[]
        for (m_id, machine) in pairs(nextnodes_possible)
            if isempty(machine)
                new_precedingmachinenodes[m_id] = precedingmachinenodes[m_id]
                continue
            end
            node = machine[rand(rng, 1:length(machine))]
            push!(nextnodes, node)
            new_precedingmachinenodes[m_id] = node
        end
        shufflednext = shuffle(rng, nextnodes)

        for node in shufflednext
            # Check whether any preceding nodes are required by the product (preceding producting operations
            # on the product). If this is the case, these operations must have been scheduled in previous runs
            # of the loop. Otherwise the loop continues, because the current node cannot be scheduled, yet.
            if indegreesproducts[node] == 0
                reqproductnode = 0
            else
                # The difference between incoming neighbor and usednodes is the set of nodes which
                # could still be available for scheduling. The intersection between these nodes and
                # and scheduled nodes is the set of nodes which can be used as predecessors of the
                # current node (are predecessors and have been scheduled already).
                reqproductnodes = schedulednodes ∩ setdiff(inneighbors(envevts.productgraph, node), usednodes)
                if length(reqproductnodes) > 0
                    reqproductnode = first(reqproductnodes)
                else
                    continue
                end
            end

            # Store the used preceding product node in usednodes.
            if reqproductnode != 0
                push!(usednodes, reqproductnode)
            end

            # Get the correct (next) variable for the resource which is being utilized by the node
            # being scheduled and advance the current resource vars.
            variable = variables[currentresourcevars[envevts.events[node].machine]]
            currentresourcevars[envevts.events[node].machine] += 1

            schedule!(
                schedule,
                envevts,
                prodsystem,
                node,
                precedingmachinenodes[envevts.events[node].machine],
                reqproductnode,
                variable,
            )
            push!(schedulednodes, node)
        end
        precedingmachinenodes = copy(new_precedingmachinenodes)
        # Remove the scheduled nodes from starters.
        starters = setdiff(starters, nextnodes)

        # Push additional starting nodes. Outgoing edges of the scheduled node on the productgraph
        for node in nextnodes
            possible_next = setdiff(outneighbors(envevts.productgraph, node), usednodes, schedulednodes, starters)
            if !isempty(possible_next)
                next =
                    length(possible_next) > 1 ? possible_next[rand(rng, 1:length(possible_next))] : first(possible_next)
                push!(starters, next)
            end
        end
    end

    # Create the event schedule which is transferred to the agent.
    eventschedule = Int[]
    for scheduleitem in schedulednodes[length(envevts.in_storage) + 1:end]
        for (e_idx, evt) in pairs(envevts.to_schedule)
            if evt == scheduleitem
                push!(eventschedule, e_idx - 1)
                break
            end
        end
    end

    return eventschedule, variables
end

"""
Schedule an event by calculating its ending time and creating the ScheduleItem object.

:param schedule: The entire production schedule with multiple ScheduleItems, which is changed by this function.
:param eventid: Index of the event which resulted in this ScheduleItem.
:param jobid: Index of the job which this ScheduleItem belongs to.
:param productid: Index of the product requested by the job.
:param opid: Index of the operation performed during this ScheduleItem.
:param starttime: Starting time of the ScheduleItem (before setup time starts).
:param duration: Processing duration for the ScheduleItem.
:param pausetime: Additional pause to added before the starttime (just informational - no calculation takes place here.)
:param setup: Setuptime needed after starttime before the actual processing can start.
:param coincides_with: Other Scheduleitems which this item coincides with due to available machine capacity.
:param done: Flag indicating whether this item is done before scheduling starts (item was in storage).
"""
function schedule!(
    schedule::Schedule,
    eventid::Int,
    jobid,
    productid,
    opid,
    starttime,
    duration,
    pausetime,
    setup,
    coincides_with,
    done,
)
    # Calculate endtime
    if !done
        endtime = starttime + setup + duration
    else
        endtime = 0
    end

    if coincides_with != 0
        push!(schedule[coincides_with].coinciding, eventid)
        coinciding = schedule[coincides_with].coinciding
    else
        coinciding = Int[eventid]
    end

    schedule[eventid] =
        ScheduleItem(jobid, productid, opid, eventid, starttime, endtime, pausetime, setup, coinciding, done)
end

"""
Schedule an item in storage for time zero.

:param schedule: The entire production schedule with multiple ScheduleItems, which is changed by this function.
:param envevts: Events of the environment to determine job, product and operation for the event.
:param event: Identifier of the event this ScheduleItem belongs to.
:param done: Flag indicating whether this item is done before scheduling starts (item was in storage). For this method,
             done can only be true
"""
function schedule!(schedule::Schedule, envevts::EventsMaps, event, done::Bool)
    if done
        schedule!(
            schedule,
            event,
            envevts.events[event].job,
            envevts.events[event].product,
            envevts.events[event].operation,
            0,
            0,
            0,
            0,
            0,
            true,
        )
    else
        error("This function can only schedule events in storage (done = true).")
    end
end

"""
Schedule an item based on preceding items on the machine and for the same product.

:param schedule: The entire production schedule with multiple ScheduleItems, which is changed by this function.
:param envevts: Events of the environment to determine job, product, and operation for the specific event.
:param prodsystem: Object representing the production system, especially regarding products and machines.
:param event: Identifier of the event this ScheduleItem should belong to.
:param precedingmachine: Identifier of the preceding event on the same machine.
:param precedingproduct: Identifier of a preceding event for the same product.
:param variable: Index of the variable to be used to determine the additional pause time before the event.
"""
function schedule!(
    schedule::Schedule,
    envevts::EventsMaps,
    prodsystem::ProductionSystem,
    event,
    precedingmachine,
    precedingproduct,
    variable,
)
    earlieststart, setup, duration, coincides_with =
        schedule_times(schedule, envevts, prodsystem, event, precedingmachine, precedingproduct)

    # Determine starting time of the event based on coinciding operations or separately depending on the situation.
    if coincides_with != 0
        pausetime = schedule[precedingmachine].pausetime
        starttime = schedule[precedingmachine].starttime
    else
        # Get the preceding pause from the variables (only if not coinciding with previous event).
        pausetime = envevts.varsecondmap[variable+1]
        # Calculate the starting time
        starttime = earlieststart + pausetime
    end

    schedule!(
        schedule,
        event,
        envevts.events[event].job,
        envevts.events[event].product,
        envevts.events[event].operation,
        starttime,
        duration,
        pausetime,
        setup,
        coincides_with,
        false,
    )
end

"""Calculate the production durations for an event to be scheduled.

:param schedule: The schedule to which the event would be added.
:param envevts: Events of the environment to determine job, product, and operation for the specific event.
:param prodsystem: Object representing the production system, especially regarding products and machines.
:param event: Identifier of the event this ScheduleItem should belong to.
:param precedingmachine: Identifier of the preceding event on the same machine.
:param precedingproduct: Identifier of a preceding event for the same product.
"""
function schedule_times(
    schedule::Schedule,
    envevts::EventsMaps,
    prodsystem::ProductionSystem,
    event,
    precedingmachine,
    precedingproduct,
)
    productendtime = precedingproduct == 0 ? 0 : schedule[precedingproduct].endtime

    coinciding_with_previous = false
    if precedingmachine == 0
        machineendtime = 0

        # If there is a previous item scheduled on the machine and the machine's capacity is greater
        # than what is already scheduled, more events can be scheduled for the same point in time.
        # In that case we differentiate whether the machine requires unique jobs, which in turn means that
        # the product and operation of the current and previous processes have to be equal. If the machine
        # does not specify this requirement, any job can be scheduled at the same time as a previous job.
    elseif length(schedule[precedingmachine].coinciding) <
           prodsystem.machines[envevts.events[event].machine].capacity &&
           !schedule[precedingmachine].done &&
           ((
               (
                   prodsystem.machines[envevts.events[event].machine].unique_job &&
                   schedule[precedingmachine].product == envevts.events[event].product &&
                   schedule[precedingmachine].operation == envevts.events[event].operation
               ) || !prodsystem.machines[envevts.events[event].machine].unique_job
           )) &&
           schedule[precedingmachine].starttime > productendtime

        machineendtime = schedule[precedingmachine].starttime
        coinciding_with_previous = true

        # Finally, once the machine's capacity is exceeded, the current job has to start after the previous
        # job is finished.
    else
        machineendtime = schedule[precedingmachine].endtime
    end

    # The earliest starting time depends on the latest ending time of the preceding events.
    earlieststart = max(machineendtime, productendtime)

    # Calculate the preparation time for the job based on whether the machine requires unique jobs and whether there
    # are coinciding events on the same machine.
    if coinciding_with_previous
        setup = schedule[precedingmachine].setuptime
    elseif (
        precedingmachine != 0 &&
        schedule[precedingmachine].product == envevts.events[event].product &&
        schedule[precedingmachine].operation == envevts.events[event].operation
    )
        setup = 0
    else
        setup = eventoperation(prodsystem, envevts, event).setup
    end
    coincides_with = !coinciding_with_previous ? 0 : precedingmachine

    return earlieststart, setup, eventoperation(prodsystem, envevts, event).duration, coincides_with
end

"""
Determine the resulting schedules for all machines by sorting the main production schedule by time and machine.

:param envevts: Events in the production schedule.
:param schedule: The stored schedule items to determine times for each of the events.
:param machinegraph: The machine graph sets the order of events for each machine.
:param machinestartingnodes: The first event for each machine from where the function begins following the order
                             set by the machine graph.
:return: Vector of schedules for each machine.
"""
function machineschedule(
    envevts::EventsMaps,
    schedule::Schedule,
    machinegraph::DiGraph,
    machinestartingnodes::Array{Int},
)
    machineschedules = Vector{Vector{ScheduleItem}}(undef, length(machinestartingnodes))
    machines = falses(length(machinestartingnodes))
    for node in machinestartingnodes
        node == 0 && continue
        machines[envevts.events[node].machine] = true

        machineschedules[envevts.events[node].machine] = _machineschedule(machinegraph, schedule, node)
    end
    for (idx, machine) in pairs(machines)
        if !machine
            machineschedules[idx] = ScheduleItem[]
        end
    end

    return machineschedules
end

"""
Determine the schedule for a specific machine by sorting the main production schedule by time and machine.

:param envevts: Events in the production schedule.
:param schedule: The stored schedule items to determine times for each of the events.
:param machinegraph: The machine graph sets the order of events for each machine.
:param machinestartingnodes: The first event for each machine from where the function begins following the order
                             set by the machine graph.
:param machine: Identifier of the machine for which the schedule should be created.
:return: Schedule for the selected machine
"""
function machineschedule(
    envevts::EventsMaps,
    schedule::Schedule,
    machinegraph::DiGraph,
    machinestartingnodes::Array{Int},
    machine::Int,
)
    startingnode = nothing
    for starter in machinestartingnodes
        starter == 0 && continue

        if envevts.events[starter].machine == machine
            startingnode = [starter]
            break
        end
    end
    if startingnode === nothing
        error("The requested machine $machine could not be found.")
    end

    return _machineschedule(machinegraph, schedule, startingnode)
end

"""
Create a schedule for a specific machine by following the machine graph.

:param machinegraph: The machine graph sets the order of events for each machine.
:param schedule: The stored schedule items to determine times for each of the events.
:param startingnode: The first event occuring on the machine to tell the function where
                     it should start traversing the machine graph.
"""
function _machineschedule(machinegraph::DiGraph, schedule::Schedule, startingnode::Int)
    machineschedule = ScheduleItem[]
    node = startingnode
    while outdegree(machinegraph, node) > 0
        push!(machineschedule, schedule[node])
        node = first(outneighbors(machinegraph, node))
    end
    push!(machineschedule, schedule[node])
    return machineschedule
end

function log_step_errors(infos::Vector{Dict{String, Any}})
    problems = Dict{String, Int}()
    for info in infos
        if haskey(info, "valid") && info["valid"] == false
            key = haskey(info, "error") ? info["error"] : "other"
            if !haskey(problems, key)
                problems[key] = 1
            else
                problems[key] += 1
            end
        end
    end
    if !isempty(problems)
        message = String[]
        push!(message, "ERRORS during environment step:")
        for (name, occurrences) in pairs(problems)
            push!(message, "Error '$name' occurred $occurrences times.")
        end

        join(message, "\n")
    end
end
