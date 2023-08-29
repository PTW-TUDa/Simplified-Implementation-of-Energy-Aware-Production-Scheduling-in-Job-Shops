using PyCall

"""
An operation performed on a production machine.
"""
struct Operation
    "Unique identifier of the operation."
    id::Int
    "Sorting value used to order the operations for a product."
    sort::Int
    "Idenfier of the machine that performs this operation."
    machine::Int
    "Duration of the operation in seconds."
    duration::Int
    "Duration of setup for the operation in seconds."
    setup::Int
    "Additional parameter to represent process specific energy consumption."
    processparam::Float64
end

"""
An object representing a product which can be produced in the factory.
"""
struct Product
    "Human readable name of the product."
    name::String
    "Unique identifier of the product."
    id::Int
    "Array of operations that need to be performed to manufacture the product."
    operations::Vector{Operation}
end

"""
A job belongs to a product and defines a quantity of workpieces to be produced.
"""
struct Job
    "Index of the product in the list of products of the production system."
    product::Int
    "Quantity of workpieces to be produced."
    quantity::Int
end

"""
Mapping of a job number to the number of stored items for each of its operations.
"""
StorageMap = Dict{Int, Vector{Int}}

"""
Instantiates a production system from scenario configuration files. The production system consists
of machines and products. Jobs are processed on machines and there are stored workpieces.

:param pathscenario: Path to the scenario files.
:param machines_file: Filename of a JSON file with machine configurations.
:param products_file: Filename of a JSON file with product configurations.
:param orders_file: Filename of a JSON file with an order and storage configuration.
"""
struct ProductionSystem
    "A vector of machines available in the production system."
    machines::Vector{Machine}
    "Products produced by the production system."
    products::Vector{Product}
    "Customer orders for products are jobs."
    jobs::Vector{Job}
    "Stored items are partially processed workpieces in storage."
    storeditems::StorageMap

    function ProductionSystem(pathscenarios, machines_file, products_file, orders_file)
        machines, products = import_resources(pathscenarios, machines_file, products_file)
        jobs, storeditems = import_orders(products, pathscenarios, orders_file)
        storagemap = map_storage(storeditems, products, jobs)
        
        return new(machines, products, jobs, storagemap)
    end
end

"""
Read json files containing machines available in a factory and products which can be produced with
the production system.

:param pathscenarios: Absolute path to scenario files.
:param machines_file: Relative path to the json file with information about the machines.
:param products_file: Relative path to the json file with information about the products.
:return: Tuple of arrays of Machine objects and Product objects.
"""
function import_resources(pathscenarios, machines_file, products_file)
    # Read the json file specifying the production resources and create a
    # corresponding mapping in the environment. Each resource also has an
    # associated energy mode.
    _machines = etautility.json_import(joinpath(abspath(pathscenarios), machines_file))
    machines = Machine[]

    for machine in _machines
        if !("name" in keys(machine))
            error("A machine could not be imported from '$machines_file' because the 'name' field is missing.")
        end

        if !("type" in keys(machine))
            error("Machine '$machine[\"name\"] could not be imported from '$machines_file' because the 
                  'type' field is missing")
        end

        push!(machines, import_machine(machine["type"], machine["name"], machine, machines_file))
    end

    # Read the products from a json file. Each job has a number of operations
    # which are performed on different resources (machines).
    _products = etautility.json_import(joinpath(abspath(pathscenarios), products_file))
    products = Product[]

    for product in _products
        if !("name" in keys(product))
            error("A job could not be imported from '$products_file' because the 'name' field is missing.")
        end
        if !("id" in keys(product))
            error(
                "Job '$product[\"name\"]' could not be imported from '$products_file' because the 'id' field is missing.",
            )
        end
        if !("operations" in keys(product))
            error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
            the 'operations' array is missing.")
        end

        _operations = Operation[]

        for operation in product["operations"]
            if !("id" in keys(operation))
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the 'id' field is missing from operation number $o is missing.")
            end

            if !("machine" in keys(operation))
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the 'machine' field is missing from operation id $operation[\"id\"] is missing.")
            end

            # Find the index in the machine array for the specified machine id.
            machine_idx = nothing
            for (m_idx, m) in pairs(machines)
                if m.id == operation["machine"]
                    machine_idx = m_idx
                    break
                end
            end
            if machine_idx === nothing
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the the machine id '$operation[\"machine\"]' did not exist in the machines configuration 
                (see operation id '$operation[\"id\"]').")
            end

            if !("sort" in keys(operation))
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the 'sort' field is missing from operation id $operation[\"id\"] is missing.")
            end
            if !("duration" in keys(operation))
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the 'duration' field is missing from operation id $operation[\"id\"] is missing.")
            end
            if !("setup" in keys(operation))
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the 'setup' field is missing from operation id $operation[\"id\"] is missing.")
            end
            if !("processparam" in keys(operation))
                error("Job '$product[\"name\"]' could not be imported from '$products_file' because 
                the 'processparam' field is missing from operation id $operation[\"id\"] is missing.")
            end

            push!(
                _operations,
                Operation(operation["id"], operation["sort"], machine_idx, operation["duration"], operation["setup"], operation["processparam"]),
            )
        end

        push!(products, Product(product["name"], product["id"], sort!(_operations, by=x -> x.sort)))
    end

    return machines, products
end

"""
Import a machine object from data in a dictionary.

:param type: Type of the machine to import, for example 'machinetool' or 'cleaningmachine'.
:param name: Name of the machine (human readable).
:param data: Dictionary  describing the machine.
:param errorinfo: Description string to add to error descriptions (for example a file name).
"""
function import_machine(type::String, name::String, data, errorinfo="file")
    if !("capacity" in keys(data))
        error("Machine $name could not be imported from '$errorinfo' because the 'capacity' field is missing.")
    end
    if !("resource_id" in keys(data))
        error("Machine $name could not be imported from '$errorinfo' because the 'resource_id' field is missing.")
    end
    if !("unique_job" in keys(data))
        error("Machine $name could not be imported from '$errorinfo' because the 'unique_job' field is missing.")
    end
    if !("parameters" in keys(data))
        error("Machine $name  could not be imported from '$errorinfo' because the 'parameters' field is missing.")
    end

    if type == "machinetool"
        if !("beta_th_c" in keys(data["parameters"]))
            error(
                "Machine Tool $name could not be imported from '$errorinfo' because the 'beta_th_c' parameter is missing.",
            )
        end
        if !("beta_th_o" in keys(data["parameters"]))
            error(
                "Machine Tool $name could not be imported from '$errorinfo' because the 'beta_th_o' parameter is missing.",
            )
        end
        if !("beta_el" in keys(data["parameters"]))
            error(
                "Machine Tool $name could not be imported from '$errorinfo' because the 'beta_el' parameter is missing.",
            )

            if length(data["parameters"]["beta_el"] != 4)
                error("Machine Tool $name could not be imported from '$errorinfo' because the 'beta_el' parameter 
                        does not have exactly four values..")
            end
        end

        machine = MachineTool(
            name,
            data["resource_id"],
            data["capacity"],
            data["unique_job"],
            MachineToolParameterData(
                data["parameters"]["beta_el"],
                data["parameters"]["beta_th_c"],
                data["parameters"]["beta_th_o"],
                data["parameters"]["beta_c_M"],
                data["parameters"]["T_c"],
            ),
        )
    elseif type == "cleaningmachine"
        if !("electric" in keys(data))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'electric' field is missing.",
            )
        end
        if !("t_upper" in keys(data))
            error("Machine $name  could not be imported from '$errorinfo' because the 't_upper' field is missing.")
        end
        if !("t_lower" in keys(data))
            error("Machine $name  could not be imported from '$errorinfo' because the 't_lower' field is missing.")
        end
        if !("beta_el" in keys(data["parameters"]))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'beta_el' parameter is missing.",
            )
        end
        if !("beta_th_h" in keys(data["parameters"]))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'beta_th_h' parameter is missing.",
            )
        end
        if !("beta_th_o" in keys(data["parameters"]))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'beta_th_o' parameter is missing.",
            )
        end
        if !("beta_th_wp" in keys(data["parameters"]))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'beta_th_wp' parameter is missing.",
            )
        end
        if !("beta_th_s" in keys(data["parameters"]))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'beta_th_s' parameter is missing.",
            )
        end
        if !("beta_c_M" in keys(data["parameters"]))
            error(
                "Cleaning Machine $name  could not be imported from '$errorinfo' because the 'beta_c_M' parameter is missing.",
            )
        end

        machine = CleaningMachine(
            name,
            data["resource_id"],
            data["capacity"],
            data["unique_job"],
            data["t_lower"],
            data["t_upper"],
            CleaningMachineParameterData(
                data["electric"],
                data["parameters"]["beta_el"],
                data["parameters"]["beta_th_h"],
                data["parameters"]["beta_th_o"],
                data["parameters"]["beta_th_wp"],
                data["parameters"]["beta_th_s"],
                data["parameters"]["beta_c_M"],
            ),
        )
    end
    return machine
end

"""
Object identifying stored workpieces which can be utilized by a job.
"""
mutable struct StoredItem
    "Index of the product in the list of products of the production system."
    product::Int
    "Index of the operation within the product."
    operation::Int
    "Quantity of workpieces in storage."
    quantity::Int
end

"""
Import customer orders for scheduling from a JSON file.

:param products: Vector of products known to the production system environment.
:param pathscenarios: Path where the system should look for the orders file.
:param orders_file: Filename of the orders file.
:return: Tuple of vectors of Job objects and StoredItem objects.
"""
function import_orders(products::Vector{Product}, pathscenarios, orders_file)
    _orders = etautility.json_import(joinpath(abspath(pathscenarios), orders_file))
    jobs = Job[]

    if !("orders" in keys(_orders))
        error("Orders could not be imported from '$orders_file' because the 'orders' list is missing.")
    end

    for (idx, order) in pairs(_orders["orders"])
        if !("product" in keys(order))
            error("Order number '$idx' could not be imported from '$orders_file' because the product field is missing.")
        end

        product_idx = nothing
        for (p_idx, p) in pairs(products)
            if p.id == order["product"]
                product_idx = p_idx
                break
            end
        end

        if product_idx === nothing
            error("Order number '$idx' could not be imported from '$orders_file' because the product 
            '$order[\"product\"]' is not defined in the production system.")
        end

        if !("quantity" in keys(order))
            error(
                "Order number '$idx' could not be imported from ' $orders_file' because the quantity field is missing.",
            )
        end

        push!(jobs, Job(product_idx, order["quantity"]))
    end

    storeditems = StoredItem[]
    if !("storeditems" in keys(_orders))
        error("Stored Items could not be imported from '$orders_file' because the 'storeditems' list is missing.")
    end

    for (idx, item) in pairs(_orders["storeditems"])

        # Search for the product requested by the order and save its index.
        if !("product" in keys(item))
            error(
                "Stored item number '$idx' could not be imported from '$orders_file' because the 'product' field is missing.",
            )
        end
        product_idx = nothing
        for (p_idx, p) in pairs(products)
            if p.id == item["product"]
                product_idx = p_idx
                break
            end
        end
        if product_idx === nothing
            error("Stored item number '$idx' could not be imported from '$orders_file' because the product 
            '$item[\"product\"]' is not defined in the production system.")
        end

        # Search for the operation requested by the stored item and save its index.
        if !("operation" in keys(item))
            error(
                "Stored item number '$idx' could not be imported from ' $orders_file' because the 'operation' field is missing.",
            )
        end
        operation_idx = nothing
        for (op_idx, op) in pairs(products[product_idx].operations)
            if op.id == item["operation"]
                operation_idx = op_idx
                break
            end
        end
        if operation_idx === nothing
            error("Stored item number '$idx' could not be imported from '$orders_file' because the operation 
            '$item[\"operation\"]' is not defined for product '$item[\"product\"]' in the production system.")
        end

        if !("quantity" in keys(item))
            error(
                "Stored item number '$idx' could not be imported from ' $orders_file' because the 'quantity' field is missing.",
            )
        end

        push!(storeditems, StoredItem(product_idx, operation_idx, item["quantity"]))
    end

    return jobs, storeditems
end

"""
Create a neat mapping of products and operations to the number of stored items.

:param storeditems: A vector of stored items with no particular order.
:param products: All products known to the production system environment.
:param jobs: All jobs known to the production system environment.
:return: Mapping of the number of stored items to jobs and operations.
"""
function map_storage(storeditems::Vector{StoredItem}, products::Vector{Product}, jobs::Vector{Job})
    sort!(storeditems, by=x -> x.operation, rev=true)

    # Create the storage map with a vector of stored items for each job
    storagemap = StorageMap()
    for (j_idx, job) in pairs(jobs)
        push!(storagemap, j_idx => zeros(Int, length(products[job.product].operations)))
    end

    # Iterate through the storage map and insert items into the storage map
    for item in storeditems, (j_idx, job) in pairs(jobs)
        if job.product != item.product || item.quantity <= 0
            continue
        end

        itemchange = 0
        for o_idx in length(storagemap[j_idx]):-1:1
            if o_idx <= item.operation
                # Calculate the number of stored items used up by the job (number of items in storage map should not 
                # be greater than the order quantity for that job).
                if storagemap[j_idx][o_idx] + item.quantity <= job.quantity
                    change = item.quantity
                else
                    change = job.quantity - storagemap[j_idx][o_idx]
                end
                
                # Avoid negative change of quantity
                change = max(0, change)
                storagemap[j_idx][o_idx] += change
                if o_idx == item.operation
                    itemchange = change
                end
            end
        end

        # Subtract used quantity from the stored items
        item.quantity -= itemchange
    end

    return storagemap
end
