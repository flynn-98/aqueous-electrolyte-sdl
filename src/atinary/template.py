import scientia_sdk as sct
from typing import List
from .atinary_client import client, SDLABS_GROUP_NAME


sdlabs_api_client = client()


def show_templates():
    tpl_api = sct.TemplateApi(sdlabs_api_client)

    tpls = tpl_api.templates_list(group_id=SDLABS_GROUP_NAME).objects

    [print(template.name) for template in tpls]


def get_template_id_from_name(tpl_name):
    tpl_api = sct.TemplateApi(sdlabs_api_client)
    tpls = tpl_api.templates_list(group_id=SDLABS_GROUP_NAME).objects

    template_id = [tpl.id for tpl in tpls if tpl.name == tpl_name]

    return template_id


def get_parameters(workstation):
    # Since we can re-fine the parameter ranges in the template,
    #  we have to copy the parameters
    prm_api = sct.ParameterApi(sdlabs_api_client)
    tpl_params = [
        prm_api.parameter_copy(
            prm.id,
            parameter_copy=sct.ParameterCopy(name=prm.name),
        ).object
        for prm in workstation.parameters
    ]

    tpl_prms_dict = {
        prm.id: {
            "type": prm.type,
            "name": prm.name,
            "categories": [d.category for d in (prm.descriptors or [])],
            "low_value": prm.low_value,
            "high_value": prm.high_value,
            "stride": prm.stride,  # may be None
        }
        for prm in tpl_params
    }

    return tpl_prms_dict


def set_constraints(tpl_params, targets=2):
    # Equality constraint:
    # All numerical parameters in the template (i.e. that are not Categorical)
    #  will add up to the target (5) when suggested by the optimizer:
    eq_cstr_obj = sct.ConstraintObj(
        name="linear equality",
        type=sct.ConstraintType.LINEAR_EQ,
        targets=[targets],
        definitions=[
            # Weight is mandatory here (from 1 to 10, normalized internally)
            sct.ConstraintDefinition(parameter=prm.id, weight=1.0)
            for prm in tpl_params
            if prm.type != sct.ParameterType.CATEGORICAL
        ]
    )
    print("Constraint object: ",eq_cstr_obj)
    for prm in tpl_params:
        print("param type: ", prm.type)
    
    # Please note that all linear_* constraints work the same way as 'linear_eq',
    #  except Between, which requires 2 different targets;
    #  also, they all apply only to numerical parameters.

    cstr_api = sct.ConstraintApi(sdlabs_api_client)

    constraints: List[sct.ConstraintObjServer] = cstr_api.constraint_create_many(
        constraint_obj=[eq_cstr_obj]
    ).objects

    print(constraints)
    return constraints


def set_objectives(workstation):

    # Objective creation
    
    obj_api = sct.ObjectiveApi(sdlabs_api_client)
    
    objectives: List[sct.ObjectiveObjServer] = []
    
    obj_goal = sct.Goal.MAX  # can be 'MIN', 'MAX' or 'TARGET'


    wst_measurements = workstation.measurements
    objectives.append(
        obj_api.objective_create(
            objective_obj=sct.ObjectiveObj(
                name=wst_measurements[0],  # must match a workstation measurement
                description=f"{obj_goal.value}imize the {wst_measurements[0]}",
                goal=obj_goal,
            )
        ).object
    )
    
    obj_goal = sct.Goal.MAX  # can be 'MIN', 'MAX' or 'TARGET'
    objectives.append(
        obj_api.objective_create(
            objective_obj=sct.ObjectiveObj(
                name=wst_measurements[1],  # must match a workstation measurement
                description=f"{obj_goal.value}imize the {wst_measurements[1]}",
                goal=obj_goal,
            )
        ).object
    )
    
    mof_api = sct.MultiObjectiveFunctionApi(sdlabs_api_client)
    
    
    obj_kwargs = {}
    ## Create multi-objective function (can be Chimera or Weighted Sum)
    obj_kwargs["multi_objective_function"] = mof_api.mof_create(
        mof_obj=sct.MofObj(
            function="chimera",
            name=f"{SDLABS_GROUP_NAME} with Chimera",
            # chimera-specific configuration:
            #   Hierarchy starts at 0; the lower the value, the higher its priority
            #   Relative (from 0 to 100) or absolute values are the
            #     threshold after which the optimizer will start optimizing the next objective
            configuration=[
                sct.MofObjConfig(objective_id=objectives[0].id, relative=0.3, hierarchy=0),
                sct.MofObjConfig(objective_id=objectives[1].id, relative=0.7, hierarchy=1),
            ],
        )
    ).object.id
    return obj_kwargs
    

def set_optimiser():
    # Select an OPT_FUNCTION that can be changed according to the list
    # (list provided in the SDLabs documentation or in the call above)
    OPT_FUNCTION = "falcongpbo"
    # create optimizer configuration (here, select random seed and batch size)
    opt_configuration = [
        sct.OptObjConfiguration(key=property, value=str(val))
        for property, val in [
            # Configuration properties depend on the optimizer function
            ("batch_size", 1),  # should be less than the workstation bandwidth
            ("random_seed", 41),
        ]
    ]

    opt_api = sct.OptimizerApi(sdlabs_api_client)

    # Create optimizer (fetch id)
    opt_object = sct.OptObj(
        configuration=opt_configuration,
        function=OPT_FUNCTION,
        name=f"{SDLABS_GROUP_NAME}-{OPT_FUNCTION}",
    )
    opt_id = opt_api.optimizer_create(opt_obj=opt_object).object.id
    print(f"Optimizer {opt_id} created")

    return opt_id



def upload_template(workstation,
                    tpl_params, 
                    obj_kwargs, 
                    opt_id,
                    tpl_name,
                    tpl_budget,
                    cons):

    tpl_api = sct.TemplateApi(sdlabs_api_client)


    # Check that the template is not yet created
    tpls = tpl_api.templates_list(group_id=SDLABS_GROUP_NAME).objects
    template_id = next(
        (tpl.id for tpl in tpls if tpl.name == tpl_name),
        None,
    )





    if template_id:
        template = tpl_api.template_get(template_id).object
        print(f"Template found with id {template_id}. If you would like to create a new one, then use a new name")
    else:
        # create object
        tpl_obj = sct.TemplateObj(
            # budget: total number of objective function measurements allowed
            budget=tpl_budget,
            # define optimizer
            optimizer=opt_id,
            # name of the optimization template
            name=tpl_name,
            # define the objective(s)
            **obj_kwargs,
            # define the parameters for each workstation
            parameters=[
                sct.StepObj(
                    level=1,
                    parameters=[
                        sct.ParameterCpgObj(
                            parameter_id=prm.id,  # Copy of the workstation's parameter / parameters names should match!
                            workstation_id=workstation.id,
                        )
                        for prm in tpl_params
                    ],
                )
            ],
            constraints = [cons],
        )
        # create it
        template = tpl_api.template_create(template_obj=tpl_obj).object
    print("Template available!")
    return template
