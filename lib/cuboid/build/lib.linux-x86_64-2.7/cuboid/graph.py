from blocks.filter import VariableFilter
from blocks.roles import ALGORITHM_BUFFER
from blocks.graph import ComputationGraph
import logging
from blocks.select import Path
from blocks.filter import get_brick
from blocks.bricks.base import Brick
from blocks.model import Model

logger = logging.getLogger(__name__)


def parameter_stats(cg):
    observables = []
    for name, parameter in cg.get_parameter_dict().items():
        observables.append(
            parameter.norm(2).copy(name=name + "_norm"))
        observables.append(
            parameter.mean().copy(name=name + "_mean"))
        observables.append(
            parameter.var().copy(name=name + "_var"))
    return observables


def gradient_stats(cg, algorithm):
    observables = []
    for name, parameter in cg.get_parameter_dict().items():
        norm_param = algorithm.gradients[parameter].norm(2)
        observables.append(
            norm_param.copy(name=name + "_grad_norm"))
    return observables


def step_stats(cg, algorithm):
    observables = []
    for name, parameter in cg.get_parameter_dict().items():
        observables.append(
            algorithm.steps[parameter].norm(2).copy(name=name + "_step_norm"))
    return observables


def get_algorithm_parameters_dict(algorithm, model):
    name_to_var = model.get_parameter_dict()
    var_to_name = {v: k for k, v in name_to_var.items()}

    output_dict = dict()

    for val, update in algorithm.steps.items():
        cg = ComputationGraph([update])
        shared_to_save = VariableFilter(roles=[ALGORITHM_BUFFER])(cg)

        parent_name = var_to_name[val]
        for k in shared_to_save:
            output_dict[parent_name+"/"+k.name] = k
    return output_dict


def get_algorithm_parameters_values(algorithm, model):
    dd = get_algorithm_parameters_dict(algorithm, model)
    out = dict()
    for key, var in dd.items():
        out[key] = var.get_value()
    return out


def set_algorithm_parameters_values(algorithm, model, values_dict):
    parameters_dict = get_algorithm_parameters_dict(algorithm, model)
    unknown = set(values_dict) - set(parameters_dict)
    missing = set(parameters_dict) - set(values_dict)
    if len(unknown):
        logger.error("unknown parameter names: {}\n".format(unknown))
    if len(missing):
        logger.error("missing values for parameters: {}\n".format(missing))

    for name, value in values_dict.items():
        if name in parameters_dict:
            parameters_dict[name].set_value(value)


def _get_name(brick):
    if len(brick.parents) > 0:
        return _get_name(brick.parents[0]) + Path.BrickName(brick.name)
    elif len(brick.parents) == 0:
        return Path.BrickName(brick.name).part()
    else:
        raise ValueError("Only one parent per brick supported at "
                         "this time. (%s)" % str(brick))


def get_parameter_name(parameter):
    return "%s%s" % (_get_name(get_brick(parameter)),
                     Path.ParameterName(parameter).part())


def _get_children_bricks(brick):
    bricks = []
    for c in brick.children:
        bricks.extend(_get_children_bricks(c))
    return [brick] + bricks


def get_bricks(model):
    """ Return list of bricks
    Parameters:
    ---------
    model: blocks.brick.base.Brick or blocks.model.Model

    """
    bricks = []
    if isinstance(model, Model):
        for top in model.get_top_bricks():
            bricks.extend(_get_children_bricks(top))
    elif isinstance(model, Brick):
        bricks.extend(_get_children_bricks(model))
    else:
        raise AttributeError("No implementation for type %s" % type(model))
    return list(set(bricks))


def get_bricks_matching(model, predicate):
    return [b for b in get_bricks(model) if predicate(b)]
